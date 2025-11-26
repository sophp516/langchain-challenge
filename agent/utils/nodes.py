from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.types import RunnableConfig, interrupt
from langgraph.prebuilt import ToolNode
from pydantic import create_model
from utils.model import llm, llm_quality, evaluator_llm
from utils.subresearcher import subresearcher_graph
from utils.verification import verify_research_cross_references
from utils.db import save_report
from utils.configuration import get_config_from_configurable, tavily_client
from utils.tools import report_tools
import asyncio, json, re, uuid


def format_source_as_markdown_link(source: str) -> str:
    """
    Format a source as a proper markdown link.
    Handles:
    - Raw URLs: "https://example.com" -> "[example.com](https://example.com)"
    - Title (URL) format: "Title (https://example.com)" -> "[Title](https://example.com)"
    - Already formatted or other: return as-is
    """
    # Check if it's "Title (URL)" format
    title_url_match = re.match(r'^(.+?)\s*\((https?://[^\)]+)\)$', source)
    if title_url_match:
        title = title_url_match.group(1).strip()
        url = title_url_match.group(2)
        return f"[{title}]({url})"

    # Check if it's a raw URL
    if re.match(r'^https?://', source):
        # Extract domain for display text
        domain_match = re.match(r'https?://(?:www\.)?([^/]+)', source)
        display = domain_match.group(1) if domain_match else source
        return f"[{display}]({source})"

    # Return as-is if already formatted or unknown format
    return source


report_tool_node = ToolNode(report_tools)
llm_with_tools = llm.bind_tools(report_tools)


# ============================================================================
# NODES
# ============================================================================

async def check_user_intent(state: dict) -> dict:
    """
    Check user intent at entry point.
    Routes to either:
    - "new_research": User wants to research a topic
    - "retrieve_report": User wants to get a specific report
    - "list_reports": User wants to see available reports
    """
    topic = state.get("topic", "")
    messages = state.get("messages", [])

    # Get query from topic or last message
    query = topic
    if not query and messages:
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage):
                query = msg.content
                break

    if not query:
        return {"user_intent": "new_research"}

    print(f"check_user_intent: analyzing query='{query[:50]}...'")

    # Create structured output for intent classification
    IntentOutput = create_model(
        'IntentOutput',
        intent=(str, ...),
        extracted_report_id=(str, ...),
        confidence=(float, ...),
        reasoning=(str, ...)
    )

    structured_llm = llm.with_structured_output(IntentOutput)

    classification_prompt = f"""
    Analyze the user query and determine their intent.

    Query: "{query}"

    Classify into ONE category:

    1. "retrieve_report" - User wants to fetch/view a SPECIFIC report
       Examples: "show me report abc123", "get report_xyz", "what's in report abc?"

    2. "list_reports" - User wants to see available reports or versions
       Examples: "what reports do I have?", "show all reports", "list my reports"

    3. "new_research" - User wants to START NEW research (DEFAULT for ambiguous)
       Examples: "research AI", "tell me about climate change", any topic

    IMPORTANT: If unsure, default to "new_research".
    Extract report_id if mentioned (e.g., "report_abc123" → "report_abc123").
    """

    response = await structured_llm.ainvoke([
        SystemMessage(content="You classify user intent: retrieve_report, list_reports, or new_research."),
        HumanMessage(content=classification_prompt)
    ])

    intent = response.intent.lower().strip()
    report_id = response.extracted_report_id.strip() if response.extracted_report_id else ""

    # Validate intent
    valid_intents = ["retrieve_report", "list_reports", "new_research"]
    if intent not in valid_intents:
        intent = "new_research"

    print(f"check_user_intent: intent={intent}, report_id={report_id}, confidence={response.confidence:.2f}")

    return {
        "user_intent": intent,
        "intent_report_id": report_id
    }


async def call_report_tools(state: dict) -> dict:
    """
    COST OPTIMIZATION: Direct tool call creation based on intent (no LLM needed)
    Saves ~$0.01 per call and reduces latency by 500-1000ms
    """
    intent = state.get("user_intent", "")
    report_id = state.get("intent_report_id", "")
    topic = state.get("topic", "")

    print(f"call_report_tools: intent={intent}, report_id={report_id}")

    # Create tool call directly based on intent (no LLM needed!)
    tool_calls = []

    if intent == "retrieve_report":
        if not report_id:
            # No report_id provided, ask user
            return {
                "messages": [AIMessage(content="Please provide a report ID to retrieve. Example: report_abc123")]
            }
        tool_calls = [{
            "name": "get_report",
            "args": {"report_id": report_id},
            "id": "call_1"
        }]
    elif intent == "list_reports":
        tool_calls = [{
            "name": "list_all_reports",
            "args": {"limit": 20},
            "id": "call_1"
        }]
    else:
        # Unknown intent, should not happen
        return {
            "messages": [AIMessage(content=f"Unknown intent: {intent}")]
        }

    # Create AIMessage with tool_calls (mimics LLM response structure)
    from langchain_core.messages import AIMessage as AIMsg
    message = AIMsg(content="", tool_calls=tool_calls)

    print(f"call_report_tools: created {len(tool_calls)} tool call(s) directly (no LLM)")

    return {"messages": [message]}


async def execute_and_format_tools(state: dict) -> dict:
    """
    OPTIMIZED: Creates tool call, executes it, and formats response in one node.
    Replaces call_report_tools + execute_and_format_tools for ~50ms latency reduction.
    """
    intent = state.get("user_intent", "")
    report_id = state.get("intent_report_id", "")

    # Create tool call directly based on intent (same logic as call_report_tools)
    tool_calls = []

    if intent == "retrieve_report":
        if not report_id:
            return {
                "messages": [AIMessage(content="Please provide a report ID to retrieve. Example: report_abc123")]
            }
        tool_calls = [{
            "name": "get_report",
            "args": {"report_id": report_id},
            "id": "call_1"
        }]
    elif intent == "list_reports":
        tool_calls = [{
            "name": "list_all_reports",
            "args": {"limit": 20},
            "id": "call_1"
        }]
    else:
        return {
            "messages": [AIMessage(content=f"Unknown intent: {intent}")]
        }

    # Create AIMessage with tool_calls
    from langchain_core.messages import AIMessage as AIMsg
    tool_call_message = AIMsg(content="", tool_calls=tool_calls)

    # Execute tools with the tool call message
    temp_state = {**state, "messages": state.get("messages", []) + [tool_call_message]}
    tool_result = await report_tool_node.ainvoke(temp_state)
    merged_state = {**temp_state, **tool_result}
    messages = merged_state.get("messages", [])

    # Step 2: Format the tool response
    tool_result_data = None
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            try:
                tool_result_data = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
            except json.JSONDecodeError:
                tool_result_data = {"raw": msg.content}
            break

    if not tool_result_data:
        return {**tool_result, "messages": merged_state["messages"] + [AIMessage(content="No results found.")]}

    # Format based on result type
    if isinstance(tool_result_data, dict):
        if tool_result_data.get("error"):
            response_text = f"Error: {tool_result_data['error']}"
        elif tool_result_data.get("found") and tool_result_data.get("content"):
            # Format report with better visual hierarchy
            report_id = tool_result_data.get('report_id', 'Unknown')
            version_id = tool_result_data.get('version_id', 'N/A')
            created_at = tool_result_data.get('created_at', 'Unknown date')
            content = tool_result_data.get('content', 'No content available')

            response_text = f"""## 📄 Report: {report_id}

**Version:** {version_id}
**Created:** {created_at}

---

{content}"""
        elif tool_result_data.get("versions"):
            report_id = tool_result_data.get('report_id', 'Unknown')
            versions = tool_result_data.get("versions", [])
            versions_text = "\n\n".join([
                f"### Version {v['version_id']}\n**Created:** {v['created_at']}\n\n{v.get('content_preview', 'No preview available')}"
                for v in versions
            ])
            response_text = f"""## 📋 Versions of Report: {report_id}

{versions_text}"""
        elif tool_result.get("reports"):
            total_reports = tool_result_data.get('total_reports', 0)
            reports = tool_result_data.get("reports", [])
            reports_text = "\n".join([
                f"- **{r['report_id']}** | Version: {r['latest_version']} | Created: {r['created_at']}"
                for r in reports
            ])
            response_text = f"""## 📚 Available Reports ({total_reports})

{reports_text}"""
        elif tool_result_data.get("found") == False:
            response_text = tool_result_data.get("error", "Report not found.")
        else:
            response_text = f"Result: {json.dumps(tool_result_data, indent=2)}"
    else:
        response_text = str(tool_result_data)

    print(f"execute_and_format_tools: executed tools and formatted response")

    return {**tool_result, "messages": merged_state["messages"] + [AIMessage(content=response_text)]}


def route_after_intent_check(state: dict) -> str:
    """
    Route based on user intent.
    - new_research -> continue to research flow
    - retrieve_report/list_reports -> go to tools
    """
    intent = state.get("user_intent", "new_research")

    if intent in ["retrieve_report", "list_reports"]:
        print(f"route_after_intent_check: routing to 'tools' for intent={intent}")
        return "tools"
    else:
        print(f"route_after_intent_check: routing to 'research' for intent={intent}")
        return "research"


def should_continue_tools(state: dict) -> str:
    """
    Check if the last message has tool calls that need execution.
    """
    messages = state.get("messages", [])
    if not messages:
        return "end"

    last_message = messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print(f"should_continue_tools: has tool calls, routing to 'execute'")
        return "execute"

    print(f"should_continue_tools: no tool calls, routing to 'end'")
    return "end"


# ============================================================================
# NODES
# ============================================================================

def check_initial_context(state: dict, config: RunnableConfig) -> dict:
    """
    Check if initial topic has enough context to proceed.
    If sufficient, mark as finalized. If not, prepare for clarification loop.
    """
    agent_config = get_config_from_configurable(config.get("configurable", {}))

    topic = state.get("topic", "")
    print(f"check_initial_context: topic='{topic[:50]}...', max_clarification_rounds={agent_config.max_clarification_rounds}")

    # If clarification is disabled (max_clarification_rounds = 0), skip validation and proceed
    if agent_config.max_clarification_rounds == 0:
        print("check_initial_context: clarification disabled, proceeding to research")
        return {
            "is_finalized": True
        }

    if not topic or len(topic.strip()) == 0:
        print("check_initial_context: no topic found, returning is_finalized=False")
        return {
            "is_finalized": False
        }


    evaluation_prompt = f"""
    You are a research coordinator. Evaluate if the following is a valid research topic that can be researched.

    Topic: "{topic}"

    The topic is SUFFICIENT if it:
    1. Is a clear research question or subject (NOT a greeting, command, or casual message)
    2. Has enough specificity to start researching (subject + some constraint like time, place, category, etc.)
    3. Is understandable - you know what to look for

    The topic is INSUFFICIENT only if:
    - It's a greeting or casual message ("hi", "hello", "help me")
    - It's a single vague word with no context ("AI", "games", "technology")
    - It's a command with no subject ("research this", "tell me about")

    Examples of SUFFICIENT topics (should NOT ask clarification):
    - "MMORPG games coming out in 2026" (has subject + time constraint)
    - "Best programming languages for web development" (has subject + domain)
    - "Climate change effects on polar bears" (has subject + scope)
    - "AI in healthcare" (has subject + industry)

    Examples of INSUFFICIENT topics (should ask clarification):
    - "hi", "hello", "help me"
    - "games" (too vague, no constraint)
    - "tell me about stuff"

    Respond with ONLY "SUFFICIENT" or "INSUFFICIENT" followed by a brief reason (one sentence).
    """

    messages = [
        SystemMessage(content="You are a research coordinator. Accept any topic that has a clear subject and at least one constraint (time, place, category, scope). Only reject greetings and completely vague requests."),
        HumanMessage(content=evaluation_prompt)
    ]

    response = llm.invoke(messages)
    evaluation = response.content.strip() if hasattr(response, 'content') else str(response).strip()

    # Check if context is sufficient (must explicitly say SUFFICIENT)
    is_sufficient = evaluation.upper().startswith("SUFFICIENT")
    print(f"check_initial_context: evaluation={is_sufficient}, response='{evaluation[:150]}...'")

    return {
        "is_finalized": is_sufficient
    }


async def generate_clarification_question(state: dict, config: RunnableConfig) -> dict:
    """
    Generate a clarification question based on current topic and previous responses.
    This node uses LLM to determine what information is still needed.
    Also validates if we have enough context after receiving user responses.
    """
    agent_config = get_config_from_configurable(config.get("configurable", {}))

    clarification_rounds = state.get("clarification_rounds", 0)
    topic = state.get("topic", "")
    clarification_questions = state.get("clarification_questions", [])
    user_responses = state.get("user_responses", [])

    print(f"generate_clarification_question: round={clarification_rounds}, topic='{topic[:50]}...'")

    # If we've asked too many questions, finalize with what we have
    if clarification_rounds >= agent_config.max_clarification_rounds:
        print(f"generate_clarification_question: max rounds reached ({agent_config.max_clarification_rounds}), finalizing")
        # Build finalized topic from existing context
        finalized_topic = topic
        if user_responses:
            finalized_topic += " " + " ".join(user_responses)
        return {
            "is_finalized": True,
            "topic": finalized_topic.strip(),
            "clarification_rounds": clarification_rounds
        }

    # Build context with all Q&A so far
    context = f"Topic: {topic}\n"
    if clarification_questions and user_responses:
        context += "\nPrevious Q&A:\n"
        for i, (q, r) in enumerate(zip(clarification_questions, user_responses)):
            context += f"Q{i+1}: {q}\nA{i+1}: {r}\n"

    # Generate a clarification question OR determine if we have enough context
    clarification_prompt = f"""
    You are a research assistant helping to clarify a research topic.

    Current topic: {topic}

    {context if user_responses else "This is the initial topic."}

    CRITICAL RULES - ONLY ask for information that CANNOT be researched:
    1. DO NOT ask about information already present in the topic or previous responses
    2. DO NOT ask about standard metrics, criteria, definitions, or methodologies - these can be researched
    3. DO NOT ask "What metrics are used?" or "What criteria determine X?" - research can find this
    4. DO NOT ask "How is X measured?" or "What are standard practices?" - research can answer this
    5. DO NOT ask about definitions or technical terms - research can provide these

    ONLY ask for:
    - User-specific preferences (e.g., "Which specific items/entities do you want to compare?")
    - Missing constraints that are user choices (e.g., "What time period?" if not mentioned)
    - Ambiguities that require user input (e.g., "Do you mean X or Y?" when both are possible)

    If the topic has a clear subject, respond with "ENOUGH_CONTEXT"

    Examples:
    - Topic "most popular X of 2025" + user said "criteria A and criteria B" → ENOUGH_CONTEXT (agent can research what metrics are used)
    - Topic "MMORPG games in 2026" → ENOUGH_CONTEXT (has subject + time)
    - Topic "games" → Ask "What type of games or time period?" (missing constraint)
    - Topic "best practices" → Ask "Which domain or field?" (missing subject scope)

    Return ONLY the question (or "ENOUGH_CONTEXT"), nothing else.
    """

    messages = [
        SystemMessage(content="You are a helpful research assistant. You ONLY ask for information that requires user input and cannot be discovered through research. You do NOT ask about standard metrics, criteria, definitions, or methodologies - these can be researched. Be conservative - if the topic has a clear subject and constraints, mark it as ENOUGH_CONTEXT."),
        HumanMessage(content=clarification_prompt)
    ]

    response = await llm.ainvoke(messages)
    question = response.content.strip() if hasattr(response, 'content') else str(response).strip()

    # Check if we have enough context
    if question.upper() == "ENOUGH_CONTEXT" or "enough context" in question.lower():
        print("generate_clarification_question: LLM indicated enough context, finalizing")
        # Build finalized topic incorporating user responses
        finalized_topic = topic
        if user_responses:
            finalized_topic += " " + " ".join(user_responses)
        return {
            "is_finalized": True,
            "topic": finalized_topic.strip(),
            "clarification_rounds": clarification_rounds
        }

    # Not enough context yet, ask another question
    new_questions = clarification_questions + [question]
    print(f"generate_clarification_question: generated question='{question[:100]}...'")
    clarification_question = AIMessage(content=question)

    return {
        "messages": [clarification_question],
        "clarification_questions": new_questions,
        "clarification_rounds": clarification_rounds + 1
    }


def collect_user_response(state: dict) -> dict:
    """
    Collect user's response to the clarification question.
    Uses interrupt() to pause execution and wait for user input via Command(resume=...).
    """
    messages = state.get("messages", [])
    topic = state.get("topic", "")
    user_responses = state.get("user_responses", [])

    print(f"collect_user_response: waiting for user input (current topic: '{topic[:50]}...')")

    # Get the last AI message (the clarification question)
    last_message = messages[-1] if messages else None
    question = last_message.content if last_message else "Please provide your response"

    user_response = interrupt(question) # Need resume on chat function

    print(f"collect_user_response: received user response '{user_response[:50]}...'")

    new_responses = user_responses + [user_response]

    return {
        "user_responses": new_responses,
        "messages": [HumanMessage(content=user_response)]
    }



async def generate_subtopics(state: dict, config: RunnableConfig) -> dict:
    """
    Generate DISCOVERY research queries to find actual entities that answer the question.
    This is Phase 1: Broad research to discover what entities (games, products, people) exist.
    Phase 2 (outline generation) will organize these entities into sections.
    """

    agent_config = get_config_from_configurable(config.get("configurable", {}))

    topic = state.get("topic", "")
    print(f"generate_subtopics: starting discovery research for topic='{topic[:50]}...'")

    # Create structured output model for subtopics
    SubtopicsOutput = create_model(
        'SubtopicsOutput',
        subtopics=(list[str], ...)
    )

    structured_llm = llm_quality.with_structured_output(SubtopicsOutput)

    prompt = f"""
    Generate 2-3 BROAD search queries that are variations/rephrases of the main topic.
    These will do initial research to discover entities. Keep them broad and search-friendly.

    Topic: {topic}

    Generate queries that:
    - Rephrase the main topic in different ways
    - Use synonyms and alternative search terms
    - Include keywords like "top", "best", "list", "popular", "upcoming"

    Example for "mobile games":
    - "Best mobile games 2024"
    - "Top mobile gaming apps rankings"
    - "Most popular mobile games list"

    Example for "MMORPG 2026":
    - "Top MMORPGs releasing 2026"
    - "Best upcoming MMORPG launches"
    - "Most anticipated MMORPG games"

    Generate 2-3 broad queries for: "{topic}"
    """

    llm_response = await structured_llm.ainvoke([
        SystemMessage(content="Generate 2-3 broad search queries that are variations of the main topic. Keep them simple and search-friendly."),
        HumanMessage(content=prompt)
    ])

    broad_queries = llm_response.subtopics[:3]  # Limit to 3
    print(f"generate_subtopics: generated {len(broad_queries)} broad queries")
    for q in broad_queries:
        print(f"  - {q}")

    # Pass broad queries to subresearcher (which will do iterative deepening internally)
    async def process_broad_query(idx: int, query: str, main_topic: str):
        """Process a single broad query through subresearcher (which does iterative deepening)"""
        subgraph_state = {
            "subtopic_id": idx,
            "subtopic": query,  # Broad query
            "main_topic": main_topic,
            "other_subtopics": broad_queries,
            "research_results": {},
            "research_depth": 1,
            "source_credibilities": {},
            "max_search_results": agent_config.max_search_results,
            "max_research_depth": agent_config.max_research_depth,
            "search_api": agent_config.search_api
        }

        result = await subresearcher_graph.ainvoke(subgraph_state)

        return {
            "subtopic_id": idx,
            "subtopic": query,
            "research_results": result.get("research_results", {}),
            "source_credibilities": result.get("source_credibilities", {}),
            "research_depth": result.get("research_depth", 1)
        }

    tasks = [process_broad_query(idx, query, topic) for idx, query in enumerate(broad_queries)]
    print(f"generate_subtopics: sending {len(tasks)} broad queries to subresearcher for iterative deepening")
    sub_researchers = await asyncio.gather(*tasks)
    print(f"generate_subtopics: completed {len(sub_researchers)} subresearchers")

    for researcher in sub_researchers:
        depth = researcher.get("research_depth", 1)
        sources = len(researcher.get("research_results", {}))
        print(f"  - {researcher.get('subtopic', 'Unknown')}: {sources} sources, depth {depth}")

    return {
        "messages": [],  # No user-facing message
        "sub_researchers": [r for r in sub_researchers],
        "subtopics": broad_queries,  # Store broad queries as subtopics
    }



async def verify_cross_references(state: dict, config: RunnableConfig) -> dict:
    """
    Verify claims across multiple sources and identify conflicts.
    Uses verification.py functions to do the actual work.
    """

    agent_config = get_config_from_configurable(config.get("configurable", {}))

    # If cross-verification is disabled, skip this step
    if not agent_config.enable_cross_verification:
        print("verify_cross_references: cross-verification disabled, skipping")
        return {
            "verified_claims": [],
            "conflicting_info": []
        }

    sub_researchers = state.get("sub_researchers", [])
    topic = state.get("topic", "")

    # Use the main verification function from verification.py
    result = await verify_research_cross_references(topic, sub_researchers)

    verified_claims = result.get("verified_claims", [])
    conflicts = result.get("conflicting_info", [])

    # Create message about verification results
    verification_summary = f"Cross-reference verification complete:\n"
    verification_summary += f"- {len(verified_claims)} claims verified across multiple sources\n"
    if conflicts:
        verification_summary += f"- {len(conflicts)} conflicting pieces of information found\n"
        verification_summary += "\nConflicts will be noted in the report for transparency."

    return {
        "verified_claims": verified_claims,
        "conflicting_info": conflicts,
        "messages": [AIMessage(content=verification_summary)]
    }


async def generate_outline(state: dict) -> dict:
    """
    Generate a structured outline for the report based on research
    Maps subtopics to sections with specific focus areas
    """

    topic = state.get("topic", "")
    sub_researchers = state.get("sub_researchers", [])
    revision_count = state.get("revision_count", 0)

    print(f"generate_outline: creating outline for topic='{topic[:50]}...' (revision {revision_count})")

    # Build research overview
    research_overview = f"Topic: {topic}\n\nSubtopics researched:\n"
    for researcher in sub_researchers:
        subtopic = researcher.get("subtopic", "")
        sources = len(researcher.get("research_results", {}))
        research_overview += f"- {subtopic} ({sources} sources)\n"

    outline_prompt = f"""
    You are a report outline specialist. Create a structured outline that DIRECTLY ANSWERS the main research question.

    MAIN RESEARCH QUESTION: {topic}

    {research_overview}

    CRITICAL: Organize sections by THEMES and ANSWERS, not by subtopics.
    - If the same entity (person, song, product, etc.) appears in multiple subtopics' research, it should be discussed TOGETHER, not separately
    - Synthesize findings across ALL subtopics to identify the KEY ANSWERS to the main question
    - Each section should pull relevant information from ANY subtopic that has relevant data

    Example: If researching "most popular K-pop songs 2025":
    - BAD: Section per subtopic (Charts section, Streaming section, Social Media section)
    - GOOD: Section per answer (Top Songs section listing actual songs with combined evidence from charts + streaming + social)

    Required structure:
    1. Executive Summary - Directly answer the main question with specific names/titles/rankings
    2. Main sections (2-4 sections) organized by THEMES that answer the question:
       - Each section should synthesize findings from ALL relevant subtopics
       - Focus on WHAT the answer is, not WHERE the data came from
       - Include specific names, titles, rankings prominently
    3. Supporting Analysis - Context, methodology, factors behind the findings
    4. Conclusion - Concise final answer to the main research question

    For each section, specify:
    - Section title (focused on answering the main question)
    - Key questions to address (what specific answers should this section provide?)
    - Which subtopics have relevant data (for reference, but section should synthesize across all)

    Return as a JSON object with this structure:
    {{
      "sections": [
        {{
          "title": "Section Title",
          "subtopics": ["subtopic1", "subtopic2"],
          "key_questions": ["question1", "question2", "question3"]
        }}
      ]
    }}
    """

    messages = [
        SystemMessage(content="You are an expert report outline specialist. Create comprehensive, well-structured outlines that ensure complete coverage, logical flow, and clear organization. Focus on creating outlines that will result in high-quality reports with excellent coverage, evidence, structure, and clarity."),
        HumanMessage(content=outline_prompt)
    ]

    response = await llm.ainvoke(messages)
    response_text = response.content if hasattr(response, 'content') else str(response)

    # Parse JSON outline (simplified)
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            outline = json.loads(json_match.group())
        else:
            # Fallback: simple outline based on subtopics
            outline = {
                "sections": [
                    {"title": "Introduction", "subtopics": [], "key_questions": []},
                    *[{"title": r.get("subtopic", ""), "subtopics": [r.get("subtopic", "")], "key_questions": []}
                      for r in sub_researchers],
                    {"title": "Conclusion", "subtopics": [], "key_questions": []}
                ]
            }
    except Exception as e:
        print(f"generate_outline: error parsing JSON, using fallback: {e}")
        outline = {
            "sections": [
                {"title": "Introduction", "subtopics": [], "key_questions": []},
                *[{"title": r.get("subtopic", ""), "subtopics": [r.get("subtopic", "")], "key_questions": []}
                  for r in sub_researchers],
                {"title": "Conclusion", "subtopics": [], "key_questions": []}
            ]
        }

    print(f"generate_outline: created outline with {len(outline.get('sections', []))} sections")

    # DEBUG: Log the outline to see what subtopics are mapped
    for section in outline.get("sections", []):
        print(f"  Section: '{section.get('title', '')}' -> Subtopics: {section.get('subtopics', [])}")

    return {
        "report_outline": outline,
        "report_sections": []
    }


async def search_for_section_sources(section_title: str, topic: str, num_results: int = 3) -> list[dict]:
    """
    Perform a web search to find additional sources for a section.
    Used when existing research doesn't provide enough sources.
    """
    search_query = f"{topic} {section_title}"
    print(f"    Searching for additional sources: '{search_query[:50]}...'")

    try:
        loop = asyncio.get_event_loop()
        search_results = await loop.run_in_executor(
            None,
            lambda: tavily_client.search(query=search_query, max_results=num_results)
        )
        return search_results.get("results", [])
    except Exception as e:
        print(f"    Search failed: {e}")
        return []


async def write_single_section(
    section: dict,
    topic: str,
    research_by_subtopic: dict,
    min_credibility_score: float,
    all_sections: list = None,
    section_index: int = 0,
    previously_written_sections: list = None
) -> dict:
    """
    Write a single section of the report with citations.
    SEQUENTIAL WRITING: This function now receives previously written sections to maintain context.
    If insufficient sources exist, performs additional web search.
    Includes full context of previous sections for sequential, cohesive writing.
    """
    section_title = section.get("title", "")
    section_subtopics = section.get("subtopics", [])
    all_sections = all_sections or []
    previously_written_sections = previously_written_sections or []
    MIN_SOURCES_THRESHOLD = 2  # Minimum sources needed before searching for more

    print(f"  Writing section {section_index + 1}/{len(all_sections)}: {section_title}")

    # Gather relevant research for this section
    relevant_research = ""
    sources_list = []
    for subtopic in section_subtopics:
        if subtopic in research_by_subtopic:
            results = research_by_subtopic[subtopic]["results"]
            credibilities = research_by_subtopic[subtopic]["credibilities"]

            # Filter sources by credibility and take top 5 for better coverage
            credible_sources = [(source, findings) for source, findings in results.items()
                              if credibilities.get(source, 0.5) >= min_credibility_score]

            for source, findings in credible_sources[:5]:  # Top 5 credible sources per subtopic for better evidence
                credibility = credibilities.get(source, 0.5)
                relevant_research += f"\nSource: {source} (credibility: {credibility:.2f})\n{findings}\n"
                sources_list.append(source)

    # If insufficient sources, search for more
    if len(sources_list) < MIN_SOURCES_THRESHOLD:
        print(f"    Section '{section_title}' has only {len(sources_list)} sources, searching for more...")
        additional_results = await search_for_section_sources(section_title, topic)

        for result in additional_results:
            source_url = result.get("url", "")
            source_title = result.get("title", "Untitled")
            source_content = result.get("content", "")

            if source_content:
                source_key = f"{source_title} ({source_url})"
                relevant_research += f"\nSource: {source_key} (credibility: 0.70)\n{source_content[:500]}...\n"
                sources_list.append(source_key)

        print(f"    Now have {len(sources_list)} sources for section")

    # If still no research, note it
    if not relevant_research:
        relevant_research = "Limited sources available. Provide general analysis based on the topic."

    # Build sequential context - include actual content from previously written sections
    sequential_context = ""
    if previously_written_sections:
        sequential_context = "\n**PREVIOUSLY WRITTEN SECTIONS - Build on this content:**\n"
        sequential_context += f"You are writing Section {section_index + 1} of {len(all_sections)}. The reader has already read:\n\n"

        for i, prev_section in enumerate(previously_written_sections):
            prev_title = prev_section.get("title", "")
            prev_content = prev_section.get("content", "")

            # Include a summary of previous section content (first 800 chars to maintain context)
            content_preview = prev_content[:800] + "..." if len(prev_content) > 800 else prev_content

            sequential_context += f"--- Section {i + 1}: {prev_title} ---\n{content_preview}\n\n"

        sequential_context += """**YOUR SECTION MUST:**
- Reference and build upon specific facts, names, and details mentioned in previous sections
- Continue the narrative flow - don't start from scratch as if nothing was written before
- Add NEW information that extends or deepens the analysis (not repeat what was already said)
- Use phrases like "As mentioned in [previous section]...", "Building on the [X] discussed above...", "These [entities] face..."
- If previous sections introduced specific entities (games, companies, people, products), reference them by name in your section
- Focus on answering the MAIN RESEARCH QUESTION with specific, concrete details from your sources
"""

    # Build outline context for remaining sections
    outline_context = ""
    if all_sections and section_index < len(all_sections) - 1:
        sections_after = all_sections[section_index + 1:]
        outline_context = "\n**SECTIONS COMING AFTER YOURS (don't cover their content):**\n"
        for i, s in enumerate(sections_after):
            s_title = s.get("title", "")
            s_questions = s.get("key_questions", [])
            outline_context += f"  {section_index + 2 + i}. \"{s_title}\""
            if s_questions:
                outline_context += f" - will cover: {', '.join(s_questions[:2])}"
            outline_context += "\n"

    section_prompt = f"""
    Write the "{section_title}" section of a research report on: {topic}

    {sequential_context}
    {outline_context}

    SOURCES (numbered for citation):
    {relevant_research[:5000]}

    **CRITICAL ANTI-HALLUCINATION RULES - VIOLATIONS ARE UNACCEPTABLE:**
    1. ONLY write about information that EXPLICITLY appears in the SOURCES above - word for word verification
    2. Every factual claim (names, dates, numbers, titles, events, statistics) MUST have a citation [1], [2], etc.
    3. DO NOT combine information from different sources to create new facts
    4. DO NOT infer or assume connections between facts that aren't stated in sources
    5. DO NOT use your knowledge to add information not in the sources
    6. DO NOT mention specific entities, dates, rankings, or statistics unless they appear EXACTLY in the sources
    7. If a source mentions "Entity X" but doesn't mention it ranked #1, DO NOT say it was #1
    8. If sources mention different entities separately, DO NOT combine them into a single narrative unless sources explicitly do so
    9. If you're unsure whether something is in the sources, LEAVE IT OUT
    10. It's better to write less content that is accurate than more content that is fabricated

    **VERIFICATION PROCESS:**
    - Before writing each sentence, check: "Is this EXACTLY stated in the sources?"
    - For specific claims (entity names, dates, rankings, statistics): Quote or closely paraphrase the source
    - If you can't find a source for a claim, DO NOT include it

    **WHAT TO DO IF SOURCES ARE LIMITED:**
    - Focus only on what IS explicitly mentioned in the sources
    - Use phrases like "According to [source]..." or "Source [X] states..."
    - Acknowledge gaps: "Available research focuses on X, while Y remains less documented"
    - DO NOT fill gaps with invented information, even if you think you know the answer

    Write a high-quality, well-structured section that excels in:
    
    **COVERAGE:**
    - Comprehensively address all aspects of the section topic
    - Cover key questions and subtopics assigned to this section
    - Include relevant details, examples, and context from sources
    - Don't leave important aspects unaddressed

    **INCLUDE SPECIFIC DETAILS FROM SOURCES:**
    - List all specific NAMES mentioned (people, artists, companies, products)
    - List all specific TITLES mentioned (songs, albums, movies, games, etc.)
    - Include RANKINGS with positions (if source says "top 5", list all 5 items)
    - Include NUMBERS and STATISTICS (sales, views, chart positions, dates)
    - DO NOT summarize lists as "various X" or "several Y" - name them specifically
    - If sources rank items, include ALL ranked items with their exact positions

    **EVIDENCE:**
    - Every factual claim must have a citation [1], [2], etc.
    - Use multiple sources to support important claims when available
    - Integrate citations naturally into the text
    - Provide specific examples and data from sources
    
    **STRUCTURE:**
    - Start with a clear topic sentence that introduces the section's focus
    - Organize content logically (chronological, thematic, or by importance)
    - Use smooth transitions between paragraphs
    - Build arguments progressively, leading to key insights
    - End paragraphs with sentences that connect to the next idea
    
    **CLARITY:**
    - Write in clear, professional academic style
    - Use precise language and avoid vague statements
    - Explain technical terms when first introduced
    - Write concise, well-crafted sentences
    - Ensure each paragraph has a single clear focus
    - Use active voice when appropriate for readability
    
    **LENGTH:**
    - Write 3-5 substantial paragraphs (aim for 200-400 words)
    - Provide sufficient detail to comprehensively cover the topic
    - Balance depth with readability
    
    IMPORTANT: Do NOT include the section title. Start directly with the content.
    """

    # Special handling for Executive Summary and Conclusion sections
    is_executive_summary = "executive summary" in section_title.lower()
    is_conclusion = "conclusion" in section_title.lower() or "recommendations" in section_title.lower()
    
    if is_executive_summary:
        section_prompt += "\n\n**EXECUTIVE SUMMARY REQUIREMENTS:**\n- Provide a concise, high-level overview of the entire report\n- Highlight the most important findings and conclusions\n- Summarize key insights from all sections\n- Keep it brief but comprehensive (2-3 paragraphs)\n- Write for a busy executive audience - clear and impactful"
    elif is_conclusion:
        section_prompt += "\n\n**CONCLUSION REQUIREMENTS:**\n- Synthesize findings from all sections\n- Draw clear, evidence-based conclusions\n- Provide actionable recommendations when applicable\n- Connect back to the research objectives\n- End with forward-looking statements or implications"
    
    messages = [
        SystemMessage(content="You are an expert research report writer. CRITICAL: You MUST only write information that explicitly appears in the provided sources. DO NOT use your knowledge to add facts, names, dates, or statistics. Every factual claim must be directly traceable to a source. Hallucination is unacceptable. Additionally, focus on writing high-quality, well-structured content that excels in coverage, evidence, structure, and clarity."),
        HumanMessage(content=section_prompt)
    ]

    response = await llm_quality.ainvoke(messages)
    section_content = response.content if hasattr(response, 'content') else str(response)


    return {
        "title": section_title,
        "content": section_content,
        "sources": sources_list,
        "subtopics": section_subtopics
    }




async def write_sections_with_citations(state: dict, config: RunnableConfig) -> dict:
    """
    Write each section of the report with proper inline citations in parallel.
    Uses research results to build evidence-based sections.
    """
    agent_config = get_config_from_configurable(config.get("configurable", {}))

    topic = state.get("topic", "")
    outline = state.get("report_outline", {})
    sub_researchers = state.get("sub_researchers", [])
    revision_count = state.get("revision_count", 0)

    print(f"write_sections_with_citations: writing sections for topic='{topic[:50]}...' (revision {revision_count})")
    print(f"write_sections_with_citations: using min_credibility_score={agent_config.min_credibility_score}")

    sections = outline.get("sections", [])

    # Build research lookup by subtopic
    research_by_subtopic = {}
    for researcher in sub_researchers:
        subtopic = researcher.get("subtopic", "")
        research_by_subtopic[subtopic] = {
            "results": researcher.get("research_results", {}),
            "credibilities": researcher.get("source_credibilities", {})
        }

    print(f"write_sections_with_citations: Available research keys:")
    for key in research_by_subtopic.keys():
        num_sources = len(research_by_subtopic[key]["results"])
        print(f"  '{key}': {num_sources} sources")

    # Write sections one at a time so each has context from previous sections
    print(f"write_sections_with_citations: writing {len(sections)} sections SEQUENTIALLY for context flow")
    written_sections = []

    for idx, section in enumerate(sections):
        # Pass all previously written sections so this section can reference them
        section_result = await write_single_section(
            section=section,
            topic=topic,
            research_by_subtopic=research_by_subtopic,
            min_credibility_score=agent_config.min_credibility_score,
            all_sections=sections,
            section_index=idx,
            previously_written_sections=written_sections  # Pass previously completed sections
        )
        written_sections.append(section_result)
        print(f"  Completed section {idx + 1}/{len(sections)}: {section_result.get('title', 'Unknown')}")

    print(f"write_sections_with_citations: completed all {len(written_sections)} sections sequentially")

    # Combine sections into full report
    full_report = f"# {topic}\n\n"
    all_sources = []

    for section in written_sections:
        full_report += f"## {section['title']}\n\n{section['content']}\n\n"
        all_sources.extend(section['sources'])

    # Add references section with properly formatted markdown links
    unique_sources = list(dict.fromkeys(all_sources))  # Remove duplicates, preserve order
    full_report += "## References\n\n"
    for i, source in enumerate(unique_sources, 1):
        formatted_source = format_source_as_markdown_link(source)
        full_report += f"[{i}] {formatted_source}\n"

    full_report_message = AIMessage(content=full_report)

    # Generate report_id if not exists
    report_id = state.get("report_id", "")
    if not report_id:
        report_id = f"{uuid.uuid4().hex[:12]}"
        print(f"write_sections_with_citations: generated new report_id={report_id}")

    new_version_id = state.get("version_id", 0) + 1

    # Save report to MongoDB
    try:
        await save_report(report_id, new_version_id, full_report)
        print(f"write_sections_with_citations: saved report {report_id} version {new_version_id} to MongoDB")
    except Exception as e:
        print(f"write_sections_with_citations: failed to save report to MongoDB: {e}")

    return {
        "report_id": report_id,
        "report_sections": written_sections,
        "report_content": full_report,
        "report_references": unique_sources,
        "version_id": new_version_id,
        "report_history": state.get("report_history", []) + [new_version_id],
        "messages": [full_report_message]
    }


async def research_outline_and_write(state: dict, config: RunnableConfig) -> dict:
    """
    Merged node: Perform research, verification, outline, and writing.
    Combines generate_subtopics, verify_cross_references, generate_outline, and write_sections_with_citations.
    """
    print("research_outline_and_write: Starting combined research and writing process")
    
    # 1. Generate Subtopics & Research
    subtopics_result = await generate_subtopics(state, config)
    state = {**state, **subtopics_result}
    
    # Verify Cross References
    verify_result = await verify_cross_references(state, config)
    state = {**state, **verify_result}
    
    # Generate Outline
    outline_result = await generate_outline(state)
    state = {**state, **outline_result}
    
    # Write Sections
    write_result = await write_sections_with_citations(state, config)

    # SEQUENTIAL MESSAGES FIX: Only return the final report message for clean LangSmith chat
    # Intermediate messages (subtopics alert, etc.) are skipped to avoid multiple simultaneous messages
    final_message = None
    if "messages" in write_result and write_result["messages"]:
        final_message = write_result["messages"][-1]  # Take only the last message (full report)

    print("research_outline_and_write: Completed combined process")

    return {
        **subtopics_result,
        **verify_result,
        **outline_result,
        **write_result,
        "messages": [final_message] if final_message else []  # Only ONE message
    }


async def evaluate_report(state: dict) -> dict:
    """
    Evaluate a report based on content quality, structure, and evidence
    Provides detailed feedback for potential improvements
    """
    report_content = state.get("report_content", "")
    current_report_id = state.get("current_report_id", 0)
    topic = state.get("topic", "")

    print(f"evaluate_report: starting for report_id={current_report_id}")

    evaluation_prompt = f"""
    You are an expert research report evaluator. Evaluate the following report on: {topic}

    Report:
    {report_content[:5000]}  # Limit for token management

    Evaluate based on:
    1. **Coverage** (0-25): Does it comprehensively cover the topic?
    2. **Evidence** (0-25): Are claims well-supported with sources?
    3. **Structure** (0-25): Is it well-organized and logical?
    4. **Clarity** (0-25): Is it clear and well-written?

    Provide your evaluation in this exact format:
    COVERAGE: [score]/25
    EVIDENCE: [score]/25
    STRUCTURE: [score]/25
    CLARITY: [score]/25
    TOTAL: [sum of above]/100
    FEEDBACK: [One paragraph of constructive feedback on what could be improved]
    """

    messages = [
        SystemMessage(content="You are an expert research report evaluator that provides detailed, constructive feedback. Be strict but fair in your scoring."),
        HumanMessage(content=evaluation_prompt)
    ]

    # Use external evaluator (Gemini) for unbiased evaluation
    print("evaluate_report: using external evaluator (Gemini) for unbiased scoring")
    response = await evaluator_llm.ainvoke(messages)
    response_text = response.content if hasattr(response, 'content') else str(response)

    # Extract total score
    total_match = re.search(r'TOTAL:\s*(\d+)', response_text)
    score = int(total_match.group(1)) if total_match else 75

    # Extract feedback text (everything after "FEEDBACK:" to end of text)
    feedback_match = re.search(r'FEEDBACK:\s*(.+)', response_text, re.DOTALL)
    feedback_text = feedback_match.group(1).strip() if feedback_match else "N/A"

    # Log feedback (truncate if too long)
    feedback_preview = feedback_text[:200] + "..." if len(feedback_text) > 200 else feedback_text
    print(f"evaluate_report: feedback={feedback_preview}")

    scores = state.get("scores", {})
    scores[current_report_id] = score

    print(f"evaluate_report: score={score}/100 for report_id={current_report_id}")

    return {
        "scores": scores,
        "final_score": score,
        "evaluator_feedback": feedback_text  # Store feedback for revision context
    }


async def revise_sections(state: dict, config: RunnableConfig) -> dict:
    """
    Revise all sections based on evaluator feedback from Gemini.
    Uses the evaluator's qualitative feedback to guide improvements.
    """
    agent_config = get_config_from_configurable(config.get("configurable", {}))

    topic = state.get("topic", "")
    report_sections = state.get("report_sections", [])
    sub_researchers = state.get("sub_researchers", [])
    revision_count = state.get("revision_count", 0)
    evaluator_feedback = state.get("evaluator_feedback", "")

    print(f"revise_sections: starting revision based on evaluator feedback (revision {revision_count})")
    print(f"revise_sections: feedback preview: {evaluator_feedback[:200]}...")

    if not evaluator_feedback or evaluator_feedback == "N/A":
        print("revise_sections: no evaluator feedback available, returning unchanged")
        return {}

    # Build research lookup (same as write_sections_with_citations)
    research_by_subtopic = {}
    for researcher in sub_researchers:
        subtopic = researcher.get("subtopic", "")
        research_by_subtopic[subtopic] = {
            "results": researcher.get("research_results", {}),
            "credibilities": researcher.get("source_credibilities", {})
        }

    # Revise all sections based on evaluator feedback
    print(f"revise_sections: revising all {len(report_sections)} sections")
    tasks = []

    for section in report_sections:
        section_title = section.get("title", "")
        print(f"  Revising section: {section_title}")

        # Create a modified section dict with subtopics for research lookup
        section_dict = {
            "title": section_title,
            "subtopics": section.get("subtopics", []),
            "key_questions": section.get("key_questions", []),
            "original_content": section.get("content", "")
        }

        tasks.append(revise_single_section(
            section_dict,
            topic,
            research_by_subtopic,
            agent_config.min_credibility_score,
            evaluator_feedback  # Pass evaluator feedback instead of section-specific gaps
        ))

    # Execute revisions in parallel
    revised_sections = await asyncio.gather(*tasks)

    print(f"revise_sections: completed revision of {len(revised_sections)} sections")

    # Increment revision count
    new_revision_count = revision_count + 1

    # Rebuild full report from sections
    full_report = f"# {topic}\n\n"
    all_sources = []

    for section in revised_sections:
        full_report += f"## {section['title']}\n\n{section['content']}\n\n"
        all_sources.extend(section.get('sources', []))

    # Add references section with properly formatted markdown links
    unique_sources = list(dict.fromkeys(all_sources))
    full_report += "## References\n\n"
    for i, source in enumerate(unique_sources, 1):
        formatted_source = format_source_as_markdown_link(source)
        full_report += f"[{i}] {formatted_source}\n"

    # Save to MongoDB
    report_id = state.get("report_id", "")
    new_version_id = state.get("version_id", 0) + 1
    messages = state.get("messages", [])

    try:
        await save_report(report_id, new_version_id, full_report)
        print(f"revise_sections: saved report {report_id} version {new_version_id}")
        messages = messages + [AIMessage(content=f"revise_sections: saved report {report_id} version {new_version_id}")]

    except Exception as e:
        print(f"revise_sections: failed to save report to MongoDB: {e}")

    return {
        "report_sections": revised_sections,
        "report_content": full_report,
        "report_references": unique_sources,
        "version_id": new_version_id,
        "report_history": state.get("report_history", []) + [new_version_id],
        "revision_count": new_revision_count,
        "messages": messages
    }


async def search_based_on_feedback(
    section_title: str,
    topic: str,
    evaluator_feedback: str
) -> dict:
    """
    Extract search needs directly from Gemini evaluator feedback and search for specific information.

    Returns dict of {source_key: source_content} for additional sources found.
    """
    # Look for keywords indicating specific information needs
    feedback_lower = evaluator_feedback.lower()

    # Keywords that indicate need for additional search
    search_indicators = [
        "missing", "lacks", "needs more", "insufficient", "not enough",
        "should include", "could benefit from", "add more", "expand on",
        "more detail", "more information", "more specific", "more examples"
    ]

    needs_search = any(indicator in feedback_lower for indicator in search_indicators)

    if not needs_search:
        print(f"      No search indicators in feedback for '{section_title}'")
        return {}

    # Extract what needs to be searched from the feedback
    # Build targeted search query based on section title + topic + feedback hints
    search_query = f"{topic} {section_title}"

    # Look for specific topics mentioned after search indicators
    for indicator in search_indicators:
        if indicator in feedback_lower:
            # Find text after the indicator (next 50 chars)
            idx = feedback_lower.find(indicator)
            context = evaluator_feedback[idx:idx+100]
            # Add context to search
            search_query += f" {context}"
            break

    print(f"      Searching for additional info based on feedback: '{search_query[:80]}...'")

    # Execute search
    additional_sources = {}
    try:
        loop = asyncio.get_event_loop()
        search_results = await loop.run_in_executor(
            None,
            lambda: tavily_client.search(query=search_query, max_results=3)
        )

        results = search_results.get("results", [])
        print(f"      Found {len(results)} additional sources")

        for result in results:
            source_url = result.get("url", "")
            source_title = result.get("title", "Untitled")
            source_content = result.get("content", "")

            if source_content:
                source_key = f"{source_title} ({source_url})"
                additional_sources[source_key] = source_content[:800]

    except Exception as e:
        print(f"      Search failed: {e}")

    return additional_sources


async def revise_single_section(
    section: dict,
    topic: str,
    research_by_subtopic: dict,
    min_credibility_score: float,
    evaluator_feedback: str
) -> dict:
    """
    Revise a single section based on evaluator feedback from Gemini.
    Uses the original content as a base and improves it according to the feedback.

    Two-phase approach:
    1. Analyze feedback to identify specific information gaps
    2. Search for that specific information if needed
    3. Make targeted revisions with the additional context
    """
    section_title = section.get("title", "")
    original_content = section.get("original_content", "")
    section_subtopics = section.get("subtopics", [])

    print(f"    Revising: {section_title}")

    # Gather relevant research from existing sources
    relevant_research = ""
    sources_list = []

    for subtopic in section_subtopics:
        if subtopic in research_by_subtopic:
            results = research_by_subtopic[subtopic]["results"]
            credibilities = research_by_subtopic[subtopic]["credibilities"]

            for source, findings in list(results.items())[:5]:  # Top 5 sources for better coverage
                if credibilities.get(source, 0.5) >= min_credibility_score:
                    relevant_research += f"\nSource: {source}\n{findings}\n"
                    sources_list.append(source)

    # PHASE 1: Search for additional information based on Gemini feedback
    additional_sources = await search_based_on_feedback(
        section_title=section_title,
        topic=topic,
        evaluator_feedback=evaluator_feedback
    )

    # Add the additional sources to our research pool
    if additional_sources:
        print(f"      Found {len(additional_sources)} additional targeted sources")
        for source_key, source_content in additional_sources.items():
            relevant_research += f"\nSource (targeted search): {source_key}\n{source_content}\n"
            sources_list.append(source_key)

    revision_prompt = f"""
    You are revising the "{section_title}" section of a research report on: {topic}

    ORIGINAL CONTENT:
    {original_content}

    **EVALUATOR FEEDBACK (address these improvement areas):**
    {evaluator_feedback}

    SOURCES FOR REVISION:
    {relevant_research[:3000]}

    **STRICT GROUNDING RULES:**
    1. ONLY add information that appears in the SOURCES above
    2. Every new claim MUST cite a source
    3. DO NOT invent names, titles, dates, statistics, or facts not in sources
    4. DO NOT use placeholder names like "XYZ", "Company A", etc.
    5. If sources don't provide enough info to address feedback, acknowledge the limitation
    6. It's better to make conservative improvements than to fabricate information

    **REVISION APPROACH - INCREMENTAL FIXES ONLY:**
    - This is a REVISION, not a rewrite - preserve the structure and flow of the original
    - Make MINIMAL, TARGETED changes to address specific feedback points
    - Keep all accurate content unchanged - only fix what needs fixing
    - Add missing information where gaps are identified
    - Clarify confusing parts without restructuring entire paragraphs
    - Remove or correct inaccurate claims
    - Add citations [1], [2] only for new claims
    - If the original content is mostly good, make small additions/corrections rather than rewriting
    - If sources don't support addressing specific feedback, note: "Additional research needed on [topic]"

    Return ONLY the revised section content (no title/header).
    """

    messages = [
        SystemMessage(content="You are a research report editor who improves sections based on evaluator feedback while maintaining strict adherence to sources. Never fabricate information."),
        HumanMessage(content=revision_prompt)
    ]

    response = await llm_quality.ainvoke(messages)
    revised_content = response.content if hasattr(response, 'content') else str(response)

    return {
        "title": section_title,
        "content": revised_content,
        "sources": sources_list,
        "subtopics": section_subtopics
    }


def display_final_report(state: dict) -> dict:
    """
    Display the final report to the user after evaluation passes.
    This is a separate node to ensure the report is shown before the feedback prompt.
    """
    report_content = state.get("report_content", "")
    final_score = state.get("final_score", 0)

    print(f"display_final_report: report_content: {report_content}")

    if not report_content:
        return {"messages": []}

    return {
        "messages": [AIMessage(content=report_content)]
    }


def prompt_for_feedback(state: dict) -> dict:
    """
    Display the feedback prompt to the user as an AI message.
    This runs after display_final_report and before collect_user_feedback.
    """
    user_feedback_list = state.get("user_feedback", [])

    print(f"prompt_for_feedback: displaying prompt (round {len(user_feedback_list) + 1})")

    feedback_prompt = f"""I've completed the report.

Would you like me to:
- Expand on any specific areas?
- Add more detail to certain topics?
- Clarify any points?
- Make any other improvements?

Please provide your feedback, or type 'done' if the report is satisfactory."""

    return {
        "messages": [AIMessage(content=feedback_prompt)]
    }


def collect_user_feedback(state: dict) -> dict:
    """
    Collect user feedback on the report after initial generation.
    Uses interrupt to pause and wait for user input.
    Appends feedback to list for tracking multiple rounds of feedback.
    """
    user_feedback_list = state.get("user_feedback", [])
    feedback_incorporated_list = state.get("feedback_incorporated", [])

    print(f"collect_user_feedback: waiting for user feedback (round {len(user_feedback_list) + 1})")

    # Interrupt and wait for user feedback (no prompt, it was already displayed)
    user_input = interrupt("Waiting for user feedback...")

    print(f"collect_user_feedback: received feedback '{user_input[:50]}...'")

    # Check if user is satisfied
    is_done = user_input.lower().strip() in ['done', 'no', 'none', 'looks good', 'satisfied', 'good']

    # Append feedback to list
    new_feedback_list = user_feedback_list + [user_input]
    new_incorporated_list = feedback_incorporated_list + [is_done]

    return {
        "user_feedback": new_feedback_list,
        "feedback_incorporated": new_incorporated_list,
        "messages": [HumanMessage(content=user_input)]
    }


async def incorporate_feedback(state: dict) -> dict:
    """
    Incorporate user feedback holistically across the entire report.
    Processes all unincorporated feedback from the feedback list.
    """
    user_feedback_list = state.get("user_feedback", [])
    feedback_incorporated_list = state.get("feedback_incorporated", [])
    report_content = state.get("report_content", "")
    sub_researchers = state.get("sub_researchers", [])

    # Find unincorporated feedback
    unincorporated_feedback = []
    for i, (feedback, is_incorporated) in enumerate(zip(user_feedback_list, feedback_incorporated_list)):
        if not is_incorporated:
            unincorporated_feedback.append(feedback)

    print(f"incorporate_feedback: processing {len(unincorporated_feedback)} unincorporated feedback items")

    if len(unincorporated_feedback) == 0:
        return {
            "feedback_incorporated": feedback_incorporated_list,
            "messages": [AIMessage(content="All feedback has been incorporated.")]
        }

    # Gather all available research
    all_research = ""
    for researcher in sub_researchers:
        subtopic = researcher.get("subtopic", "")
        results = researcher.get("research_results", {})
        all_research += f"\n=== Research on: {subtopic} ===\n"
        for source, content in list(results.items())[:2]:  # Top 2 sources per subtopic
            all_research += f"Source: {source}\n{content[:500]}...\n\n"

    # Combine all unincorporated feedback
    combined_feedback = "\n".join(f"{i+1}. {fb}" for i, fb in enumerate(unincorporated_feedback))

    revision_prompt = f"""
    Revise and improve the following research report based on user feedback.

    Current Report:
    {report_content[:8000]}

    User Feedback to Address:
    {combined_feedback}

    Additional Research Available:
    {all_research[:3000]}

    Instructions:
    - Address ALL the feedback points provided
    - Maintain the existing structure and sections
    - Expand, clarify, or add detail where requested
    - Keep all citations intact and add new ones where appropriate
    - Ensure the report remains coherent and well-organized

    Provide the COMPLETE revised report with all sections and references.
    """

    messages = [
        SystemMessage(content="You are a research report writer who revises reports based on user feedback while maintaining quality and coherence."),
        HumanMessage(content=revision_prompt)
    ]

    response = await llm.ainvoke(messages)
    revised_report = response.content if hasattr(response, 'content') else str(response)

    # Mark all current feedback as incorporated
    new_incorporated_list = [True] * len(user_feedback_list)

    print(f"incorporate_feedback: completed revision with {len(unincorporated_feedback)} feedback items addressed")

    # Save revised report to MongoDB
    report_id = state.get("report_id", "")
    new_version_id = state.get("version_id", 0) + 1

    if not report_id:
        print("[Error] incorporate_feedback: no report_id found in state, skipping MongoDB save")
    else:
        try:
            await save_report(report_id, new_version_id, revised_report)
            print(f"incorporate_feedback: saved report {report_id} version {new_version_id} to MongoDB")
        except Exception as e:
            print(f"incorporate_feedback: failed to save report to MongoDB: {e}")

    return {
        "report_content": revised_report,
        "feedback_incorporated": new_incorporated_list,
        "version_id": new_version_id,
        "report_history": state.get("report_history", []) + [new_version_id],
        "messages": [AIMessage(revised_report), AIMessage(content=f"I've revised the report based on your feedback. {len(unincorporated_feedback)} improvement(s) have been addressed.")]
    }


def route_after_feedback(state: dict) -> str:
    """
    Route after collecting user feedback.
    - If all feedback is incorporated: continue to end
    - If any feedback needs incorporation: incorporate feedback
    """
    feedback_incorporated_list = state.get("feedback_incorporated", [])

    # Check if all feedbacks are incorporated
    all_incorporated = all(feedback_incorporated_list) if feedback_incorporated_list else False

    if all_incorporated:
        print("route_after_feedback: all feedback incorporated, routing to 'finalize'")
        return "finalize"
    else:
        unincorporated_count = sum(1 for x in feedback_incorporated_list if not x)
        print(f"route_after_feedback: {unincorporated_count} feedback items need incorporation, routing to 'incorporate'")
        return "incorporate"


# ============================================================================
# CONDITIONAL EDGES
# ============================================================================

def route_after_initial_check(state: dict) -> str:
    """
    Conditional edge: Route after checking initial context.
    - If sufficient context: proceed to subtopic generation (skip clarification loop)
    - If insufficient: enter clarification loop
    """
    is_finalized = state.get("is_finalized", False)
    route = "continue" if is_finalized else "ask_clarification"
    print(f"route_after_initial_check: is_finalized={is_finalized}, routing to '{route}'")
    return route


async def should_continue_to_report(state: dict) -> bool:
    """Check if the subtopics have been generated"""
    subtopics_count = len(state.get("subtopics", []))
    result = subtopics_count > 0
    print(f"should_continue_to_report: subtopics_count={subtopics_count}, returning {result}")
    return result


def route_after_evaluation(state: dict, config: RunnableConfig) -> str:
    """
    Conditional edge: Route after report evaluation
    - If score >= min_report_score: Report is good, finalize
    - If score < min_report_score and revision_count < max_revision_rounds: Identify gaps and revise
    - If revision_count >= max_revision_rounds: Accept report even if not perfect
    """
    agent_config = get_config_from_configurable(config.get("configurable", {}))

    final_score = state.get("final_score", 0)
    revision_count = state.get("revision_count", 0)

    print(f"route_after_evaluation: score={final_score}, revisions={revision_count}")
    print(f"route_after_evaluation: min_report_score={agent_config.min_report_score}, max_revision_rounds={agent_config.max_revision_rounds}")

    # If score is good enough, finalize
    if final_score >= agent_config.min_report_score:
        print(f"route_after_evaluation: score {final_score} >= {agent_config.min_report_score}, finalizing")
        return "finalize"

    # If we've hit max revisions, accept current version
    if revision_count >= agent_config.max_revision_rounds:
        print(f"route_after_evaluation: max revisions ({agent_config.max_revision_rounds}) reached, finalizing")
        return "finalize"

    # Otherwise, identify gaps and revise
    print(f"route_after_evaluation: score {final_score} < {agent_config.min_report_score}, identifying gaps for revision")
    return "revise"


def route_after_display_report(state: dict, config: RunnableConfig) -> str:
    """
    Conditional edge: Route after displaying the final report
    - If user feedback is enabled: Go to prompt_for_feedback
    - If user feedback is disabled: Skip to END
    """
    agent_config = get_config_from_configurable(config.get("configurable", {}))

    if agent_config.enable_user_feedback:
        print("route_after_display_report: user feedback enabled, routing to 'prompt_for_feedback'")
        return "prompt_for_feedback"
    else:
        print("route_after_display_report: user feedback disabled, routing to 'finalize'")
        return "finalize"


