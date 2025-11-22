from langchain_core.messages import SystemMessage, HumanMessage, AIMessage, ToolMessage
from langgraph.types import RunnableConfig, interrupt
from langgraph.prebuilt import ToolNode
from pydantic import create_model
from utils.model import llm, evaluator_llm
from utils.subresearcher import subresearcher_graph
from utils.verification import verify_research_cross_references
from utils.db import save_report
from utils.configuration import get_config_from_configurable, tavily_client
from utils.tools import report_tools
import asyncio, json, re, uuid


# Create ToolNode for report tools
report_tool_node = ToolNode(report_tools)

# Bind tools to LLM for tool calling
llm_with_tools = llm.bind_tools(report_tools)


# ============================================================================
# INTENT ROUTING NODES
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
    Use LLM with tools to generate tool calls for report operations.
    """
    intent = state.get("user_intent", "")
    report_id = state.get("intent_report_id", "")
    topic = state.get("topic", "")

    print(f"call_report_tools: intent={intent}, report_id={report_id}")

    tool_prompt = f"""
    User request: "{topic}"

    Use the appropriate tool:
    - get_report: To retrieve a specific report by ID
    - list_report_versions: To list versions of a specific report
    - list_all_reports: To list all available reports

    Intent: {intent}
    Report ID: {report_id or "not specified"}

    Call the appropriate tool now.
    """

    messages = [
        SystemMessage(content="You retrieve reports using the available tools."),
        HumanMessage(content=tool_prompt)
    ]

    response = await llm_with_tools.ainvoke(messages)
    print(f"call_report_tools: generated tool_calls={bool(response.tool_calls)}")

    return {"messages": [response]}


async def format_tool_response(state: dict) -> dict:
    """
    Format the tool response into a user-friendly message.
    """
    messages = state.get("messages", [])

    # Find the last tool message
    tool_result = None
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            try:
                tool_result = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
            except json.JSONDecodeError:
                tool_result = {"raw": msg.content}
            break

    if not tool_result:
        return {"messages": [AIMessage(content="No results found.")]}

    # Format based on result type
    if isinstance(tool_result, dict):
        if tool_result.get("error"):
            response_text = f"Error: {tool_result['error']}"
        elif tool_result.get("found") and tool_result.get("content"):
            response_text = f"""**Report: {tool_result.get('report_id')}** (Version {tool_result.get('version_id')})
Created: {tool_result.get('created_at')}

{tool_result.get('content', 'No content available')}"""
        elif tool_result.get("versions"):
            versions_text = "\n".join([
                f"- Version {v['version_id']} ({v['created_at']}): {v['content_preview']}"
                for v in tool_result.get("versions", [])
            ])
            response_text = f"**Versions of {tool_result.get('report_id')}:**\n{versions_text}"
        elif tool_result.get("reports"):
            reports_text = "\n".join([
                f"- **{r['report_id']}** (v{r['latest_version']}) - {r['content_preview']}"
                for r in tool_result.get("reports", [])
            ])
            response_text = f"**Available Reports ({tool_result.get('total_reports')}):**\n{reports_text}"
        elif tool_result.get("found") == False:
            response_text = tool_result.get("error", "Report not found.")
        else:
            response_text = f"Result: {json.dumps(tool_result, indent=2)}"
    else:
        response_text = str(tool_result)

    print(f"format_tool_response: formatted response")

    return {"messages": [AIMessage(content=response_text)]}


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


def generate_clarification_question(state: dict, config: RunnableConfig) -> dict:
    """
    Generate a clarification question based on current topic and previous responses.
    This node uses LLM to determine what information is still needed.
    """
    agent_config = get_config_from_configurable(config.get("configurable", {}))

    clarification_rounds = state.get("clarification_rounds", 0)
    topic = state.get("topic", "")
    clarification_questions = state.get("clarification_questions", [])
    user_responses = state.get("user_responses", [])

    print(f"generate_clarification_question: round={clarification_rounds}, topic='{topic[:50]}...'")

    # If we've asked too many questions, skip this process
    if clarification_rounds >= agent_config.max_clarification_rounds:
        print(f"generate_clarification_question: max rounds reached ({agent_config.max_clarification_rounds}), finalizing")
        return {
            "is_finalized": True,
            "clarification_rounds": clarification_rounds
        }

    context = f"Topic: {topic}\n"
    if clarification_questions:
        context += "\nPrevious questions asked:\n"
        for i, (q, r) in enumerate(zip(clarification_questions, user_responses)):
            context += f"Q{i+1}: {q}\nA{i+1}: {r}\n"
    
    # Generate a clarification question
    clarification_prompt = f"""
    You are a research assistant helping to clarify a research topic.

    Current topic: {topic}

    {context if clarification_rounds > 0 else "This is the initial topic."}

    IMPORTANT RULES:
    1. DO NOT ask about information already present in the topic
    2. If the topic mentions a time period (e.g., "2026"), don't ask "what time period?"
    3. If the topic mentions a subject (e.g., "MMORPGs"), don't ask "what type of games?"
    4. Only ask about truly MISSING information that would significantly improve research quality
    5. If the topic is already specific enough to research, respond with "ENOUGH_CONTEXT"

    Examples:
    - Topic "MMORPG games in 2026" → ENOUGH_CONTEXT (already has subject + time)
    - Topic "games" → Ask about genre, platform, or time period
    - Topic "best practices" → Ask about which domain/field

    Return ONLY the question (or "ENOUGH_CONTEXT"), nothing else.
    """
    
    messages = [
        SystemMessage(content="You are a helpful research assistant that asks clarifying questions to better understand research topics."),
        HumanMessage(content=clarification_prompt)
    ]

    response = llm.invoke(messages)
    question = response.content.strip() if hasattr(response, 'content') else str(response).strip()

    if question.upper() == "ENOUGH_CONTEXT" or "enough context" in question.lower():
        print("generate_clarification_question: LLM indicated enough context")
        return {
            "is_finalized": True,
            "clarification_rounds": clarification_rounds
        }

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


async def validate_context_after_clarification(state: dict, config: RunnableConfig) -> dict:
    """
    Validate if we have enough context after collecting clarification responses.
    Uses LLM to evaluate if the topic and clarifications provide sufficient info.
    """
    agent_config = get_config_from_configurable(config.get("configurable", {}))

    clarification_rounds = state.get("clarification_rounds", 0)
    topic = state.get("topic", "")
    clarification_questions = state.get("clarification_questions", [])
    user_responses = state.get("user_responses", [])

    context = f"Research Topic: {topic}\n\n"

    if clarification_questions:
        context += "Clarification Q&A:\n"
        for i, (q, r) in enumerate(zip(clarification_questions, user_responses)):
            context += f"Q{i + 1}: {q}\nA{i + 1}: {r}\n\n"

    TopicEvaluationOutput = create_model(
        'TopicEvaluationOutput',
        is_sufficient=(bool, ...),
        finalized_topic=(str, ...)
    )

    structured_llm = llm.with_structured_output(TopicEvaluationOutput)

    validation_prompt = f"""
You are evaluating whether there is enough information to conduct comprehensive research.

{context}

Your task:
1. Decide whether the information is sufficient to finalize the topic and proceed with research.
2. Provide your recommended finalized topic regardless of sufficiency.

Definition of sufficiency:
- Topic is specific and well-scoped.
- Clear research directions exist.
- Enough context is present to create subtopics, research steps, and a detailed report.

IMPORTANT:
Return ONLY the following fields in JSON form for structured parsing:
- `is_sufficient`: true/false
- `finalized_topic`: string
"""
    messages = [
        SystemMessage(content="You are a research coordinator evaluating the completeness of context."),
        HumanMessage(content=validation_prompt)
    ]

    response = await structured_llm.ainvoke(messages)

    is_sufficient = bool(response.is_sufficient)
    finalized_topic = response.finalized_topic.strip()

    print(f"validate_context_after_clarification: is_sufficient={is_sufficient}, rounds={clarification_rounds}")

    # If we hit the round limit, finalize no matter what
    if clarification_rounds >= agent_config.max_clarification_rounds:
        return {
            "is_finalized": True,
            "topic": finalized_topic,
            "clarification_rounds": clarification_rounds
        }

    if is_sufficient: # Start generating subtopics
        return {
            "is_finalized": True,
            "topic": finalized_topic
        }

    return {"is_finalized": False}



async def generate_subtopics(state: dict, config: RunnableConfig) -> dict:
    """Generate subtopics and create subresearchers using the subresearcher subgraph with multi-layer research"""

    agent_config = get_config_from_configurable(config.get("configurable", {}))

    topic = state.get("topic", "")
    print(f"generate_subtopics: starting for topic='{topic[:50]}...'")

    # Create structured output model for subtopics
    SubtopicsOutput = create_model(
        'SubtopicsOutput',
        subtopics=(list[str], ...)
    )

    structured_llm = llm.with_structured_output(SubtopicsOutput)

    prompt = f"""
    You are a helpful assistant that generates subtopics for a research report.
    Come up with exactly {agent_config.num_subtopics} subtopics for the given topic.

    Topic: {topic}

    IMPORTANT RULES:
    1. Each subtopic MUST preserve the core constraints of the main topic (time periods, specific subjects, scope limits)
    2. Subtopics should be SPECIFIC and SEARCHABLE, not generic categories
    3. If the topic mentions a specific time (e.g., "2026"), each subtopic must include that constraint
    4. If the topic mentions specific subjects (e.g., "MMORPGs"), each subtopic must focus on that subject

    BAD example for "MMORPG games releasing in 2026":
    - "Game Mechanics" (too generic, loses 2026 constraint)
    - "Player Engagement Strategies" (generic, not about 2026 releases)

    GOOD example for "MMORPG games releasing in 2026":
    - "List of confirmed MMORPG titles scheduled for 2026 release"
    - "Announced features and gameplay mechanics in upcoming 2026 MMORPGs"
    - "Developer studios with MMORPG projects targeting 2026 launch"
    """

    llm_response = await structured_llm.ainvoke([
        SystemMessage(content=f"You are a helpful assistant that generates exactly {agent_config.num_subtopics} subtopics for a research report."),
        HumanMessage(content=prompt)
    ])

    subtopics = llm_response.subtopics
    print(f"generate_subtopics: generated {len(subtopics)} subtopics")

    # Generate subresearchers for each subtopic in parallel
    async def process_subtopic(idx: int, subtopic: str, main_topic: str):
        """Process a single subtopic through the multi-layer subresearcher subgraph"""
        subgraph_state = {
            "subtopic_id": idx,
            "subtopic": subtopic,
            "main_topic": main_topic,  # Pass main topic for search context
            "research_results": {},
            "research_depth": 1,  # Start at layer 1
            "source_credibilities": {},
            "follow_up_queries": []
        }

        result = await subresearcher_graph.ainvoke(subgraph_state)

        return {
            "subtopic_id": idx,
            "subtopic": subtopic,
            "research_results": result.get("research_results", {}),
            "source_credibilities": result.get("source_credibilities", {}),
            "research_depth": result.get("research_depth", 1)
        }

    tasks = [process_subtopic(idx, subtopic, topic) for idx, subtopic in enumerate(subtopics)]
    print(f"generate_subtopics: processing {len(tasks)} subtopics in parallel with multi-layer research")
    sub_researchers = await asyncio.gather(*tasks)
    print(f"generate_subtopics: completed processing {len(sub_researchers)} sub_researchers")

    for researcher in sub_researchers:
        depth = researcher.get("research_depth", 1)
        sources = len(researcher.get("research_results", {}))
        print(f"  - {researcher.get('subtopic', 'Unknown')}: {sources} sources, depth {depth}")

    subtopic_list = "\n".join(f"{i + 1}. {st}" for i, st in enumerate(subtopics))

    subtopic_alert_message = f"""I've come up with {len(subtopics)} research areas on this topic:

    {subtopic_list}
    """

    return {
        "messages": [AIMessage(content=subtopic_alert_message)],
        "sub_researchers": [r for r in sub_researchers],
        "subtopics": subtopics,
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
    Uses identified gaps to improve the outline on revision rounds
    """

    topic = state.get("topic", "")
    sub_researchers = state.get("sub_researchers", [])
    research_gaps = state.get("research_gaps", [])
    revision_count = state.get("revision_count", 0)

    print(f"generate_outline: creating outline for topic='{topic[:50]}...' (revision {revision_count})")

    # Build research overview
    research_overview = f"Topic: {topic}\n\nSubtopics researched:\n"
    for researcher in sub_researchers:
        subtopic = researcher.get("subtopic", "")
        sources = len(researcher.get("research_results", {}))
        research_overview += f"- {subtopic} ({sources} sources)\n"

    # Build gaps section if this is a revision
    gaps_section = ""
    if research_gaps and revision_count > 0:
        gaps_section = "\n\n**IMPORTANT - Address these gaps from previous version:**\n"
        for i, gap in enumerate(research_gaps, 1):
            gap_desc = gap.get("gap_description", "") if isinstance(gap, dict) else gap.gap_description
            gap_type = gap.get("gap_type", "") if isinstance(gap, dict) else gap.gap_type
            priority = gap.get("priority", "") if isinstance(gap, dict) else gap.priority
            gaps_section += f"{i}. [{priority.upper()}] {gap_desc} (type: {gap_type})\n"
        gaps_section += "\nYour outline MUST include sections that specifically address these gaps.\n"

    outline_prompt = f"""
    You are a report outline specialist. Create a structured outline for a comprehensive research report.

    {research_overview}
    {gaps_section}

    Create an outline with:
    1. Executive Summary
    2. Introduction (context and scope)
    3. Main sections (3-5 sections covering the subtopics)
    4. Analysis and Key Findings
    5. Conclusion and Recommendations

    For each main section, specify:
    - Section title
    - Which subtopics it covers
    - Key questions to address
    {"- How it addresses the identified gaps (if applicable)" if gaps_section else ""}

    Return as a JSON object with this structure:
    {{
      "sections": [
        {{
          "title": "Section Title",
          "subtopics": ["subtopic1", "subtopic2"],
          "key_questions": ["question1", "question2"]
        }}
      ]
    }}
    """

    messages = [
        SystemMessage(content="You are a report outline specialist that creates structured outlines for research reports."),
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
    gaps: list = None
) -> dict:
    """
    Write a single section of the report with citations.
    This function is called in parallel for each section.
    If insufficient sources exist, performs additional web search.
    Uses gaps information to improve section quality on revisions.
    """
    section_title = section.get("title", "")
    section_subtopics = section.get("subtopics", [])
    MIN_SOURCES_THRESHOLD = 2  # Minimum sources needed before searching for more
    gaps = gaps or []

    print(f"  Writing section: {section_title}")

    # Gather relevant research for this section
    relevant_research = ""
    sources_list = []
    for subtopic in section_subtopics:
        if subtopic in research_by_subtopic:
            results = research_by_subtopic[subtopic]["results"]
            credibilities = research_by_subtopic[subtopic]["credibilities"]

            # Filter sources by credibility and take top 3
            credible_sources = [(source, findings) for source, findings in results.items()
                              if credibilities.get(source, 0.5) >= min_credibility_score]

            for source, findings in credible_sources[:3]:  # Top 3 credible sources per subtopic
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

    # Build gaps instruction if this is a revision
    gaps_instruction = ""
    if gaps:
        relevant_gaps = [g for g in gaps if g.get("priority") in ["high", "medium"]]
        if relevant_gaps:
            gaps_instruction = "\n\n**IMPORTANT - Address these issues from previous version:**\n"
            for gap in relevant_gaps:
                gap_desc = gap.get("gap_description", "")
                gaps_instruction += f"- {gap_desc}\n"
            gaps_instruction += "\nMake sure your section addresses these gaps with specific details and evidence.\n"

    section_prompt = f"""
    Write the "{section_title}" section of a research report on: {topic}

    Research findings:
    {relevant_research[:3000]}
    {gaps_instruction}

    Write a well-structured section with:
    - Clear topic sentences
    - Evidence from sources with inline citations like [1], [2], etc.
    - Logical flow and transitions
    - 2-4 paragraphs depending on content
    - Specific facts, dates, names, and details (not vague statements)

    Include inline citations referencing the sources by number.

    IMPORTANT: Do NOT include the section title or any header in your response. Start directly with the content.
    """

    messages = [
        SystemMessage(content="You are a research report writer that creates well-structured, evidence-based sections with proper citations."),
        HumanMessage(content=section_prompt)
    ]

    response = await llm.ainvoke(messages)
    section_content = response.content if hasattr(response, 'content') else str(response)

    return {
        "title": section_title,
        "content": section_content,
        "sources": sources_list
    }


# TODO: Improve citation quality
# Map it to the right claim properly
async def write_sections_with_citations(state: dict, config: RunnableConfig) -> dict:
    """
    Write each section of the report with proper inline citations in parallel.
    Uses research results to build evidence-based sections.
    Passes identified gaps to section writers for improved quality on revisions.
    """
    agent_config = get_config_from_configurable(config.get("configurable", {}))

    topic = state.get("topic", "")
    outline = state.get("report_outline", {})
    sub_researchers = state.get("sub_researchers", [])
    research_gaps = state.get("research_gaps", [])
    revision_count = state.get("revision_count", 0)

    print(f"write_sections_with_citations: writing sections for topic='{topic[:50]}...' (revision {revision_count})")
    print(f"write_sections_with_citations: using min_credibility_score={agent_config.min_credibility_score}")

    sections = outline.get("sections", [])

    # Build research lookup by subtopic (includes gap research from follow-up queries)
    research_by_subtopic = {}
    for researcher in sub_researchers:
        subtopic = researcher.get("subtopic", "")
        research_by_subtopic[subtopic] = {
            "results": researcher.get("research_results", {}),
            "credibilities": researcher.get("source_credibilities", {})
        }

    # Also add gap research to a general pool accessible by all sections
    gap_research = {}
    for researcher in sub_researchers:
        subtopic = researcher.get("subtopic", "")
        if subtopic.startswith("[Gap Research]"):
            for source, content in researcher.get("research_results", {}).items():
                gap_research[source] = content

    if gap_research:
        research_by_subtopic["[Gap Research]"] = {
            "results": gap_research,
            "credibilities": {s: 0.7 for s in gap_research.keys()}
        }
        print(f"write_sections_with_citations: added {len(gap_research)} gap research sources")

    # Write all sections in parallel, passing gaps for context
    print(f"write_sections_with_citations: writing {len(sections)} sections in parallel")
    tasks = [
        write_single_section(section, topic, research_by_subtopic, agent_config.min_credibility_score, research_gaps)
        for section in sections
    ]
    written_sections = await asyncio.gather(*tasks)

    print(f"write_sections_with_citations: completed {len(written_sections)} sections")

    # Combine sections into full report
    full_report = f"# {topic}\n\n"
    all_sources = []

    for section in written_sections:
        full_report += f"## {section['title']}\n\n{section['content']}\n\n"
        all_sources.extend(section['sources'])

    # Add references section
    unique_sources = list(dict.fromkeys(all_sources))  # Remove duplicates, preserve order
    full_report += "## References\n\n"
    for i, source in enumerate(unique_sources, 1):
        full_report += f"[{i}] {source}\n"

    # Generate report_id if not exists
    report_id = state.get("report_id", "")
    if not report_id:
        report_id = f"report_{uuid.uuid4().hex[:12]}"
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
        "report_history": state.get("report_history", []) + [new_version_id]
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

    scores = state.get("scores", {})
    scores[current_report_id] = score

    print(f"evaluate_report: score={score}/100 for report_id={current_report_id}")

    return {
        "scores": scores,
        "final_score": score
    }


async def identify_report_gaps(state: dict) -> dict:
    """
    Analyze the report and identify gaps that need addressing.
    Uses gap analysis utility to find issues, then executes follow-up
    queries to gather additional research for each gap.
    """
    from utils.gap_analysis import analyze_report_gaps

    topic = state.get("topic", "")
    report_content = state.get("report_content", "")
    sub_researchers = state.get("sub_researchers", [])
    final_score = state.get("final_score", 0)

    print(f"identify_report_gaps: analyzing report with score={final_score}")

    # Use gap analysis to find issues
    gaps = await analyze_report_gaps(topic, report_content, sub_researchers)

    # Convert ResearchGap objects to dicts for state storage
    gaps_dicts = [
        {
            "gap_description": gap.gap_description,
            "gap_type": gap.gap_type,
            "priority": gap.priority,
            "follow_up_query": gap.follow_up_query
        }
        for gap in gaps
    ]

    revision_count = state.get("revision_count", 0) + 1
    print(f"identify_report_gaps: found {len(gaps_dicts)} gaps (revision {revision_count})")

    # Execute follow-up queries for high/medium priority gaps to enrich research
    updated_sub_researchers = list(sub_researchers)  # Copy existing researchers

    async def execute_follow_up(gap_dict: dict) -> dict | None:
        """Execute a follow-up query and return research results"""
        query = gap_dict.get("follow_up_query", "")
        if not query or query.lower().startswith("n/a"):
            return None

        try:
            loop = asyncio.get_event_loop()
            search_results = await loop.run_in_executor(
                None,
                lambda: tavily_client.search(query=query, max_results=3)
            )
            results = search_results.get("results", [])
            if results:
                research_results = {}
                source_credibilities = {}
                for result in results:
                    url = result.get("url", "")
                    content = result.get("content", "")
                    if url and content:
                        research_results[url] = content
                        source_credibilities[url] = 0.7  # Default credibility for follow-up

                return {
                    "subtopic_id": len(updated_sub_researchers),
                    "subtopic": f"[Gap Research] {gap_dict.get('gap_description', '')[:50]}...",
                    "research_results": research_results,
                    "source_credibilities": source_credibilities,
                    "research_depth": 1
                }
        except Exception as e:
            print(f"identify_report_gaps: follow-up search failed: {e}")
        return None

    # Execute follow-up queries for high and medium priority gaps
    high_medium_gaps = [g for g in gaps_dicts if g.get("priority") in ["high", "medium"]]
    if high_medium_gaps:
        print(f"identify_report_gaps: executing {len(high_medium_gaps)} follow-up queries")
        follow_up_tasks = [execute_follow_up(gap) for gap in high_medium_gaps]
        follow_up_results = await asyncio.gather(*follow_up_tasks)

        for result in follow_up_results:
            if result:
                updated_sub_researchers.append(result)
                print(f"  Added gap research: {result.get('subtopic', '')[:50]}...")

    return {
        "research_gaps": gaps_dicts,
        "revision_count": revision_count,
        "sub_researchers": updated_sub_researchers
    }


def display_final_report(state: dict) -> dict:
    """
    Display the final report to the user after evaluation passes.
    This is a separate node to ensure the report is shown before the feedback prompt.
    """
    report_content = state.get("report_content", "")
    final_score = state.get("final_score", 0)

    print(f"display_final_report: displaying report (score: {final_score}/100)")

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
    final_score = state.get("final_score", 0)

    print(f"prompt_for_feedback: displaying prompt (round {len(user_feedback_list) + 1})")

    feedback_prompt = f"""I've completed the report (score: {final_score}/100).

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


def route_after_clarification(state: dict) -> str:
    """
    Conditional edge: Route after collecting clarification response.
    - If sufficient context: proceed to subtopic generation
    - If insufficient: loop back to ask another clarification question
    """
    is_finalized = state.get("is_finalized", False)
    route = "continue" if is_finalized else "ask_clarification"
    print(f"route_after_clarification: is_finalized={is_finalized}, routing to '{route}'")
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


def route_after_gap_identification(state: dict) -> str:
    """
    Conditional edge: Route after identifying gaps
    - If significant gaps exist: Go back to outline/rewrite
    - If minor/no gaps: Finalize
    """
    research_gaps = state.get("research_gaps", [])
    revision_count = state.get("revision_count", 0)

    high_priority_gaps = [g for g in research_gaps if g.get("priority") == "high"]
    medium_priority_gaps = [g for g in research_gaps if g.get("priority") == "medium"]

    print(f"route_after_gap_identification: {len(high_priority_gaps)} high, {len(medium_priority_gaps)} medium priority gaps")

    # If significant gaps exist, revise
    if high_priority_gaps or medium_priority_gaps:
        print("route_after_gap_identification: significant gaps found, regenerating outline")
        return "regenerate"

    # Otherwise, finalize
    print("route_after_gap_identification: no significant gaps, finalizing")
    return "finalize"


