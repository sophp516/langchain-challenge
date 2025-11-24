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
            # Format report with better visual hierarchy
            report_id = tool_result.get('report_id', 'Unknown')
            version_id = tool_result.get('version_id', 'N/A')
            created_at = tool_result.get('created_at', 'Unknown date')
            content = tool_result.get('content', 'No content available')
            
            # Create a nicely formatted header with metadata
            response_text = f"""## 📄 Report: {report_id}

**Version:** {version_id}  
**Created:** {created_at}

---

{content}"""
        elif tool_result.get("versions"):
            report_id = tool_result.get('report_id', 'Unknown')
            versions = tool_result.get("versions", [])
            versions_text = "\n\n".join([
                f"### Version {v['version_id']}\n**Created:** {v['created_at']}\n\n{v.get('content_preview', 'No preview available')}"
                for v in versions
            ])
            response_text = f"""## 📋 Versions of Report: {report_id}

{versions_text}"""
        elif tool_result.get("reports"):
            total_reports = tool_result.get('total_reports', 0)
            reports = tool_result.get("reports", [])
            reports_text = "\n".join([
                f"- **{r['report_id']}** | Version: {r['latest_version']} | Created: {r['created_at']}"
                for r in reports
            ])
            response_text = f"""## 📚 Available Reports ({total_reports})

{reports_text}"""
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
    
    If the topic has a clear subject and at least one constraint (time, category, scope, etc.), respond with "ENOUGH_CONTEXT"
    
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

Definition of sufficiency (BE LENIENT):
- Topic has a clear subject/entity
- Topic has at least one constraint (time period, category, scope, criteria, etc.)
- You can identify what to research, even if some details need to be discovered through research
- Standard metrics, criteria, definitions, and methodologies can be researched - don't require them upfront

IMPORTANT - Be generous in marking as sufficient:
- If the topic mentions criteria like "success metrics" or "evaluation criteria", that's ENOUGH - you can research what specific metrics are used
- If the topic has subject + constraint, it's sufficient even if some technical details are missing
- Only mark as insufficient if the topic is truly vague or missing fundamental constraints

Examples:
- "most popular X of 2025" + "criteria A and criteria B" → SUFFICIENT (can research metrics)
- "MMORPG games in 2026" → SUFFICIENT (has subject + time)
- "games" → INSUFFICIENT (too vague, no constraint)

Return ONLY the following fields in JSON form for structured parsing:
- `is_sufficient`: true/false
- `finalized_topic`: string
"""
    messages = [
        SystemMessage(content="You are a research coordinator evaluating context. Be LENIENT - if a topic has a clear subject and constraints, mark it as sufficient even if some technical details are missing. The agent can research standard metrics, criteria, and methodologies. Only mark as insufficient if truly vague or missing fundamental constraints."),
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

    structured_llm = llm_quality.with_structured_output(SubtopicsOutput)

    prompt = f"""
    You are a research strategist that generates optimal subtopics for comprehensive research.
    Generate exactly {agent_config.num_subtopics} subtopics for the given topic.

    Topic: {topic}

    CRITICAL REQUIREMENT: Every subtopic MUST directly answer or contribute to answering the main question/topic. 
    Do NOT include tangential topics, background information, or general context that doesn't directly address the core question.

    FIRST, identify the core question being asked:
    - If topic asks "most popular X" or "best X" → Focus on ranking, metrics, comparisons, specific candidates
    - If topic asks "what is X" or "how does X work" → Focus on explanation, components, mechanisms
    - If topic asks "effects of X" or "impact of X" → Focus on specific effects, outcomes, consequences
    - If topic asks "X in [time period]" → Focus on what happened/occurred in that specific period

    THEN, analyze the topic type and choose the best research approach:

    **APPROACH A - Entity-Focused** (use when topic is about identifying/comparing specific items):
    Best for: "Most popular X", "Top X", "Games releasing in 2026", "Best X"
    Strategy: Research specific entities/candidates individually, then compare and rank
    Example for "Most popular X of 2024":
    - "Top X items on relevant charts/rankings in 2024 - performance metrics"
    - "X items with highest engagement metrics (views, sales, downloads) in 2024"
    - "X items with most social media engagement and viral moments in 2024"
    - "Comparison of top X items across different metrics in 2024"
    
    Example for "MMORPGs releasing in 2026":
    - "Ashes of Creation 2026 release - features, development status, community reception"
    - "Throne and Liberty 2026 - gameplay systems, publisher, launch expectations"
    - "Blue Protocol Western release 2026 - localization, content differences, hype"
    - "Comparison of business models across 2026 MMORPG releases"

    **APPROACH B - Thematic/Analytical** (use when topic needs conceptual analysis):
    Best for: "Effects of climate change", "Best practices for X", "How does Y work"
    Strategy: Break down by themes, causes, effects, or analytical angles - ALL must directly answer the question
    Example for "Impact of AI on healthcare":
    - "AI diagnostic tools and accuracy improvements in medical imaging"
    - "AI-powered drug discovery and development timelines"
    - "Ethical concerns and regulatory challenges of AI in healthcare"

    **APPROACH C - Hybrid** (combine both when appropriate):
    Best for: Complex topics that benefit from both specific examples AND thematic analysis
    Example for "Electric vehicles market 2025":
    - "Tesla Model 3 and Model Y - 2025 updates and market position"
    - "BYD and Chinese EV manufacturers expanding globally in 2025"
    - "Charging infrastructure developments across major markets"
    - "Price trends and affordability of EVs in 2025"

    STRICT RULES:
    1. Each subtopic MUST directly answer the main question - no tangential topics
    2. For "most popular/best X" questions: Focus on ranking, metrics, comparisons, NOT general background
    3. Subtopics should be specific enough to return good search results
    4. Avoid generic/vague subtopics like "Overview", "General trends", or "Background"
    5. Do NOT include subtopics about "cultural impact", "fan engagement", "visual trends" unless they directly answer the main question
    6. For entity-focused topics, try to identify SPECIFIC entities (names, titles, companies) when possible
    7. Ensure subtopics are distinct and don't overlap significantly

    Generate {agent_config.num_subtopics} subtopics that will produce the most comprehensive and useful research DIRECTLY answering the topic.
    """

    llm_response = await structured_llm.ainvoke([
        SystemMessage(content=f"You are a research strategist that generates exactly {agent_config.num_subtopics} optimal subtopics. CRITICAL: Every subtopic must directly answer the main question - do NOT include tangential topics, background information, or general context. For 'most popular/best/top X' questions, focus on ranking, metrics, and comparisons, NOT tangential topics unless they directly determine the answer."),
        HumanMessage(content=prompt)
    ])

    subtopics = llm_response.subtopics
    print(f"generate_subtopics: generated {len(subtopics)} subtopics")
    
    # Validate subtopics are relevant to the main topic
    if subtopics:
        RelevanceCheckOutput = create_model(
            'RelevanceCheckOutput',
            relevant_subtopics=(list[str], ...),
            reasoning=(str, ...)
        )
        
        relevance_llm = llm.with_structured_output(RelevanceCheckOutput)
        
        relevance_prompt = f"""
        Review the following subtopics for the main topic: "{topic}"
        
        Subtopics:
        {chr(10).join(f"{i+1}. {st}" for i, st in enumerate(subtopics))}
        
        The main topic asks: "{topic}"
        
        Filter out subtopics that are:
        - Tangential background (e.g., "Cultural Impact", "General Trends", "Background History" for "most popular X")
        - General context that doesn't identify/rank/compare specific items
        - Topics that are related but don't directly contribute to answering the core question
        
        Keep subtopics that:
        - Directly identify, rank, or compare specific items/entities
        - Focus on metrics, data, or evidence that answers the question
        - Help determine "most popular", "best", "top", etc. through specific criteria
        
        Return the filtered list of relevant subtopics (keep all if they're all relevant).
        """
        
        try:
            relevance_check = await relevance_llm.ainvoke([
                SystemMessage(content="You filter subtopics to ensure they directly answer the main question. Remove any that are tangential or provide only background context."),
                HumanMessage(content=relevance_prompt)
            ])
            
            if relevance_check.relevant_subtopics and len(relevance_check.relevant_subtopics) > 0:
                original_count = len(subtopics)
                subtopics = relevance_check.relevant_subtopics
                print(f"generate_subtopics: filtered from {original_count} to {len(subtopics)} relevant subtopics")
                if original_count != len(subtopics):
                    print(f"generate_subtopics: filtering reasoning: {relevance_check.reasoning}")
            else:
                print("generate_subtopics: relevance check returned empty, using original subtopics")
        except Exception as e:
            print(f"generate_subtopics: relevance check failed: {e}, using original subtopics")

    # Generate subresearchers for each subtopic in parallel
    async def process_subtopic(idx: int, subtopic: str, main_topic: str):
        """Process a single subtopic through the multi-layer subresearcher subgraph"""
        subgraph_state = {
            "subtopic_id": idx,
            "subtopic": subtopic,
            "main_topic": main_topic,  # Pass main topic for search context
            "other_subtopics": subtopics,
            "research_results": {},
            "research_depth": 1,  # Start at layer 1
            "source_credibilities": {},
            "follow_up_queries": [],
            # Pass config values to subresearcher
            "max_search_results": agent_config.max_search_results,
            "max_research_depth": agent_config.max_research_depth,
            "search_api": agent_config.search_api
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

{subtopic_list}"""

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
    You are a report outline specialist. Create a structured outline that DIRECTLY ANSWERS the main research question.

    MAIN RESEARCH QUESTION: {topic}

    {research_overview}
    {gaps_section}

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
    {"- How it addresses the identified gaps" if gaps_section else ""}

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
    gaps: list = None,
    all_sections: list = None,
    section_index: int = 0
) -> dict:
    """
    Write a single section of the report with citations.
    This function is called in parallel for each section.
    If insufficient sources exist, performs additional web search.
    Uses gaps information to improve section quality on revisions.
    Includes outline context to avoid overlap with other sections.
    """
    section_title = section.get("title", "")
    section_subtopics = section.get("subtopics", [])
    all_sections = all_sections or []
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

    # Build outline context to avoid overlap between sections
    outline_context = ""
    if all_sections and len(all_sections) > 1:
        sections_before = all_sections[:section_index]
        sections_after = all_sections[section_index + 1:]

        outline_context = "\n**REPORT STRUCTURE - Your section's role:**\n"
        outline_context += f"You are writing Section {section_index + 1} of {len(all_sections)}.\n\n"

        if sections_before:
            outline_context += "SECTIONS BEFORE YOURS (readers will have already seen this info - DO NOT repeat):\n"
            for i, s in enumerate(sections_before):
                s_title = s.get("title", "")
                s_questions = s.get("key_questions", [])
                outline_context += f"  {i + 1}. \"{s_title}\""
                if s_questions:
                    outline_context += f" - covers: {', '.join(s_questions[:2])}"
                outline_context += "\n"
            outline_context += "\n"

        if sections_after:
            outline_context += "SECTIONS AFTER YOURS (DO NOT cover their content - they will handle it):\n"
            for i, s in enumerate(sections_after):
                s_title = s.get("title", "")
                s_questions = s.get("key_questions", [])
                outline_context += f"  {section_index + 2 + i}. \"{s_title}\""
                if s_questions:
                    outline_context += f" - will cover: {', '.join(s_questions[:2])}"
                outline_context += "\n"
            outline_context += "\n"

        outline_context += """**AVOID REDUNDANCY:**
- Do NOT re-introduce entities/games/items that previous sections already introduced
- If you mention an entity, focus on THIS section's unique angle (not general overview)
- Reference prior sections if needed: "As noted above..." or "Building on the overview..."
- Save details for later sections if they fit better there
"""

    section_prompt = f"""
    Write the "{section_title}" section of a research report on: {topic}
    {outline_context}
    SOURCES (numbered for citation):
    {relevant_research[:5000]}
    {gaps_instruction}

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
    
    # Verify grounding to catch hallucinations
    if relevant_research and section_content:
        try:
            verification_result = await verify_section_grounding(section_content, relevant_research, section_title)
            if verification_result.get("grounding_score", 1.0) < 0.7:
                ungrounded = verification_result.get("ungrounded_claims", [])
                if ungrounded:
                    print(f"  WARNING: Potential hallucinations detected in '{section_title}': {ungrounded[:3]}")
                    # Note: We still return the content, but log the warning
                    # In production, you might want to regenerate or flag for review
        except Exception as e:
            print(f"  Warning: Grounding verification failed for '{section_title}': {e}")

    return {
        "title": section_title,
        "content": section_content,
        "sources": sources_list
    }


async def verify_section_grounding(section_content: str, sources_text: str, section_title: str) -> dict:
    """
    Verify that a section's content is properly grounded in its sources.
    Returns a grounding score and flags potential hallucinations.
    """
    GroundingOutput = create_model(
        'GroundingOutput',
        grounding_score=(float, ...),  # 0.0-1.0
        ungrounded_claims=(list[str], ...),
        has_placeholder_names=(bool, ...),
        placeholder_examples=(list[str], ...)
    )

    structured_llm = llm.with_structured_output(GroundingOutput)

    verification_prompt = f"""
    Verify if the following section content is properly grounded in the provided sources.

    SECTION: {section_title}
    CONTENT:
    {section_content[:2000]}

    SOURCES:
    {sources_text[:2000]}

    CRITICAL: Check for hallucinations - claims that are NOT explicitly stated in the sources:
    1. Specific names, entities, dates, or statistics mentioned in CONTENT but NOT in SOURCES
    2. Claims that combine information from different sources to create new facts
    3. Inferences or assumptions that go beyond what sources explicitly state
    4. Specific rankings, positions, or achievements (e.g., "#1", "first", "most popular", "top") not stated in sources
    5. Placeholder names like "XYZ", "Company A", "Entity A", "the product", etc.

    EXAMPLES OF HALLUCINATIONS TO CATCH:
    - If sources mention "Entity A" but CONTENT says "Entity A was #1" and sources don't say it was #1 → HALLUCINATION
    - If sources mention "Entity B" and "Entity C" separately, but CONTENT says "Entity B and Entity C both achieved X" and sources don't state this together → HALLUCINATION
    - If CONTENT mentions a specific entity name, date, ranking, or statistic that doesn't appear in SOURCES → HALLUCINATION
    - If sources mention a general category but CONTENT names a specific item in that category not mentioned in sources → HALLUCINATION

    Return:
    - grounding_score: 0.0 (completely ungrounded) to 1.0 (fully grounded)
    - ungrounded_claims: List of specific claims/sentences that appear fabricated or not in sources
    - has_placeholder_names: True if placeholder names detected
    - placeholder_examples: Examples of placeholder names found
    """

    try:
        response = await structured_llm.ainvoke([
            SystemMessage(content="You are a strict fact-checker. You identify ANY claim in the content that is not explicitly stated in the sources. Be thorough - flag any names, dates, statistics, rankings, or facts that don't appear in the sources, even if they seem plausible."),
            HumanMessage(content=verification_prompt)
        ])

        return {
            "grounding_score": response.grounding_score,
            "ungrounded_claims": response.ungrounded_claims,
            "has_placeholder_names": response.has_placeholder_names,
            "placeholder_examples": response.placeholder_examples
        }
    except Exception as e:
        print(f"verify_section_grounding: verification failed: {e}")
        return {
            "grounding_score": 0.5,
            "ungrounded_claims": [],
            "has_placeholder_names": False,
            "placeholder_examples": []
        }


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

    # Write all sections in parallel, passing gaps and outline context to avoid redundancy
    print(f"write_sections_with_citations: writing {len(sections)} sections in parallel with outline context")
    tasks = [
        write_single_section(
            section, topic, research_by_subtopic, agent_config.min_credibility_score,
            research_gaps, all_sections=sections, section_index=idx
        )
        for idx, section in enumerate(sections)
    ]
    written_sections = await asyncio.gather(*tasks)

    print(f"write_sections_with_citations: completed {len(written_sections)} sections")

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

    # Convert ResearchGap objects to dicts for state storage (including affected_sections)
    gaps_dicts = [
        {
            "gap_description": gap.gap_description,
            "gap_type": gap.gap_type,
            "priority": gap.priority,
            "follow_up_query": gap.follow_up_query,
            "affected_sections": gap.affected_sections
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


async def revise_sections(state: dict, config: RunnableConfig) -> dict:
    """
    Targeted revision: Only rewrite sections that are affected by identified gaps.
    Much more efficient than regenerating the entire report.
    """
    agent_config = get_config_from_configurable(config.get("configurable", {}))

    topic = state.get("topic", "")
    report_sections = state.get("report_sections", [])
    research_gaps = state.get("research_gaps", [])
    sub_researchers = state.get("sub_researchers", [])
    revision_count = state.get("revision_count", 0)

    print(f"revise_sections: starting targeted revision (revision {revision_count})")

    # Collect all affected sections from gaps
    sections_to_revise = set()
    gaps_by_section = {}

    for gap in research_gaps:
        affected = gap.get("affected_sections", [])
        priority = gap.get("priority", "low")

        # Only consider high/medium priority gaps
        if priority not in ["high", "medium"]:
            continue

        for section_title in affected:
            sections_to_revise.add(section_title)
            if section_title not in gaps_by_section:
                gaps_by_section[section_title] = []
            gaps_by_section[section_title].append(gap)

    print(f"revise_sections: {len(sections_to_revise)} sections need revision: {list(sections_to_revise)}")

    if not sections_to_revise:
        print("revise_sections: no sections to revise, returning unchanged")
        return {}

    # Build research lookup (same as write_sections_with_citations)
    research_by_subtopic = {}
    for researcher in sub_researchers:
        subtopic = researcher.get("subtopic", "")
        research_by_subtopic[subtopic] = {
            "results": researcher.get("research_results", {}),
            "credibilities": researcher.get("source_credibilities", {})
        }

    # Add gap research to pool
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

    # Revise only affected sections
    revised_sections = []
    tasks = []

    for section in report_sections:
        section_title = section.get("title", "")

        if section_title in sections_to_revise:
            # This section needs revision - pass its specific gaps
            section_gaps = gaps_by_section.get(section_title, [])
            print(f"  Revising section: {section_title} ({len(section_gaps)} gaps)")

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
                section_gaps
            ))
        else:
            # Keep section unchanged
            print(f"  Keeping section unchanged: {section_title}")
            revised_sections.append(section)

    # Execute revisions in parallel
    if tasks:
        revised_results = await asyncio.gather(*tasks)
        # Merge revised sections in correct order
        revised_idx = 0
        final_sections = []
        for section in report_sections:
            section_title = section.get("title", "")
            if section_title in sections_to_revise:
                final_sections.append(revised_results[revised_idx])
                revised_idx += 1
            else:
                final_sections.append(section)
        revised_sections = final_sections

    print(f"revise_sections: completed revision of {len(tasks)} sections")

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

    try:
        await save_report(report_id, new_version_id, full_report)
        print(f"revise_sections: saved report {report_id} version {new_version_id} to MongoDB")
    except Exception as e:
        print(f"revise_sections: failed to save report to MongoDB: {e}")

    return {
        "report_sections": revised_sections,
        "report_content": full_report,
        "report_references": unique_sources,
        "version_id": new_version_id,
        "report_history": state.get("report_history", []) + [new_version_id]
    }


async def revise_single_section(
    section: dict,
    topic: str,
    research_by_subtopic: dict,
    min_credibility_score: float,
    section_gaps: list
) -> dict:
    """
    Revise a single section to address specific gaps.
    Uses the original content as a base and improves it.
    """
    section_title = section.get("title", "")
    original_content = section.get("original_content", "")
    section_subtopics = section.get("subtopics", [])

    print(f"    Revising: {section_title}")

    # Gather relevant research
    relevant_research = ""
    sources_list = []

    for subtopic in section_subtopics:
        if subtopic in research_by_subtopic:
            results = research_by_subtopic[subtopic]["results"]
            credibilities = research_by_subtopic[subtopic]["credibilities"]

            for source, findings in list(results.items())[:3]:
                if credibilities.get(source, 0.5) >= min_credibility_score:
                    relevant_research += f"\nSource: {source}\n{findings}\n"
                    sources_list.append(source)

    # Also include gap research
    if "[Gap Research]" in research_by_subtopic:
        gap_results = research_by_subtopic["[Gap Research]"]["results"]
        for source, findings in list(gap_results.items())[:3]:
            relevant_research += f"\nSource (gap research): {source}\n{findings}\n"
            sources_list.append(source)

    # Build gaps instruction
    gaps_instruction = "\n**GAPS TO ADDRESS:**\n"
    for gap in section_gaps:
        gaps_instruction += f"- {gap.get('gap_description', '')}\n"

    revision_prompt = f"""
    You are revising the "{section_title}" section of a research report on: {topic}

    ORIGINAL CONTENT:
    {original_content}

    {gaps_instruction}

    SOURCES FOR REVISION:
    {relevant_research[:2500]}

    **STRICT GROUNDING RULES:**
    1. ONLY add information that appears in the SOURCES above
    2. Every new claim MUST cite a source
    3. DO NOT invent names, titles, dates, statistics, or facts not in sources
    4. DO NOT use placeholder names like "XYZ", "Company A", etc.
    5. If sources don't provide enough info to fill a gap, acknowledge the limitation
    6. It's better to leave a gap partially addressed than to fabricate information

    INSTRUCTIONS:
    - Keep accurate parts of original content
    - Address gaps ONLY with information from the sources provided
    - Add citations [1], [2] for all new claims
    - If you cannot find source support for a gap, write: "Further research needed on [topic]"

    Return ONLY the revised section content (no title/header).
    """

    messages = [
        SystemMessage(content="You are a research report editor who improves sections by addressing specific gaps while preserving good content."),
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
    final_score = state.get("final_score", 0)

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


