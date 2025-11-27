from langgraph.types import RunnableConfig, interrupt
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from pydantic import create_model
from utils.model import llm
from utils.configuration import get_config_from_configurable



async def check_user_intent(state: dict) -> dict:
    """
    Check user intent at entry point with conversational context awareness.
    Routes to either:
    - "new_research": User wants to research a topic
    - "retrieve_report": User wants to get a specific report
    - "list_reports": User wants to see available reports
    - "revise_report": User wants to revise an existing report
    """
    topic = state.get("topic", "")
    messages = state.get("messages", [])
    last_viewed_report_id = state.get("last_viewed_report_id", "")
    last_action = state.get("last_action", "")

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
    print(f"check_user_intent: context - last_viewed_report_id={last_viewed_report_id}, last_action={last_action}")

    # Create structured output for intent classification
    IntentOutput = create_model(
        'IntentOutput',
        intent=(str, ...),
        extracted_report_id=(str, ...),
        confidence=(float, ...),
        reasoning=(str, ...)
    )

    structured_llm = llm.with_structured_output(IntentOutput)

    # Build context string for LLM
    context_info = ""
    if last_viewed_report_id:
        context_info = f"\n\nCONVERSATIONAL CONTEXT:\n- Last viewed report: {last_viewed_report_id}\n- Last action: {last_action}\n- If the user mentions 'the report', 'this report', or uses revision keywords (change, update, modify) WITHOUT specifying a report_id, they likely mean report '{last_viewed_report_id}'"

    classification_prompt = f"""
    Analyze the user query and determine their intent.

    Query: "{query}"{context_info}

    Classify into ONE category:

    1. "retrieve_report" - User wants to fetch/view a SPECIFIC report
       Examples: "show me report abc123", "get report_xyz", "what's in report abc?"

    2. "list_reports" - User wants to see available reports or versions
       Examples: "what reports do I have?", "show all reports", "list my reports"

    3. "revise_report" - User wants to REVISE/MODIFY an existing report
       Examples: "revise report abc123 with...", "update report xyz to include...", "improve report abc by..."
       Keywords: revise, update, modify, improve, change, edit, add, remove, fix
       IMPORTANT: If user says "change the title", "add X", "update Y" without a report_id, use the report from context

    4. "new_research" - User wants to START NEW research (DEFAULT for ambiguous)
       Examples: "research AI", "tell me about climate change", any topic

    IMPORTANT:
    - If unsure, default to "new_research".
    - Extract report_id if mentioned explicitly (e.g., "report_abc123" → "report_abc123").
    - If no report_id mentioned but context shows a recently viewed report, use that report_id for revise_report intent.
    """

    response = await structured_llm.ainvoke([
        SystemMessage(content="You classify user intent: retrieve_report, list_reports, revise_report, or new_research."),
        HumanMessage(content=classification_prompt)
    ])

    intent = response.intent.lower().strip()
    report_id = response.extracted_report_id.strip() if response.extracted_report_id else ""

    # Validate intent
    valid_intents = ["retrieve_report", "list_reports", "revise_report", "new_research"]
    if intent not in valid_intents:
        intent = "new_research"

    # CONTEXT FALLBACK: If intent is revise_report but no report_id was extracted, use last viewed report
    if intent == "revise_report" and not report_id and last_viewed_report_id:
        report_id = last_viewed_report_id
        print(f"check_user_intent: no report_id in query, using last_viewed_report_id={report_id}")

    print(f"check_user_intent: intent={intent}, report_id={report_id}, confidence={response.confidence:.2f}")

    return {
        "user_intent": intent,
        "intent_report_id": report_id
    }


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