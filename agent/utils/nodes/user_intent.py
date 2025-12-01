from langgraph.types import RunnableConfig, interrupt
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from pydantic import create_model
from utils.model import llm, llm_quality
from utils.configuration import get_config_from_configurable
from typing import Optional



async def check_user_intent(state: dict, config: RunnableConfig) -> dict:
    """
    Check user intent at entry point with full conversational context.
    Routes to either:
    - "new_research": User wants to research a topic
    - "retrieve_report": User wants to get a specific report
    - "list_reports": User wants to see available reports
    - "revise_report": User wants to revise an existing report
    """
    messages = state.get("messages", [])

    # Filter to relevant messages (HumanMessage and AIMessage only, skip ToolMessages)
    conversation_messages = [
        msg for msg in messages
        if isinstance(msg, (HumanMessage, AIMessage))
    ][-10:]  # Last 10 messages for context

    latest_query = messages[-1].content

    print(f"check_user_intent: analyzing query='{latest_query[:200]}...'")

    IntentOutput = create_model(
        'IntentOutput',
        intent=(str, ...),
        extracted_report_id=(str, ...),
        version_id=(Optional[int], None),  # Optional version_id for get_report, list_report_versions, revise_report
        feedback=(Optional[str], None),  # Optional feedback text for revise_report
        confidence=(float, ...),
        reasoning=(str, ...)
    )

    # Get model config and create structured output model
    structured_llm = llm.with_structured_output(IntentOutput)

    system_prompt = """You classify user intent based on conversation history and extract relevant parameters.

Classify into ONE category:

1. "retrieve_report" - User wants to fetch/view a SPECIFIC report
   Examples: "show me report abc123", "get report_xyz", "what's in report abc?", "fetch report abc123", "retrieve report xyz", "view report 123"
   Keywords: fetch, get, show, view, retrieve, display, see, find + "report" + report_id
   Extract: report_id (required), version_id (optional - if user specifies "version 2" or "v3")

2. "list_reports" - User wants to see available reports or versions
   Examples: "what reports do I have?", "show all reports", "list my reports", "show versions of report abc123"
   Extract: report_id (optional - only if user wants versions of a specific report)

3. "revise_report" - User wants to REVISE/MODIFY an existing report
   Examples: "revise report abc123 with...", "update report xyz to include...", "improve report abc by..."
   Keywords: revise, update, modify, improve, change, edit, add, remove, fix
   Extract: report_id (required), version_id (optional), feedback (required - what changes to make)
   IMPORTANT: If the user mentions "the report", "this report", "that report", or "it" when referring to a previously mentioned report_id in the conversation, extract that report_id.

4. "new_research" - User wants to START NEW research (DEFAULT for ambiguous)
   Examples: "research AI", "tell me about climate change", any topic

EXTRACTION RULES:
- report_id: Extract if mentioned explicitly (e.g., "report_abc123" → "abc123"). If user refers to a report implicitly ("the report", "it", "that one"), look for report_ids mentioned earlier in the conversation.
- version_id: Extract if user mentions a specific version (e.g., "version 2", "v3", "the second version" → 2)
- feedback: For revise_report intent, extract the user's description of what to change/improve

DEFAULTS:
- If unsure about intent, default to "new_research"
- version_id defaults to None (means latest version)
- feedback defaults to None (only needed for revise_report)
"""

    # Build message list: SystemMessage + conversation history
    llm_messages = [SystemMessage(content=system_prompt)] + conversation_messages

    response = await structured_llm.ainvoke(llm_messages)

    intent = response.intent.lower().strip()
    report_id = response.extracted_report_id.strip() if response.extracted_report_id else ""
    version_id = response.version_id if hasattr(response, 'version_id') else None
    feedback = response.feedback.strip() if response.feedback else ""

    # Validate intent
    valid_intents = ["retrieve_report", "list_reports", "revise_report", "new_research"]
    print(f"check_user_intent: intent={intent}")
    if intent not in valid_intents:
        intent = "new_research"

    print(f"check_user_intent: intent={intent}, report_id={report_id}, version_id={version_id}, feedback={feedback[:50] if feedback else 'None'}..., confidence={response.confidence:.2f}")

    return {
        "user_intent": intent,
        "intent_report_id": report_id,
        "intent_version_id": version_id,
        "feedback": feedback
    }


async def check_initial_context(state: dict, config: RunnableConfig) -> dict:
    """
    OPTIMIZED: Check if initial topic has enough context to proceed.
    If insufficient, generates the clarification question in the SAME LLM call (cost savings).
    The question is stored in state but NOT added to messages yet (generate_clarification does that).
    """
    agent_config = get_config_from_configurable(config.get("configurable", {}))

    topic = state.get("topic", "")
    clarification_rounds = state.get("clarification_rounds", 0)
    clarification_questions = state.get("clarification_questions", [])
    user_responses = state.get("user_responses", [])

    print(f"check_initial_context: topic='{topic[:50]}...', round={clarification_rounds}, max_clarification_rounds={agent_config.max_clarification_rounds}")

    # If clarification is disabled (max_clarification_rounds = 0), skip validation and proceed
    if agent_config.max_clarification_rounds == 0:
        print("check_initial_context: clarification disabled, proceeding to research")
        return {
            "is_finalized": True,
            "messages": [HumanMessage(content=f"Thank you. I will now start research on topic: {topic.strip()}")]
        }

    # If we've asked too many questions, finalize with what we have
    if clarification_rounds >= agent_config.max_clarification_rounds:
        print(f"check_initial_context: max rounds reached ({agent_config.max_clarification_rounds}), finalizing")
        finalized_topic = topic
        if user_responses:
            finalized_topic += " " + " ".join(user_responses)
        return {
            "is_finalized": True,
            "topic": finalized_topic.strip(),
            "clarification_rounds": clarification_rounds,
            "messages": [HumanMessage(content=f"Thank you. I will now start research on topic: {finalized_topic.strip()}")]
        }

    if not topic or len(topic.strip()) == 0:
        print("check_initial_context: no topic found, returning is_finalized=False")
        return {
            "is_finalized": False,
            "pending_clarification_question": "What would you like to research?",
            "messages": [AIMessage(content="What would you like to research?")]
        }

    # Build context with all Q&A so far
    context = f"Topic: {topic}\n"
    if clarification_questions and user_responses:
        context += "\nPrevious Q&A:\n"
        for i, (q, r) in enumerate(zip(clarification_questions, user_responses)):
            context += f"Q{i+1}: {q}\nA{i+1}: {r}\n"

    # OPTIMIZATION: Single LLM call for both evaluation AND question generation
    TopicEvaluation = create_model(
        'TopicEvaluation',
        is_sufficient=(bool, ...),
        reasoning=(str, ...),
        clarifying_question=(str, ...)  # Empty string if sufficient
    )

    topic_evaluation_llm = llm.with_structured_output(TopicEvaluation)

    evaluation_prompt = f"""
    You are a research coordinator. Evaluate if the following is a valid research topic that can be researched.

    {context}

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

    CRITICAL RULES - If insufficient, ONLY ask for information that CANNOT be researched:
    1. DO NOT ask about information already present in the topic or previous responses
    2. DO NOT ask about standard metrics, criteria, definitions, or methodologies - these can be researched
    3. DO NOT ask "What metrics are used?" or "What criteria determine X?" - research can find this
    4. DO NOT ask "How is X measured?" or "What are standard practices?" - research can answer this
    5. DO NOT ask about definitions or technical terms - research can provide these

    ONLY ask for:
    - User-specific preferences (e.g., "Which specific items/entities do you want to compare?")
    - Missing constraints that are user choices (e.g., "What time period?" if not mentioned)
    - Ambiguities that require user input (e.g., "Do you mean X or Y?" when both are possible)

    If is_sufficient=True, set clarifying_question to empty string "".
    If is_sufficient=False, provide a specific clarifying question.
    """

    messages = [
        SystemMessage(content="You are a research coordinator. Accept any topic that has a clear subject and at least one constraint (time, place, category, scope). Only reject greetings and completely vague requests. You ONLY ask for information that requires user input and cannot be discovered through research."),
        HumanMessage(content=evaluation_prompt)
    ]

    response = await topic_evaluation_llm.ainvoke(messages)
    is_sufficient = response.is_sufficient
    question = response.clarifying_question.strip()

    print(f"check_initial_context: is_sufficient={is_sufficient}, reasoning='{response.reasoning[:100]}...'")

    # If sufficient, mark as finalized
    if is_sufficient:
        finalized_topic = topic
        if user_responses:
            finalized_topic += " " + " ".join(user_responses)
        return {
            "is_finalized": True,
            "topic": finalized_topic.strip(),
        }

    # Not sufficient - store the pre-generated question for generate_clarification to use
    if question:
        print(f"check_initial_context: generated question='{question[:100]}...'")
        return {
            "is_finalized": False,
            "pending_clarification_question": question,  # Store for next node to add to messages
        }

    # Fallback: if no question generated but marked insufficient, finalize anyway
    print("check_initial_context: marked insufficient but no question generated, finalizing")
    return {
        "is_finalized": True,
        "topic": topic.strip(),
        "messages": [HumanMessage(content=f"Thank you. I will now start research on topic: {topic.strip()}")]
    }


def generate_clarification_question(state: dict, config: RunnableConfig) -> dict:
    """
    OPTIMIZED: This node now simply takes the pre-generated question from check_initial_context
    and adds it to messages. No LLM call needed here - saves 50% cost on clarification loop!
    """
    clarification_rounds = state.get("clarification_rounds", 0)
    clarification_questions = state.get("clarification_questions", [])
    pending_question = state.get("pending_clarification_question", "")

    print(f"generate_clarification_question: using pre-generated question='{pending_question[:100]}...'")

    # Add the pre-generated question to messages and tracking
    new_questions = clarification_questions + [pending_question]
    clarification_message = AIMessage(content=pending_question)

    return {
        "messages": [clarification_message],
        "clarification_questions": new_questions,
        "clarification_rounds": clarification_rounds + 1,
        "pending_clarification_question": ""  # Clear the pending question
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