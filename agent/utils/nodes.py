from typing import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, create_model
from utils.state import TopicInquiryState, SubtopicGenerationState, SubResearcherState
from utils.model import llm
from utils.subresearcher import subresearcher_graph
import asyncio



###### Nodes ######

async def check_initial_context(state: dict) -> dict:
    """
    Check if initial topic has enough context to proceed.
    If sufficient, mark as finalized. If not, prepare for clarification loop.
    """
    topic = state.get("topic", "")
    if not topic:
        return {
            "is_finalized": False
        }
    
    # Use LLM to evaluate if initial topic has enough context
    evaluation_prompt = f"""
    Evaluate if the following research topic has enough information to proceed with comprehensive research:
    
    Topic: {topic}
    
    Determine if this topic is specific and detailed enough to:
    1. Generate meaningful subtopics
    2. Conduct thorough research
    3. Create a comprehensive report
    
    Consider:
    - Is the topic specific enough? (vs too vague/broad)
    - Are there clear research directions?
    - Is the scope well-defined?
    
    Respond with ONLY "SUFFICIENT" or "INSUFFICIENT" followed by a brief reason.
    """
    
    messages = [
        SystemMessage(content="You are a research coordinator evaluating if a topic has enough information to proceed."),
        HumanMessage(content=evaluation_prompt)
    ]
    
    response = await llm.ainvoke(messages)
    evaluation = response.content.strip() if hasattr(response, 'content') else str(response).strip()
    
    # Check if context is sufficient
    is_sufficient = "SUFFICIENT" in evaluation.upper() or "sufficient" in evaluation.lower()
    
    return {
        "is_finalized": is_sufficient
    }


async def generate_clarification_question(state: dict) -> dict:
    """
    Generate a clarification question based on current topic and previous responses.
    This node uses LLM to determine what information is still needed.
    """
    MAX_CLARIFICATION_ROUNDS = 5  # Safety limit
    
    clarification_rounds = state.get("clarification_rounds", 0)
    topic = state.get("topic", "")
    clarification_questions = state.get("clarification_questions", [])
    user_responses = state.get("user_responses", [])
    
    # Safety check: if we've asked too many questions, proceed anyway
    if clarification_rounds >= MAX_CLARIFICATION_ROUNDS:
        return {
            "is_finalized": True,
            "clarification_rounds": clarification_rounds
        }
    
    # Build context from previous interactions
    context = f"Topic: {topic}\n"
    if clarification_questions:
        context += "\nPrevious questions asked:\n"
        for i, (q, r) in enumerate(zip(clarification_questions, user_responses)):
            context += f"Q{i+1}: {q}\nA{i+1}: {r}\n"
    
    # Use LLM to generate a clarification question
    clarification_prompt = f"""
    You are a research assistant helping to clarify a research topic.
    
    Current topic: {topic}
    
    {context if clarification_rounds > 0 else "This is the initial topic."}
    
    Generate a specific clarification question to better understand what the user wants to research.
    The question should help narrow down the scope, focus, or specific aspects of the topic.
    
    If you have enough information to proceed with research, respond with "ENOUGH_CONTEXT" instead of a question.
    
    Return ONLY the question (or "ENOUGH_CONTEXT"), nothing else.
    """
    
    messages = [
        SystemMessage(content="You are a helpful research assistant that asks clarifying questions to better understand research topics."),
        HumanMessage(content=clarification_prompt)
    ]
    
    response = await llm.ainvoke(messages)
    question = response.content.strip() if hasattr(response, 'content') else str(response).strip()
    
    # Check if LLM says we have enough context
    if question.upper() == "ENOUGH_CONTEXT" or "enough context" in question.lower():
        return {
            "is_finalized": True,
            "clarification_rounds": clarification_rounds
        }
    
    # Add the question to the list
    new_questions = clarification_questions + [question]
    
    return {
        "clarification_questions": new_questions,
        "clarification_rounds": clarification_rounds + 1
    }


async def collect_user_response(state: dict) -> dict:
    """
    Collect user's response to the clarification question.
    Updates topic with additional context from the response.
    """
    messages = state.get("messages", [])
    topic = state.get("topic", "")
    user_responses = state.get("user_responses", [])
    
    logger.info(f"Collecting user response (current topic: '{topic[:50]}...')")
    
    if messages:
        last_message = messages[-1]
        if isinstance(last_message, HumanMessage):
            response = last_message.content
            
            # Add response to the list
            new_responses = user_responses + [response]
            
            # Update topic with additional context
            updated_topic = f"{topic}. {response}" if topic else response
            logger.info(f"User response collected: '{response[:50]}...'")
            
            return {
                "user_responses": new_responses,
                "topic": updated_topic
            }
    
    return {}


async def validate_context_after_clarification(state: dict) -> dict:
    """
    Validate if we have enough context after collecting clarification responses.
    Uses LLM to evaluate if the topic and clarifications provide sufficient information.
    """
    MAX_CLARIFICATION_ROUNDS = 5
    
    clarification_rounds = state.get("clarification_rounds", 0)
    topic = state.get("topic", "")
    clarification_questions = state.get("clarification_questions", [])
    user_responses = state.get("user_responses", [])
    
    # Safety check: if we've asked too many questions, proceed anyway
    if clarification_rounds >= MAX_CLARIFICATION_ROUNDS:
        return {
            "is_finalized": True,
            "clarification_rounds": clarification_rounds
        }
    
    # Build full context
    context = f"Research Topic: {topic}\n\n"
    
    if clarification_questions:
        context += "Clarification Q&A:\n"
        for i, (q, r) in enumerate(zip(clarification_questions, user_responses)):
            context += f"Q{i+1}: {q}\nA{i+1}: {r}\n\n"
    
    validation_prompt = f"""
    You are evaluating whether there is enough information to conduct comprehensive research.
    
    {context}
    
    Determine if this information is sufficient to:
    1. Generate meaningful subtopics
    2. Conduct thorough research
    3. Create a comprehensive report
    
    Consider:
    - Is the topic specific enough?
    - Are there clear research directions?
    - Is the scope well-defined?
    
    Respond with ONLY "SUFFICIENT" or "INSUFFICIENT" followed by a brief reason.
    """
    
    messages = [
        SystemMessage(content="You are a research coordinator evaluating if enough information has been gathered."),
        HumanMessage(content=validation_prompt)
    ]
    
    response = await llm.ainvoke(messages)
    evaluation = response.content.strip() if hasattr(response, 'content') else str(response).strip()
    
    # Check if context is sufficient
    is_sufficient = "SUFFICIENT" in evaluation.upper() or "sufficient" in evaluation.lower()
    
    return {
        "is_finalized": is_sufficient
    }


# ============================================================================
# STATE TRANSFORMATION NODES
# ============================================================================

async def transform_to_subtopic_state(state: dict) -> dict:
    """
    Transformation node: Convert TopicInquiryState to SubtopicGenerationState structure.
    This node adds the fields needed for subtopic generation.
    """
    # Extract TopicInquiryState fields
    topic = state.get("topic", "")
    messages = state.get("messages", [])
    
    # Add SubtopicGenerationState fields
    return {
        "subtopics": [],
        "sub_researchers": []
    }


async def transform_to_report_state(state: dict) -> dict:
    """
    Transformation node: Convert SubtopicGenerationState to ReportWriterState structure.
    This node adds the fields needed for report writing.
    """
    # Extract research data
    sub_researchers = state.get("sub_researchers", [])
    
    # Build references from research results
    references = []
    for researcher in sub_researchers:
        if isinstance(researcher, dict):
            research_results = researcher.get("research_results", {})
        else:
            research_results = getattr(researcher, "research_results", {})
        
        for source in research_results.keys():
            references.append(source)
    
    # Add ReportWriterState fields
    return {
        "report_history": [],
        "current_report_id": 0,
        "report_content": "",
        "report_summary": "",
        "report_conclusion": "",
        "report_recommendations": [],
        "report_references": references,
        "report_citations": [],
        "report_footnotes": [],
        "report_endnotes": []
    }


async def transform_to_evaluator_state(state: dict) -> dict:
    """
    Transformation node: Convert ReportWriterState to ReportEvaluatorState structure.
    This node adds the fields needed for report evaluation.
    """
    report_history = state.get("report_history", [])
    current_report_id = state.get("current_report_id", 0)
    
    return {
        "scores": {}
    }


# ============================================================================
# WORKFLOW NODES
# ============================================================================

async def generate_subtopics(state: dict) -> dict:
    """Generate subtopics and create subresearchers using the subresearcher subgraph"""
    # Read topic from state (works with unified state)
    topic = state.get("topic", "")
    
    # Create structured output model for subtopics
    SubtopicsOutput = create_model(
        'SubtopicsOutput',
        subtopics=(list[str], ...)
    )
    
    structured_llm = llm.with_structured_output(SubtopicsOutput)
    
    prompt = f"""
    You are a helpful assistant that generates subtopics for a research report. 
    Come up with 3-5 subtopics for the given topic based on the topic's scope and complexity.
    
    Topic: {topic}
    """
    
    llm_response = await structured_llm.ainvoke([
        SystemMessage(content="You are a helpful assistant that generates subtopics for a research report. Come up with 3-5 subtopics for the given topic based on the topic's scope and complexity."),
        HumanMessage(content=prompt)
    ])

    subtopics = llm_response.subtopics

    # Generate subresearchers for each subtopic using the subgraph in parallel
    async def process_subtopic(idx: int, subtopic: str):
        """Process a single subtopic through the subresearcher subgraph"""
        subgraph_state = {
            "subtopic_id": idx,
            "subtopic": subtopic,
            "research_results": {}
        }
        
        # Invoke the subresearcher subgraph
        result = await subresearcher_graph.ainvoke(subgraph_state)
        
        # Return as dict (will be stored in unified state)
        return {
            "subtopic_id": idx,
            "subtopic": subtopic,
            "research_results": result.get("research_results", {})
        }
    
    # Process all subtopics in parallel
    tasks = [process_subtopic(idx, subtopic) for idx, subtopic in enumerate(subtopics)]
    sub_researchers = await asyncio.gather(*tasks)
    
    return {
        "sub_researchers": [r for r in sub_researchers],
        "subtopics": subtopics,
    }


async def write_report(state: dict) -> dict:
    """Write a report based on the research results"""
    # Extract data from unified state
    topic = state.get("topic", "")
    sub_researchers = state.get("sub_researchers", [])
    
    # Build research summary
    research_summary = f"Research on: {topic}\n\n"
    for researcher in sub_researchers:
        if isinstance(researcher, dict):
            subtopic = researcher.get("subtopic", "")
            research_results = researcher.get("research_results", {})
        else:
            subtopic = getattr(researcher, "subtopic", "")
            research_results = getattr(researcher, "research_results", {})
        
        research_summary += f"Subtopic: {subtopic}\n"
        research_summary += f"Sources: {len(research_results)}\n\n"
    
    llm_response = await llm.ainvoke([
        SystemMessage(content="You are a helpful assistant that writes a report based on the research results."),
        HumanMessage(content=f"Write a comprehensive report based on the following research:\n\n{research_summary}"),
    ])
    
    report_content = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
    
    return {
        "report_content": report_content,
        "report_summary": f"Research report on {topic}",
        "current_report_id": state.get("current_report_id", 0) + 1,
        "report_history": state.get("report_history", []) + [state.get("current_report_id", 0) + 1]
    }


async def evaluate_report(state: dict) -> dict:
    """Evaluate a report based on the research results"""
    report_content = state.get("report_content", "")
    current_report_id = state.get("current_report_id", 0)
    
    llm_response = await llm.ainvoke([
        SystemMessage(content="You are a helpful assistant that evaluates reports. Provide a score from 1-100."),
        HumanMessage(content=f"Evaluate the following report and provide a score (1-100):\n\n{report_content}"),
    ])
    
    # Extract score from response (simplified - you might want structured output)
    response_text = llm_response.content if hasattr(llm_response, 'content') else str(llm_response)
    # Try to extract a number from the response
    import re
    scores_found = re.findall(r'\d+', response_text)
    score = int(scores_found[0]) if scores_found else 75  # Default score
    
    scores = state.get("scores", {})
    scores[current_report_id] = score
    
    return {
        "scores": scores
    }


###### Edge Functions ######

def route_after_initial_check(state: dict) -> str:
    """
    Conditional edge: Route after checking initial context.
    - If sufficient context: proceed to subtopic generation (skip clarification loop)
    - If insufficient: enter clarification loop
    """
    if state.get("is_finalized", False):
        return "continue"  # Skip clarification, proceed to subtopic generation
    else:
        return "ask_clarification"  # Enter clarification loop


def route_after_clarification(state: dict) -> str:
    """
    Conditional edge: Route after collecting clarification response.
    - If sufficient context: proceed to subtopic generation
    - If insufficient: loop back to ask another clarification question
    """
    if state.get("is_finalized", False):
        return "continue"  # Enough context, proceed to subtopic generation
    else:
        return "ask_clarification"  # Need more context, loop back


async def should_continue_to_report(state: dict) -> bool:
    """Check if the subtopics have been generated"""
    return len(state.get("subtopics", [])) > 0


