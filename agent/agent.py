from utils.nodes import *
from utils.unified_state import UnifiedAgentState
from langgraph.graph import StateGraph, END


def create_agent():
    """
    Create and compile the agent graph.
    
    Starts with TopicInquiryState structure (via UnifiedAgentState).
    State transformations happen in nodes:
    - transform_to_subtopic_state: Adds subtopic fields
    - transform_to_report_state: Adds report fields
    - transform_to_evaluator_state: Adds evaluator fields
    """
    workflow = StateGraph(UnifiedAgentState)

    # Topic inquiry nodes
    workflow.add_node("check_initial_context", check_initial_context)
    workflow.add_node("generate_clarification", generate_clarification_question)
    workflow.add_node("collect_response", collect_user_response)
    workflow.add_node("validate_context", validate_context_after_clarification)
    
    # State transformation nodes
    workflow.add_node("transform_to_subtopic", transform_to_subtopic_state)
    workflow.add_node("transform_to_report", transform_to_report_state)
    workflow.add_node("transform_to_evaluator", transform_to_evaluator_state)
    
    # Workflow nodes
    workflow.add_node("generate_subtopics", generate_subtopics)
    workflow.add_node("write_report", write_report)
    workflow.add_node("evaluate_report", evaluate_report)

    # Set entry point
    workflow.set_entry_point("check_initial_context")
    
    # Add edges
    # After checking initial context, route conditionally
    workflow.add_conditional_edges(
        "check_initial_context",
        route_after_initial_check,
        {
            "continue": "transform_to_subtopic",  # Skip clarification if enough context
            "ask_clarification": "generate_clarification"  # Enter clarification loop
        }
    )
    
    # Clarification loop
    workflow.add_edge("generate_clarification", "collect_response")
    workflow.add_edge("collect_response", "validate_context")
    workflow.add_conditional_edges(
        "validate_context",
        route_after_clarification,
        {
            "continue": "transform_to_subtopic",  # Enough context, proceed
            "ask_clarification": "generate_clarification"  # Need more, loop back
        }
    )
    
    # After transformation, generate subtopics
    workflow.add_edge("transform_to_subtopic", "generate_subtopics")
    
    # After generating subtopics, transform to report state and write report
    workflow.add_edge("generate_subtopics", "transform_to_report")
    workflow.add_edge("transform_to_report", "write_report")
    
    # After writing report, transform to evaluator state and evaluate
    workflow.add_edge("write_report", "transform_to_evaluator")
    workflow.add_edge("transform_to_evaluator", "evaluate_report")
    
    # End after evaluation
    workflow.add_edge("evaluate_report", END)

    return workflow.compile()


agent = create_agent()

