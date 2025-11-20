from utils.nodes import *
from utils.state import UnifiedAgentState
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


def create_agent(use_checkpointer=False):
    """
    Create and compile the enhanced agent graph with cross-reference verification and user feedback.

    Flow:
    1. Topic inquiry + clarification
    2. Multi-layer research on subtopics
    3. Cross-reference verification
    4. Outline generation
    5. Section-by-section writing (full report generated)
    6. User feedback loop (with interrupt)
    7. Incorporate feedback (if requested)
    8. Finalize
    """
    workflow = StateGraph(UnifiedAgentState)

    # Topic Inquiry Nodes
    workflow.add_node("check_initial_context", check_initial_context)
    workflow.add_node("generate_clarification", generate_clarification_question)
    workflow.add_node("collect_response", collect_user_response)
    workflow.add_node("validate_context", validate_context_after_clarification)

    # Research Workflow Nodes
    workflow.add_node("generate_subtopics", generate_subtopics)
    workflow.add_node("verify_cross_references", verify_cross_references)

    # Report Writing Nodes
    workflow.add_node("generate_outline", generate_outline)
    workflow.add_node("write_sections", write_sections_with_citations)

    # User Feedback Nodes (after full report is generated)
    workflow.add_node("prompt_for_feedback", prompt_for_feedback)
    workflow.add_node("collect_feedback", collect_user_feedback)
    workflow.add_node("incorporate_feedback", incorporate_feedback)

    # Evaluation Nodes (after user is satisfied)
    workflow.add_node("evaluate_report", evaluate_report)
    workflow.add_node("identify_gaps", identify_report_gaps)

    # Entry Point
    workflow.set_entry_point("check_initial_context")

    # Topic Inquiry Flow
    workflow.add_conditional_edges(
        "check_initial_context",
        route_after_initial_check,
        {
            "continue": "generate_subtopics",
            "ask_clarification": "generate_clarification"
        }
    )

    # Clarification loop
    workflow.add_edge("generate_clarification", "collect_response")
    workflow.add_edge("collect_response", "validate_context")
    workflow.add_conditional_edges(
        "validate_context",
        route_after_clarification,
        {
            "continue": "generate_subtopics",
            "ask_clarification": "generate_clarification"
        }
    )

    # Research Flow with Cross-Reference Verification
    workflow.add_edge("generate_subtopics", "verify_cross_references")

    # Report Writing Flow
    workflow.add_edge("verify_cross_references", "generate_outline")
    workflow.add_edge("generate_outline", "write_sections")

    # LLM Evaluation Flow (after writing sections)
    workflow.add_edge("write_sections", "evaluate_report")
    workflow.add_conditional_edges(
        "evaluate_report",
        route_after_evaluation,
        {
            "finalize": "prompt_for_feedback",
            "revise": "identify_gaps"
        }
    )

    # Gap identification can lead to regenerating outline or proceeding to user feedback
    workflow.add_conditional_edges(
        "identify_gaps",
        route_after_gap_identification,
        {
            "regenerate": "generate_outline",
            "finalize": "prompt_for_feedback"
        }
    )

    # Show feedback prompt, then collect feedback
    workflow.add_edge("prompt_for_feedback", "collect_feedback")

    # User Feedback Loop (after LLM evaluation passes)
    workflow.add_conditional_edges(
        "collect_feedback",
        route_after_feedback,
        {
            "incorporate": "incorporate_feedback",
            "finalize": END
        }
    )

    # After incorporating feedback, collect more feedback or end
    workflow.add_edge("incorporate_feedback", "prompt_for_feedback")

    if use_checkpointer:
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    else:
        return workflow.compile()


agent = create_agent(use_checkpointer=False)

