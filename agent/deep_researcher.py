from utils.nodes import *
from utils.state import UnifiedAgentState
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


def create_agent(use_checkpointer=False):
    """
    Create and compile the enhanced agent graph with intent routing, tool support,
    cross-reference verification and user feedback.

    Flow:
    1. Check user intent (research vs report retrieval)
    2. If report tools: call tools -> execute -> format -> back to intent check
    3. If research: Topic inquiry + clarification
    4. Multi-layer research on subtopics
    5. Cross-reference verification
    6. Outline generation
    7. Section-by-section writing (full report generated)
    8. User feedback loop (with interrupt)
    9. Incorporate feedback (if requested)
    10. Finalize
    """
    workflow = StateGraph(UnifiedAgentState)

    # Intent Routing Nodes
    workflow.add_node("check_user_intent", check_user_intent)
    workflow.add_node("call_report_tools", call_report_tools)
    workflow.add_node("execute_and_format_tools", execute_and_format_tools)  # MERGED: execute_tools + format_tool_response

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
    workflow.add_node("revise_sections", revise_sections)  # Targeted section revision

    # User Feedback Nodes (after full report is generated)
    workflow.add_node("display_report", display_final_report)
    workflow.add_node("prompt_for_feedback", prompt_for_feedback)
    workflow.add_node("collect_feedback", collect_user_feedback)
    workflow.add_node("incorporate_feedback", incorporate_feedback)

    # Evaluation Nodes (after user is satisfied)
    workflow.add_node("evaluate_report", evaluate_report)

    # Entry Point - Intent Check
    workflow.set_entry_point("check_user_intent")

    # Intent Routing
    workflow.add_conditional_edges(
        "check_user_intent",
        route_after_intent_check,
        {
            "research": "check_initial_context",
            "tools": "call_report_tools"
        }
    )

    # Tool Flow: call_report_tools -> check for tool calls -> execute -> format -> END
    workflow.add_conditional_edges(
        "call_report_tools",
        should_continue_tools,
        {
            "execute": "execute_and_format_tools",
            "end": END
        }
    )
    workflow.add_edge("execute_and_format_tools", END)

    # Topic Inquiry Flow
    workflow.add_conditional_edges(
        "check_initial_context",
        route_after_initial_check,
        {
            "continue": "generate_subtopics",
            "ask_clarification": "generate_clarification"
        }
    )

    # Clarification loop - conditional: if ENOUGH_CONTEXT, skip to research
    workflow.add_conditional_edges(
        "generate_clarification",
        route_after_initial_check,  # checks is_finalized
        {
            "continue": "generate_subtopics",  # LLM said ENOUGH_CONTEXT
            "ask_clarification": "collect_response"  # Need user input
        }
    )
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
            "finalize": "display_report",
            "revise": "revise_sections"  # Revise directly based on evaluator feedback
        }
    )

    # After revising sections, re-evaluate the report
    workflow.add_edge("revise_sections", "evaluate_report")

    # Display report first, then conditionally prompt for feedback (or skip to END)
    workflow.add_conditional_edges(
        "display_report",
        route_after_display_report,
        {
            "prompt_for_feedback": "prompt_for_feedback",
            "finalize": END
        }
    )
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

