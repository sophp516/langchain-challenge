from utils.nodes import *
from utils.state import UnifiedAgentState
from langgraph.graph import StateGraph, END


def create_agent(use_checkpointer=False):
    """
    Create and compile the enhanced agent graph with feedback loop.

    Flow:
    1. Topic inquiry + clarification
    2. Multi-layer research on subtopics
    3. Outline generation
    4. Section-by-section writing
    5. Evaluation
    6. [If score < 85] Gap identification → Regenerate outline → Rewrite
    7. Finalize
    """
    workflow = StateGraph(UnifiedAgentState)

    # Topic Inquiry Nodes
    workflow.add_node("check_initial_context", check_initial_context)
    workflow.add_node("generate_clarification", generate_clarification_question)
    workflow.add_node("collect_response", collect_user_response)
    workflow.add_node("validate_context", validate_context_after_clarification)

    # State Transformation Nodes
    workflow.add_node("transform_to_subtopic", transform_to_subtopic_state)
    workflow.add_node("transform_to_report", transform_to_report_state)
    workflow.add_node("transform_to_evaluator", transform_to_evaluator_state)

    # Research Workflow Nodes
    workflow.add_node("generate_subtopics", generate_subtopics)  # Now with multi-layer research

    #  Report Writing Nodes (Enhanced)
    workflow.add_node("generate_outline", generate_outline)
    workflow.add_node("write_sections", write_sections_with_citations)

    # Evaluation & Improvement Nodes
    workflow.add_node("evaluate_report", evaluate_report)
    workflow.add_node("identify_gaps", identify_report_gaps)

    # Entry Point
    workflow.set_entry_point("check_initial_context")

    # Topic Inquiry Flow
    workflow.add_conditional_edges(
        "check_initial_context",
        route_after_initial_check,
        {
            "continue": "transform_to_subtopic",
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
            "continue": "transform_to_subtopic",
            "ask_clarification": "generate_clarification"
        }
    )

    # Research Flow
    workflow.add_edge("transform_to_subtopic", "generate_subtopics")

    # Report Writing Flow
    workflow.add_edge("generate_subtopics", "transform_to_report")
    workflow.add_edge("transform_to_report", "generate_outline")
    workflow.add_edge("generate_outline", "write_sections")

    # Evaluation & Feedback Loop
    workflow.add_edge("write_sections", "transform_to_evaluator")
    workflow.add_edge("transform_to_evaluator", "evaluate_report")

    # Finalize or Revise
    workflow.add_conditional_edges(
        "evaluate_report",
        route_after_evaluation,
        {
            "finalize": END,
            "revise": "identify_gaps"
        }
    )

    # After identifying gaps, decide whether to regenerate or finalize
    workflow.add_conditional_edges(
        "identify_gaps",
        route_after_gap_identification,
        {
            "regenerate": "generate_outline",  # Loop back to regenerate outline
            "finalize": END
        }
    )

    # Compile the graph
    # The collect_response node calls interrupt() internally to pause and wait for user input
    # Resume happens via Command(resume=user_input) in main.py
    if use_checkpointer:
        from langgraph.checkpoint.memory import MemorySaver
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    else:
        return workflow.compile()


agent = create_agent(use_checkpointer=False)

