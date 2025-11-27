from utils.nodes.tools import *
from utils.nodes.helpers import *
from utils.nodes.user_intent import *
from utils.nodes.user_feedback import *
from utils.nodes.writing import *
from utils.edges import *
from utils.state import UnifiedAgentState
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver


def create_agent(use_checkpointer=False):
    """
    Create and compile the enhanced agent graph with intent routing and tool support.

    Flow:
    1. Check user intent (research vs report retrieval/revision)
    2. If report tools: call tools (get_report, list_reports, revise_report) -> execute -> format -> END
    3. If research: Topic inquiry + clarification
    4. Multi-layer research on subtopics
    5. Outline generation
    6. Section-by-section writing (full report generated)
    7. Display final report -> END

    Note: Report revision is now handled via the revise_report tool instead of feedback loop
    """
    workflow = StateGraph(UnifiedAgentState)

    # Intent Routing Nodes
    workflow.add_node("check_user_intent", check_user_intent)
    workflow.add_node("execute_and_format_tools", execute_and_format_tools)  # OPTIMIZED: creates tool call + executes + formats in one node

    # Topic Inquiry Nodes
    workflow.add_node("check_initial_context", check_initial_context)
    workflow.add_node("generate_clarification", generate_clarification_question)
    workflow.add_node("collect_response", collect_user_response)

    # Research Workflow Nodes
    workflow.add_node("research_outline_and_write", research_outline_and_write)  # MERGED: generate_subtopics + verify + outline + write

    # Final Display Node
    workflow.add_node("display_report", display_final_report)

    # Evaluation Node (for quality scoring)
    workflow.add_node("evaluate_report", evaluate_report)

    # Entry Point - Intent Check
    workflow.set_entry_point("check_user_intent")

    # Intent Routing - OPTIMIZED: skip call_report_tools, go directly to execute_and_format_tools
    workflow.add_conditional_edges(
        "check_user_intent",
        route_after_intent_check,
        {
            "research": "check_initial_context",
            "tools": "execute_and_format_tools"  # Direct route (no intermediate node)
        }
    )

    # Tool Flow: execute_and_format_tools does everything in one node, then END
    workflow.add_edge("execute_and_format_tools", END)

    # Topic Inquiry Flow
    workflow.add_conditional_edges(
        "check_initial_context",
        route_after_initial_check,
        {
            "continue": "research_outline_and_write",
            "ask_clarification": "generate_clarification"
        }
    )

    # Clarification loop
    # generate_clarification checks if enough context, routes accordingly
    workflow.add_conditional_edges(
        "generate_clarification",
        route_after_initial_check,  # checks is_finalized
        {
            "continue": "research_outline_and_write",  # LLM said ENOUGH_CONTEXT
            "ask_clarification": "collect_response"  # Need user input
        }
    )
    # After collecting user response, go back to generate_clarification to validate
    workflow.add_edge("collect_response", "generate_clarification")

    # Research Flow: research -> evaluate -> display -> END
    workflow.add_edge("research_outline_and_write", "evaluate_report")
    workflow.add_edge("evaluate_report", "display_report")
    workflow.add_edge("display_report", END)

    if use_checkpointer:
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    else:
        return workflow.compile()


agent = create_agent(use_checkpointer=False)

