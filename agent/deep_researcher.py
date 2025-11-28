from utils.nodes.tools import *
from utils.nodes.helpers import *
from utils.nodes.user_intent import *
from utils.nodes.writing import *
from utils.edges import *
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
    5. Outline generation
    6. Section-by-section writing (full report generated)
    7. User feedback loop (with interrupt)
    8. Incorporate feedback (if requested)
    10. Finalize
    """
    workflow = StateGraph(UnifiedAgentState)

    # Intent Routing Nodes
    workflow.add_node("check_user_intent", check_user_intent)
    workflow.add_node("execute_and_format_tools", execute_and_format_tools)  # OPTIMIZED: creates tool call + executes + formats in one node

    # Topic Inquiry Nodes (OPTIMIZED: single LLM call for evaluation + question generation)
    workflow.add_node("check_initial_context", check_initial_context)
    workflow.add_node("ask_clarification", generate_clarification_question)
    workflow.add_node("collect_response", collect_user_response)

    # Research Workflow Nodes
    workflow.add_node("write_outline", write_outline)
    workflow.add_node("research_and_write", research_and_write)

    # Evaluation Nodes (after user is satisfied)
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

    # Topic Inquiry Flow - OPTIMIZED: check_initial_context does evaluation + question generation in ONE LLM call
    workflow.add_conditional_edges(
        "check_initial_context",
        route_after_initial_check,
        {
            "continue": "write_outline",                  # Sufficient context - proceed to research
            "ask_clarification": "ask_clarification" # Need more info - use pre-generated question
        }
    )

    # generate_clarification adds pre-generated question to messages, then collects response
    workflow.add_edge("ask_clarification", "collect_response")

    # After collecting user response, go back to check_initial_context to re-evaluate
    # check_initial_context -> generate_clarification -> collect_response -> check_initial_context
    workflow.add_edge("collect_response", "check_initial_context")

    # Research Flow
    workflow.add_edge("write_outline", "research_and_write")
    workflow.add_edge("research_and_write", "evaluate_report")
    workflow.add_edge("evaluate_report", END)

    if use_checkpointer:
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    else:
        return workflow.compile()


agent = create_agent(use_checkpointer=False)

