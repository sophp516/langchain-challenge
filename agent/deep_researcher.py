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
    workflow.add_node("execute_and_format_tools", execute_and_format_tools)

    # Topic Inquiry Nodes
    workflow.add_node("check_initial_context", check_initial_context)
    workflow.add_node("ask_clarification", generate_clarification_question)
    workflow.add_node("collect_response", collect_user_response)

    # Research Workflow Nodes
    workflow.add_node("generate_plan_and_research", generate_plan_and_research)
    workflow.add_node("write_full_report", write_full_report)
<<<<<<< HEAD
    workflow.add_node("evaluate_report", evaluate_report)
=======
>>>>>>> d031b308f48d740212c5ddf7867e5c0dce9a5e74

    workflow.set_entry_point("check_user_intent")

    # Intent Routing
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
            "continue": "generate_plan_and_research",  # Sufficient context - proceed to combined research
            "ask_clarification": "ask_clarification"   # Need more info - use pre-generated question
        }
    )

    # generate_clarification adds pre-generated question to messages, then collects response
    workflow.add_edge("ask_clarification", "collect_response")

    # After collecting user response, go back to check_initial_context to re-evaluate
    # check_initial_context -> generate_clarification -> collect_response -> check_initial_context
    workflow.add_edge("collect_response", "check_initial_context")

    # Research Flow
    workflow.add_edge("generate_plan_and_research", "write_full_report")
<<<<<<< HEAD
    workflow.add_edge("write_full_report", "evaluate_report")
    workflow.add_edge("evaluate_report", END)
=======
    workflow.add_edge("write_full_report", END)
>>>>>>> d031b308f48d740212c5ddf7867e5c0dce9a5e74

    if use_checkpointer:
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)
    else:
        return workflow.compile()


agent = create_agent(use_checkpointer=False)

