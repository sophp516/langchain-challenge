from langgraph.types import RunnableConfig
from utils.configuration import get_config_from_configurable



def route_after_intent_check(state: dict) -> str:
    """
    Route based on user intent.
    - new_research -> continue to research flow
    - retrieve_report/list_reports/revise_report -> go to tools
    """
    intent = state.get("user_intent", "new_research")

    if intent in ["retrieve_report", "list_reports", "revise_report"]:
        print(f"route_after_intent_check: routing to 'tools' for intent={intent}")
        return "tools"
    else:
        print(f"route_after_intent_check: routing to 'research' for intent={intent}")
        return "research"


def should_continue_tools(state: dict) -> str:
    """
    Check if the last message has tool calls that need execution.
    """
    messages = state.get("messages", [])
    if not messages:
        return "end"

    last_message = messages[-1]

    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print(f"should_continue_tools: has tool calls, routing to 'execute'")
        return "execute"

    print(f"should_continue_tools: no tool calls, routing to 'end'")
    return "end"


def route_after_initial_check(state: dict) -> str:
    """
    Conditional edge: Route after checking initial context.
    - If sufficient context: proceed to subtopic generation (skip clarification loop)
    - If insufficient: enter clarification loop
    """
    is_finalized = state.get("is_finalized", False)
    route = "continue" if is_finalized else "ask_clarification"
    print(f"route_after_initial_check: is_finalized={is_finalized}, routing to '{route}'")
    return route


async def should_continue_to_report(state: dict) -> bool:
    """Check if the subtopics have been generated"""
    subtopics_count = len(state.get("subtopics", []))
    result = subtopics_count > 0
    print(f"should_continue_to_report: subtopics_count={subtopics_count}, returning {result}")
    return result


def route_after_display_report(state: dict, config: RunnableConfig) -> str:
    """
    Conditional edge: Route after displaying the final report
    - If user feedback is enabled: Go to prompt_for_feedback
    - If user feedback is disabled: Skip to END
    """
    agent_config = get_config_from_configurable(config.get("configurable", {}))

    if agent_config.enable_user_feedback:
        print("route_after_display_report: user feedback enabled, routing to 'prompt_for_feedback'")
        return "prompt_for_feedback"
    else:
        print("route_after_display_report: user feedback disabled, routing to 'finalize'")
        return "finalize"


def route_after_feedback(state: dict) -> str:
    """
    Route after collecting user feedback.
    - If all feedback is incorporated: continue to end
    - If any feedback needs incorporation: incorporate feedback
    """
    feedback_incorporated_list = state.get("feedback_incorporated", [])

    # Check if all feedbacks are incorporated
    all_incorporated = all(feedback_incorporated_list) if feedback_incorporated_list else False

    if all_incorporated:
        print("route_after_feedback: all feedback incorporated, routing to 'finalize'")
        return "finalize"
    else:
        unincorporated_count = sum(1 for x in feedback_incorporated_list if not x)
        print(f"route_after_feedback: {unincorporated_count} feedback items need incorporation, routing to 'incorporate'")
        return "incorporate"


