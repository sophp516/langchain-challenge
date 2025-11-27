from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
from langgraph.prebuilt import ToolNode
from utils.tools import report_tools
from utils.model import llm
import json



report_tool_node = ToolNode(report_tools)
llm_with_tools = llm.bind_tools(report_tools)


async def call_report_tools(state: dict) -> dict:
    """
    COST OPTIMIZATION: Direct tool call creation based on intent (no LLM needed)
    Saves ~$0.01 per call and reduces latency by 500-1000ms
    """
    intent = state.get("user_intent", "")
    report_id = state.get("intent_report_id", "")
    topic = state.get("topic", "")

    print(f"call_report_tools: intent={intent}, report_id={report_id}")

    # Create tool call directly based on intent (no LLM needed!)
    tool_calls = []

    if intent == "retrieve_report":
        if not report_id:
            # No report_id provided, ask user
            return {
                "messages": [AIMessage(content="Please provide a report ID to retrieve. Example: report_abc123")]
            }
        tool_calls = [{
            "name": "get_report",
            "args": {"report_id": report_id},
            "id": "call_1"
        }]
    elif intent == "list_reports":
        tool_calls = [{
            "name": "list_all_reports",
            "args": {"limit": 20},
            "id": "call_1"
        }]
    else:
        # Unknown intent, should not happen
        return {
            "messages": [AIMessage(content=f"Unknown intent: {intent}")]
        }

    # Create AIMessage with tool_calls (mimics LLM response structure)
    message = AIMessage(content="", tool_calls=tool_calls)

    print(f"call_report_tools: created {len(tool_calls)} tool call(s) directly (no LLM)")

    return {"messages": [message]}


async def execute_and_format_tools(state: dict) -> dict:
    """
    OPTIMIZED: Creates tool call, executes it, and formats response in one node.
    Replaces call_report_tools + execute_and_format_tools for ~50ms latency reduction.
    """
    intent = state.get("user_intent", "")
    report_id = state.get("intent_report_id", "")
    topic = state.get("topic", "")
    messages = state.get("messages", [])

    # Create tool call directly based on intent (same logic as call_report_tools)
    tool_calls = []

    if intent == "retrieve_report":
        if not report_id:
            return {
                "messages": [AIMessage(content="Please provide a report ID to retrieve. Example: report_abc123")]
            }
        tool_calls = [{
            "name": "get_report",
            "args": {"report_id": report_id},
            "id": "call_1"
        }]
    elif intent == "list_reports":
        tool_calls = [{
            "name": "list_all_reports",
            "args": {"limit": 20},
            "id": "call_1"
        }]
    elif intent == "revise_report":
        if not report_id:
            return {
                "messages": [AIMessage(content="Please provide a report ID to revise. Example: revise report_abc123 with feedback...")]
            }

        # Extract feedback from user's message
        # Get the original query from topic or last HumanMessage
        user_query = topic
        if not user_query and messages:
            for msg in reversed(messages):
                if isinstance(msg, HumanMessage):
                    user_query = msg.content
                    break

        # Extract feedback by removing the report_id and command words
        feedback = user_query
        for keyword in ["revise", "update", "modify", "improve", "change", "edit", "report", report_id]:
            feedback = feedback.replace(keyword, "")
        feedback = feedback.strip().lstrip("with").strip()

        if not feedback:
            feedback = "Please improve the report quality and add more details."

        tool_calls = [{
            "name": "revise_report",
            "args": {"report_id": report_id, "feedback": feedback},
            "id": "call_1"
        }]
    else:
        return {
            "messages": [AIMessage(content=f"Unknown intent: {intent}")]
        }

    # Create AIMessage with tool_calls
    tool_call_message = AIMessage(content="", tool_calls=tool_calls)

    # Execute tools with the tool call message
    temp_state = {**state, "messages": state.get("messages", []) + [tool_call_message]}
    tool_result = await report_tool_node.ainvoke(temp_state)
    merged_state = {**temp_state, **tool_result}
    messages = merged_state.get("messages", [])

    # Step 2: Format the tool response
    tool_result_data = None
    for msg in reversed(messages):
        if isinstance(msg, ToolMessage):
            try:
                tool_result_data = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
            except json.JSONDecodeError:
                tool_result_data = {"raw": msg.content}
            break

    if not tool_result_data:
        return {**tool_result, "messages": merged_state["messages"] + [AIMessage(content="No results found.")]}

    # Format based on result type
    # Track state updates separately
    state_updates = {}

    if isinstance(tool_result_data, dict):
        if tool_result_data.get("error"):
            response_text = f"Error: {tool_result_data['error']}"
        elif tool_result_data.get("success") and tool_result_data.get("content"):
            # Handle revise_report response
            report_id = tool_result_data.get('report_id', 'Unknown')
            version_id = tool_result_data.get('version_id', 'N/A')
            previous_version = tool_result_data.get('previous_version', 'N/A')
            content = tool_result_data.get('content', 'No content available')

            # Update state with revised report content
            state_updates["report_content"] = content
            state_updates["report_id"] = report_id
            state_updates["version_id"] = version_id
            state_updates["last_viewed_report_id"] = report_id  # Track context
            state_updates["last_action"] = "revised_report"

            response_text = f"""## ✏️ Revised Report: {report_id}

**New Version:** {version_id} (revised from version {previous_version})

---

{content}"""
        elif tool_result_data.get("found") and tool_result_data.get("content"):
            # Format report with better visual hierarchy (get_report response)
            report_id = tool_result_data.get('report_id', 'Unknown')
            version_id = tool_result_data.get('version_id', 'N/A')
            created_at = tool_result_data.get('created_at', 'Unknown date')
            content = tool_result_data.get('content', 'No content available')

            # Update state with fetched report content
            state_updates["report_content"] = content
            state_updates["report_id"] = report_id
            state_updates["version_id"] = version_id
            state_updates["last_viewed_report_id"] = report_id  # Track context
            state_updates["last_action"] = "viewed_report"

            response_text = f"""## 📄 Report: {report_id}

**Version:** {version_id}
**Created:** {created_at}

---

{content}"""
        elif tool_result_data.get("versions"):
            report_id = tool_result_data.get('report_id', 'Unknown')
            versions = tool_result_data.get("versions", [])
            versions_text = "\n\n".join([
                f"### Version {v['version_id']}\n**Created:** {v['created_at']}\n\n{v.get('content_preview', 'No preview available')}"
                for v in versions
            ])
            response_text = f"""## 📋 Versions of Report: {report_id}

{versions_text}"""
        elif tool_result_data.get("reports"):
            total_reports = tool_result_data.get('total_reports', 0)
            reports = tool_result_data.get("reports", [])
            reports_text = "\n".join([
                f"- **{r['report_id']}** | Version: {r['latest_version']} | Created: {r['created_at']}"
                for r in reports
            ])
            response_text = f"""## 📚 Available Reports ({total_reports})

{reports_text}"""
        elif tool_result_data.get("found") == False:
            response_text = tool_result_data.get("error", "Report not found.")
        else:
            response_text = f"Result: {json.dumps(tool_result_data, indent=2)}"
    else:
        response_text = str(tool_result_data)

    print(f"execute_and_format_tools: executed tools and formatted response")

    # Return with state updates if any
    return {
        **tool_result,
        **state_updates,  # Include state updates (report_content, report_id, version_id)
        "messages": merged_state["messages"] + [AIMessage(content=response_text)]
    }