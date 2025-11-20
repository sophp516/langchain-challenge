from langchain_core.messages import HumanMessage
from langgraph.types import Command
import deep_researcher
import asyncio, uuid

# Create agent with checkpointer for local chat execution
agent = deep_researcher.create_agent(use_checkpointer=True)

async def chat():
    # print(agent.get_graph().draw_mermaid_png())

    agent.get_graph().print_ascii()
    thread_id = f"local_{uuid.uuid4().hex[:8]}"

    # change to override agentConfig
    config = {
        "configurable": {
            "thread_id": thread_id
        }
    }

    # initialize state with user's topic
    state = {
        "topic": "",
        "messages": [],
        "is_finalized": False,
        "clarification_rounds": 0,
        "clarification_questions": [],
        "user_responses": []
    }

    first_run = True
    chatting = True
    seen_message_ids = set()  # track messages we've already displayed

    while chatting:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            break

        current_state = agent.get_state(config)

        if current_state.next:
            # resuming from interrupt
            print("[Resuming from interrupt]")
            input_to_agent = Command(resume=user_input)
        elif first_run:
            # first run -> use initial state with user's topic
            state["topic"] = user_input
            state["messages"] = [HumanMessage(content=user_input)]
            input_to_agent = state
            first_run = False
        else:
            # new conversation
            input_to_agent = {
                "topic": user_input,
                "messages": [HumanMessage(content=user_input)]
            }

        result = await agent.ainvoke(input_to_agent, config)

        # get all messages from the result and display only new AI messages
        if "messages" in result and result["messages"]:
            for message in result["messages"]:
                msg_id = message.id

                # Skip if we've already seen this message
                if msg_id in seen_message_ids:
                    continue

                # Only print AI messages
                if hasattr(message, 'type') and message.type == 'ai':
                    content = message.content if hasattr(message, 'content') else str(message)
                    if content.strip():
                        print(f"\nAgent: {content}\n")

                # Mark as seen
                seen_message_ids.add(msg_id)

if __name__ == "__main__":
    asyncio.run(chat())