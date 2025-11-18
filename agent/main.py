# Import only the function to avoid triggering module-level agent creation
# This prevents the API from seeing any state from local runs
import deep_researcher

# Create agent with checkpointer for local chat execution
agent = deep_researcher.create_agent(use_checkpointer=True)
from langchain_core.messages import HumanMessage
import asyncio

async def chat():
    # print(agent.get_graph().draw_mermaid_png())
    agent.get_graph().print_ascii()

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            break

        # Initialize state with user's topic
        state = {
            "topic": user_input,
            "messages": [HumanMessage(content=user_input)],
            "is_finalized": False,
            "clarification_rounds": 0,
            "clarification_questions": [],
            "user_responses": []
        }

        # Run the graph, handling interrupts for clarification questions
        # Use a unique thread_id for each run to avoid conflicts with API threads
        import uuid
        thread_id = f"local_{uuid.uuid4().hex[:8]}"
        config = {"configurable": {"thread_id": thread_id}}

        # First invocation with initial state
        result = await agent.ainvoke(state, config)

        while True:
            # Check if the graph is waiting for clarification (interrupted before collect_response)
            next_node = agent.get_state(config).next
            if next_node and "collect_response" in next_node:
                # Get the clarification question
                clarification_questions = result.get("clarification_questions", [])
                if clarification_questions:
                    latest_question = clarification_questions[-1]
                    print(f"\nAgent: {latest_question}")

                    # Get user's response to the clarification question
                    clarification_response = input("User: ")

                    if clarification_response.lower() in ["exit", "quit", "bye"]:
                        return

                    # Update state with user's clarification response and resume
                    # Use update_state to add the message, then resume with None
                    messages = result.get("messages", [])
                    messages.append(HumanMessage(content=clarification_response))
                    agent.update_state(config, {"messages": messages})

                    # Resume from interrupt by passing None (not the full state!)
                    result = await agent.ainvoke(None, config)
                    continue
            else:
                # Graph completed, show final result
                if "messages" in result and result["messages"]:
                    final_message = result["messages"][-1]
                    if hasattr(final_message, 'content'):
                        print(f"\n{final_message.content}\n")
                break

if __name__ == "__main__":
    asyncio.run(chat())