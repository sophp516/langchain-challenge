from deep_researcher import agent
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
        config = {"configurable": {"thread_id": "1"}}
        while True:
            result = await agent.ainvoke(state, config)

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

                    # Update state with user's clarification response
                    messages = result.get("messages", [])
                    messages.append(HumanMessage(content=clarification_response))
                    state = {**result, "messages": messages}
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