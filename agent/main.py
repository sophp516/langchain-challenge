from deep_researcher import create_agent
from langchain_core.messages import HumanMessage
from langgraph.types import Command
import asyncio, uuid

async def chat():
    # Need checkpointer for interrupt/resume support
    agent = create_agent(use_checkpointer=True)
    agent.get_graph().print_ascii()

    config = {
        "configurable": {
            "thread_id": str(uuid.uuid4()),
            # Add agent configuration here
            # "max_clarification_rounds": 2,
            # "max_subtopics": 7,
            # "max_search_results": 5,
            # "max_research_depth": 5,
            # "search_api": "tavily",
            # "min_credibility_score": 0.3,
        }
    }

    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            break

        graph_state = agent.get_state(config)

        if graph_state.values:
            # Conversation already exists 
            state = {
                "messages": [HumanMessage(content=user_input)],
            }
        else:
            # First message -> Initialize full state
            state = {
                "topic": user_input,
                "messages": [HumanMessage(content=user_input)],
                "is_finalized": False,
                "clarification_rounds": 0,
                "clarification_questions": [],
                "user_responses": []
            }

        result = await agent.ainvoke(state, config)

        # Handle interrupt/resume loop
        while True:
            graph_state = agent.get_state(config)
            next_nodes = graph_state.next

            if not next_nodes:
                # Show final result
                if "messages" in result and result["messages"]:
                    final_message = result["messages"][-1]
                    if hasattr(final_message, 'content'):
                        print(f"\nAgent: {final_message.content}\n")
                break

            # Handle clarification interrupt
            if "collect_response" in next_nodes:
                clarification_questions = result.get("clarification_questions", [])
                if clarification_questions:
                    print(f"\nAgent: {clarification_questions[-1]}")

            # Handle feedback interrupt
            elif "collect_feedback" in next_nodes:
                # Prompt already shown via prompt_for_feedback node
                pass

            # Get user input for resume
            user_response = input("User: ")
            if user_response.lower() in ["exit", "quit", "bye"]:
                return

            # Resume with user's response
            result = await agent.ainvoke(Command(resume=user_response), config)

if __name__ == "__main__":
    asyncio.run(chat())