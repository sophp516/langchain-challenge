from agent import agent

def chat():
    print(agent.get_graph().draw_mermaid_png())
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit", "bye"]:
            break
        result = agent.invoke({"user_input": user_input})
        print(result)

if __name__ == "__main__":
    chat()