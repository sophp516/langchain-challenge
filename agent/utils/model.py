import os

from langchain_openai import ChatOpenAI

openai_api_key = os.getenv("OPENAI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=openai_api_key,
)

