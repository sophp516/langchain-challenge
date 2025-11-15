import os

from langchain_openai import ChatOpenAI

openai_api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=openai_api_key,
)
