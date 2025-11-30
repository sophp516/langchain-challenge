import os

from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

openai_api_key = os.getenv("OPENAI_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")

if not openai_api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

# Fast LLM for simple tasks (intent classification, validation, etc.)
llm = ChatOpenAI(
    model="gpt-4o-mini",
    api_key=openai_api_key,
)

# Quality LLM for complex tasks (writing, extraction, planning)
llm_quality = ChatOpenAI(
    model="gpt-5.1",
    api_key=openai_api_key,
)

# External evaluator LLM (Gemini) for unbiased evaluation
# Matches RACE benchmark which uses Gemini as judge
if gemini_api_key:
    evaluator_llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=gemini_api_key,
    )
    print("Using Gemini as external evaluator (matches RACE benchmark)")
else:
    print("GOOGLE_API_KEY not set - using GPT for evaluation")
    evaluator_llm = llm