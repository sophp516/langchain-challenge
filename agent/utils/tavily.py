import os

from tavily import TavilyClient

tavily_api_key = os.getenv("TAVILY_API_KEY")

if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY environment variable is not set")

tavily_client = TavilyClient(api_key=tavily_api_key)

