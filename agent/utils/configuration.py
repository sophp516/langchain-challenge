import os
from langchain_community.utilities import GoogleSerperAPIWrapper
from pydantic import BaseModel, Field
from typing import Literal
from tavily import TavilyClient


try:
    class SerperClient:
        """Wrapper for Serper.dev API using LangChain's GoogleSerperAPIWrapper"""

        def __init__(self, api_key: str):
            self.wrapper = GoogleSerperAPIWrapper(serper_api_key=api_key)

        def search(self, query: str, max_results: int = 10, search_depth: str = "basic") -> dict:
            """
            Perform a search using Serper API via LangChain wrapper
            Returns results in a format compatible with Tavily
            """
            try:
                # Get results from LangChain wrapper
                raw_results = self.wrapper.results(query)

                # Convert to Tavily-like format
                results = []
                for item in raw_results.get("organic", [])[:max_results]:
                    results.append({
                        "title": item.get("title", ""),
                        "url": item.get("link", ""),
                        "content": item.get("snippet", ""),
                        "score": 0.8
                    })

                return {"results": results}

            except Exception as e:
                print(f"Serper API error: {e}")
                return {"results": []}

    SERPER_AVAILABLE = True
except ImportError:
    SerperClient = None  # Define as None if import fails
    print("WARNING: langchain-community not installed, Serper search unavailable")
    SERPER_AVAILABLE = False


# Initialize Web Search Clients
tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY environment variable is not set")

tavily_client = TavilyClient(api_key=tavily_api_key)

serper_api_key = os.getenv("SERPER_API_KEY")
if SERPER_AVAILABLE and serper_api_key:
    serper_client = SerperClient(api_key=serper_api_key)
else:
    serper_client = None
    if not serper_api_key:
        print("WARNING: SERPER_API_KEY not set, Serper search will not be available")


class AgentConfig(BaseModel):
    """Configuration for the deep research agent - can be passed from UI"""

    # Search API settings
    search_api: Literal["tavily", "serper"] = Field(
        default="serper",
        description="Which search API to use for research"
    )

    max_search_results: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum number of search results per query"
    )

    # Research depth settings
    max_research_depth: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum depth for multi-layer research (1-5)"
    )

    max_clarification_rounds: int = Field(
        default=0,  # TESTING: Disabled clarification to test core flow
        ge=0,
        le=10,
        description="Maximum number of clarification questions (0-10)"
    )

    # Report quality settings
    min_report_score: int = Field(
        default=85,
        ge=0,
        le=100,
        description="Minimum acceptable report score (0-100)"
    )

    max_revision_rounds: int = Field(
        default=1,
        ge=0,
        le=5,
        description="Maximum number of report revision attempts"
    )

    # User feedback settings
    enable_user_feedback: bool = Field(
        default=True,
        description="Whether to collect user feedback on the report"
    )

    # Cross-reference verification
    enable_cross_verification: bool = Field(
        default=False,
        description="Whether to verify cross-references between sources"
    )

    min_credibility_score: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum credibility score for sources (0.0-1.0)"
    )


def get_config_from_configurable(configurable: dict) -> AgentConfig:
    """
    Extract agent configuration from LangGraph configurable dict.
    Filters out system keys like thread_id, checkpoint_ns, etc.
    """
    # Filter out non-config keys
    config_keys = AgentConfig.model_fields.keys()
    config_dict = {k: v for k, v in configurable.items() if k in config_keys}
    return AgentConfig(**config_dict)
