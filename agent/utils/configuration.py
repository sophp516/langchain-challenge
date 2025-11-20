import os
from pydantic import BaseModel, Field
from typing import Literal
from tavily import TavilyClient


# Initialize Web Search Clients
tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY environment variable is not set")

tavily_client = TavilyClient(api_key=tavily_api_key)


class AgentConfig(BaseModel):
    """Configuration for the deep research agent - can be passed from UI"""

    # Search API settings
    search_api: Literal["tavily", "serper", "bing"] = Field(
        default="tavily",
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
        default=2,
        ge=1,
        le=5,
        description="Maximum depth for multi-layer research (1-5)"
    )

    num_subtopics: int = Field(
        default=4,
        ge=2,
        le=10,
        description="Number of subtopics to generate (2-10)"
    )

    # Clarification settings
    max_clarification_rounds: int = Field(
        default=3,
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
        default=2,
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

    # Model settings
    model_name: str = Field(
        default="gpt-4o",
        description="LLM model to use"
    )

    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Temperature for LLM responses"
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
