from pydantic import BaseModel, Field
from typing import Literal
from tavily import TavilyClient
import os



class AgentConfig(BaseModel):
    """Configuration for the deep research agent - can be passed from UI"""

    # Model Configuration
    fast_model: str = Field(
        default="gpt-4o-mini",
        description="Fast LLM for simple tasks (intent classification, clarification, outline generation)"
    )

    quality_model: str = Field(
        default="gpt-5.1",
        description="Quality LLM for complex tasks (writing, section generation, planning)"
    )

    evaluator_model: str = Field(
        default="gemini-2.0-flash",
        description="External evaluator LLM for unbiased report evaluation (use different provider than main models)"
    )

    evaluator_provider: Literal["openai", "google"] = Field(
        default="google",
        description="Provider for evaluator model (openai or google)"
    )

    # Search API settings
    # TODO: Find and add web search apis that have full content extraction with search
    search_api: Literal["tavily"] = Field(
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
        default=3,
        ge=1,
        le=5,
        description="Maximum depth for multi-layer research (1-5)"
    )

    max_clarification_rounds: int = Field(
        default=0,
        ge=0,
        le=10,
        description="Maximum number of clarification questions (0-10)"
    )

    max_revision_rounds: int = Field(
        default=1,
        ge=0,
        le=5,
        description="Maximum number of report revision attempt"
    )

    min_credibility_score: float = Field(
        default=0.3,
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


# Initialize Web Search Clients
tavily_api_key = os.getenv("TAVILY_API_KEY")
if not tavily_api_key:
    raise ValueError("TAVILY_API_KEY environment variable is not set")

tavily_client = TavilyClient(api_key=tavily_api_key)
