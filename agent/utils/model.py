"""
Model initialization using LangChain's configurable model pattern.

All models use init_chat_model with configurable fields for flexibility.
"""

from langchain.chat_models import init_chat_model
import os



configurable_model = init_chat_model(
    configurable_fields=("model", "max_tokens", "api_key"),
)


def get_fast_model_config(config) -> dict:
    """
    Get model configuration dict for fast LLM (simple tasks).

    Args:
        config: AgentConfig instance

    Returns:
        Dict with model, max_tokens, api_key for use with configurable_model.with_config()
    """
    return {
        "model": config.fast_model,
        "max_tokens": 4096,
        "api_key": os.getenv("OPENAI_API_KEY"),
    }


def get_quality_model_config(config) -> dict:
    """
    Get model configuration dict for quality LLM (complex tasks).

    Args:
        config: AgentConfig instance

    Returns:
        Dict with model, max_tokens, api_key for use with configurable_model.with_config()
    """
    return {
        "model": config.quality_model,
        "max_tokens": 8192,
        "api_key": os.getenv("OPENAI_API_KEY"),
    }


def get_evaluator_model_config(config) -> dict:
    """
    Get model configuration dict for evaluator LLM (unbiased evaluation).

    Args:
        config: AgentConfig instance

    Returns:
        Dict with model, max_tokens, api_key for use with configurable_model.with_config()
    """
    if config.evaluator_provider == "google":
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            print("GEMINI_API_KEY not set - falling back to OpenAI for evaluation")
            api_key = os.getenv("OPENAI_API_KEY")
            model = "gpt-4o-mini"
        else:
            model = config.evaluator_model
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        model = config.evaluator_model

    return {
        "model": model,
        "max_tokens": 4096,
        "api_key": api_key,
    }