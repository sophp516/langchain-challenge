"""
Run evaluation on Deep Research Bench dataset via LangSmith.

This script:
1. Loads the deep_research_bench dataset from LangSmith
2. Runs your research agent on each example
3. Stores results in a LangSmith experiment

Usage:
    python tests/run_evaluate.py [--limit N] [--experiment-name NAME]

Warning: Running all 100 examples can cost ~$20-$100 depending on model selection.
"""

import os
import sys
import argparse
import asyncio
from datetime import datetime
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langsmith import Client, evaluate
from langsmith.schemas import Example, Run

from deep_researcher import create_agent
from utils.configuration import Configuration


# LangSmith dataset name
DATASET_NAME = "deep_research_bench"


def create_research_agent_target(config: Configuration | None = None):
    """
    Create a target function that runs the research agent on a given input.

    Args:
        config: Optional configuration overrides for the agent

    Returns:
        A function that takes an input dict and returns the agent's output
    """
    # Create agent without checkpointer for evaluation (stateless runs)
    agent = create_agent(use_checkpointer=False)

    async def run_agent(inputs: dict) -> dict:
        """Run the research agent on the given input."""
        # Extract the research topic/query from the input
        topic = inputs.get("topic") or inputs.get("query") or inputs.get("question", "")

        # Prepare initial state
        initial_state = {
            "topic": topic,
            "messages": [],
            "is_finalized": True,  # Skip clarification for evaluation
            "clarification_rounds": 0,
            "clarification_questions": [],
            "user_responses": [],
            "subtopics": [],
            "sub_researchers": [],
            "report_outline": {},
            "report_sections": [],
            "report_history": [],
            "current_report_id": 0,
            "report_content": "",
            "report_summary": "",
            "report_conclusion": "",
            "report_recommendations": [],
            "report_references": [],
            "report_citations": [],
            "report_footnotes": [],
            "report_endnotes": [],
            "scores": {},
            "research_gaps": [],
            "revision_count": 0,
            "final_score": 0,
            "user_feedback": [],
            "feedback_incorporated": []
        }

        # Build config with any overrides
        run_config = {"configurable": {}}
        if config:
            run_config["configurable"] = config.model_dump()

        try:
            # Run the agent
            result = await agent.ainvoke(initial_state, run_config)

            # Extract the report content
            report_content = result.get("report_content", "")

            # If no report_content, try to get from messages
            if not report_content and result.get("messages"):
                for msg in reversed(result["messages"]):
                    if hasattr(msg, "content") and len(msg.content) > 500:
                        report_content = msg.content
                        break

            return {
                "report": report_content,
                "topic": result.get("topic", topic),
                "subtopics": result.get("subtopics", []),
                "final_score": result.get("final_score", 0),
                "references": result.get("report_references", [])
            }
        except Exception as e:
            print(f"Error running agent on topic '{topic[:50]}...': {e}")
            return {
                "report": f"Error: {str(e)}",
                "topic": topic,
                "error": str(e)
            }

    def sync_wrapper(inputs: dict) -> dict:
        """Synchronous wrapper for the async agent."""
        return asyncio.run(run_agent(inputs))

    return sync_wrapper


def report_quality_evaluator(run: Run, example: Example) -> dict:
    """
    Evaluate the quality of a generated report.

    This is a simple evaluator - the main RACE score evaluation
    happens via Deep Research Bench's Gemini-based evaluator.

    Returns basic metrics for quick feedback during development.
    """
    outputs = run.outputs or {}
    report = outputs.get("report", "")

    # Basic quality checks
    has_content = len(report) > 100
    has_structure = any(marker in report for marker in ["##", "**", "1.", "-"])
    has_references = "reference" in report.lower() or "[" in report
    word_count = len(report.split())

    return {
        "results": [
            {"key": "has_content", "score": 1.0 if has_content else 0.0},
            {"key": "has_structure", "score": 1.0 if has_structure else 0.0},
            {"key": "has_references", "score": 1.0 if has_references else 0.0},
            {"key": "word_count", "score": min(word_count / 2000, 1.0)},  # Normalize to 2000 words
        ]
    }


def run_evaluation(
    experiment_name: str | None = None,
    limit: int | None = None,
    config: Configuration | None = None
) -> str:
    """
    Run evaluation on the Deep Research Bench dataset.

    Args:
        experiment_name: Name for the LangSmith experiment
        limit: Maximum number of examples to evaluate (for testing)
        config: Optional configuration overrides

    Returns:
        The experiment name/URL
    """
    client = Client()

    # Generate experiment name if not provided
    if not experiment_name:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = f"deep_research_eval_{timestamp}"

    print(f"Starting evaluation: {experiment_name}")
    print(f"Dataset: {DATASET_NAME}")
    if limit:
        print(f"Limit: {limit} examples")

    # Create the target function
    target = create_research_agent_target(config)

    # Run evaluation
    results = evaluate(
        target,
        data=DATASET_NAME,
        evaluators=[report_quality_evaluator],
        experiment_prefix=experiment_name,
        max_concurrency=2,  # Limit concurrency to avoid rate limits
        num_repetitions=1,
    )

    print(f"\nEvaluation complete!")
    print(f"Experiment: {experiment_name}")
    print(f"View results at: https://smith.langchain.com")

    return experiment_name


def main():
    parser = argparse.ArgumentParser(
        description="Run Deep Research Bench evaluation"
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name for the LangSmith experiment"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of examples to evaluate (for testing)"
    )
    parser.add_argument(
        "--research-model",
        type=str,
        default=None,
        help="Model to use for research (e.g., 'openai:gpt-4.1')"
    )
    parser.add_argument(
        "--summarization-model",
        type=str,
        default=None,
        help="Model to use for summarization"
    )

    args = parser.parse_args()

    # Build configuration if models specified
    config = None
    if args.research_model or args.summarization_model:
        config = Configuration(
            research_model=args.research_model,
            summarization_model=args.summarization_model
        )

    # Run evaluation
    experiment_name = run_evaluation(
        experiment_name=args.experiment_name,
        limit=args.limit,
        config=config
    )

    print(f"\nNext steps:")
    print(f"1. Wait for the experiment to complete")
    print(f"2. Extract results with:")
    print(f"   python tests/extract_langsmith_data.py --project-name '{experiment_name}' --model-name 'your-model' --dataset-name 'deep_research_bench'")


if __name__ == "__main__":
    main()
