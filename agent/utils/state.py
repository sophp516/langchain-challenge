"""
Unified State for LangGraph

This TypedDict contains all fields from all state types.
We start with TopicInquiryState fields, and nodes transform/add fields as needed.
"""
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from typing import Any
from langgraph.graph import add_messages


class UnifiedAgentState(TypedDict):
    """
    Unified state that can represent any state type.
    Starts with TopicInquiryState fields, nodes add/transform fields as needed.
    """
    # TopicInquiryState fields (starting state)
    messages: Annotated[list[BaseMessage], add_messages]
    report_id: str
    topic: str
    is_finalized: bool
    clarification_rounds: int
    clarification_questions: list[str]
    user_responses: list[str]

    # SubtopicGenerationState fields (added by transformation node)
    subtopics: list[str]
    sub_researchers: list[dict]  # SubResearcherState as dict with credibility info

    # Enhanced report planning fields
    report_outline: dict  # Structured outline with sections
    report_sections: list[dict]  # List of written sections with citations

    # ReportWriterState fields (added by transformation node)
    report_history: list[int]
    version_id: int
    report_content: str
    report_summary: str
    report_conclusion: str
    report_recommendations: list[str]
    report_references: list[str]
    report_citations: list[str]
    report_footnotes: list[str]
    report_endnotes: list[str]

    # ReportEvaluatorState fields (added by transformation node)
    scores: dict[int, int]

    # Iterative improvement fields
    revision_count: int  # Number of revision iterations
    final_score: int  # Final evaluation score
    evaluator_feedback: str  # Feedback from evaluator for revision

    # User feedback loop fields
    user_feedback: list[str]  # List of user's feedback on the report
    feedback_incorporated: list[bool]  # List tracking if each feedback has been addressed

    # Cross-reference verification fields
    verified_claims: list[dict]  # Claims verified across multiple sources
    conflicting_info: list[dict]  # Information that conflicts across sources

    # Intent routing fields
    user_intent: str  # "new_research", "retrieve_report", or "list_reports"
    intent_report_id: str  # Extracted report ID from user query
