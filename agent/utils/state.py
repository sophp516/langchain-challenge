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

    messages: Annotated[list[BaseMessage], add_messages]
    # TODO: Add UI messages if time

    # Intent routing
    user_intent: str  # "new_research", "retrieve_report", or "list_reports"
    intent_report_id: str  # Extracted report ID from user query

    # Conversational context tracking
    last_viewed_report_id: str  # Last report that was retrieved/viewed
    last_action: str  # Last action performed: "viewed_report", "revised_report", "created_report", etc.

    report_id: str
    topic: str
    is_finalized: bool
    clarification_rounds: int
    clarification_questions: list[str]
    user_responses: list[str]

    # SubtopicGenerationState fields (added by transformation node)
    subtopics: list[str]
    sub_researchers: list[dict]  # SubResearcherState as dict with credibility info

    # Report planning
    report_outline: dict  # Structured outline with sections
    report_sections: list[dict]  # List of written sections with citations

    # Report writing
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
