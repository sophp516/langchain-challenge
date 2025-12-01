"""
Unified State for LangGraph

This TypedDict contains all fields from all state types.
"""
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class UnifiedAgentState(TypedDict):
    """
    Unified state that can represent any state type.
    Starts with TopicInquiryState fields, nodes add/transform fields as needed.
    """

    messages: Annotated[list[BaseMessage], add_messages]

    # Intent routing
    user_intent: str  # "new_research", "retrieve_report", "list_reports", or "revise_report"
    intent_report_id: str  # Extracted report ID from user query
    intent_version_id: int  # Optional version_id for get_report, list_report_versions, revise_report
    intent_feedback: str  # Optional feedback text for revise_report

    # Topic inquiry
    report_id: str
    topic: str
    is_finalized: bool
    clarification_rounds: int
    clarification_questions: list[str]
    user_responses: list[str]
    pending_clarification_question: str  # Pre-generated question from check_initial_context

    # Report planning
    subtopics: list[str]
    sub_researchers: list[dict]  # SubResearcherState as dict with credibility info
    report_subtopics: list[dict]  # List of written sections with citations
    research_outline: dict

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

    # User feedback
    user_feedback: list[str]  # List of user's feedback on the report
    feedback_incorporated: list[bool]  # List tracking if each feedback has been addressed

    # Shared knowledge pool for cross-section learning
    shared_research_pool: dict  # Shared entities, search results, and findings across all sections
