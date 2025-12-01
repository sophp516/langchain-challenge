from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph import add_messages


class UnifiedAgentState(TypedDict):

    messages: Annotated[list[BaseMessage], add_messages]

    # Intent routing
    user_intent: str  # "new_research", "retrieve_report", "list_reports", or "revise_report"

    # Tool call arguments
    intent_report_id: str 
    intent_version_id: int 
    intent_feedback: str

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
    report_subtopics: list[dict]
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

    # Shared knowledge pool for cross-section learning
    shared_research_pool: dict  # Shared entities, search results, and findings across all sections
