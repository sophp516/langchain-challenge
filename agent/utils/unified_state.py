"""
Unified State for LangGraph

This TypedDict contains all fields from all state types.
We start with TopicInquiryState fields, and nodes transform/add fields as needed.
"""
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from typing import Any


class UnifiedAgentState(TypedDict):
    """
    Unified state that can represent any state type.
    Starts with TopicInquiryState fields, nodes add/transform fields as needed.
    """
    # TopicInquiryState fields (starting state)
    messages: Annotated[list[BaseMessage], "Chat messages"]
    topic: str
    is_finalized: bool
    clarification_rounds: int
    clarification_questions: list[str]
    user_responses: list[str]
    
    # SubtopicGenerationState fields (added by transformation node)
    subtopics: list[str]
    sub_researchers: list[dict]  # SubResearcherState as dict
    
    # ReportWriterState fields (added by transformation node)
    report_history: list[int]
    current_report_id: int
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
