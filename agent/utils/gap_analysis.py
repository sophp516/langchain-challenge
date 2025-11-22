"""
Gap analysis utilities to identify research gaps and generate follow-up queries
"""
from typing import Dict, List
from langchain_core.messages import SystemMessage, HumanMessage
from utils.model import llm
from pydantic import BaseModel, create_model
import asyncio


class ResearchGap(BaseModel):
    """Represents a gap in research coverage"""
    gap_description: str
    gap_type: str  # 'missing_topic', 'insufficient_depth', 'conflicting_info', 'outdated'
    priority: str  # 'high', 'medium', 'low'
    follow_up_query: str
    affected_sections: list[str] = []  # Section titles that need revision for this gap


async def identify_research_gaps(
    topic: str,
    subtopics: List[str],
    research_results: List[Dict],
    current_depth: int = 1
) -> List[ResearchGap]:
    """
    Identify gaps in research coverage by analyzing what's been researched
    and what's missing or needs more depth
    """
    print(f"identify_research_gaps: analyzing {len(subtopics)} subtopics, {len(research_results)} results, depth={current_depth}")

    # Build summary of current research
    research_summary = f"Main Topic: {topic}\n\n"
    research_summary += f"Subtopics researched ({len(subtopics)}):\n"
    for i, subtopic in enumerate(subtopics):
        research_summary += f"{i+1}. {subtopic}\n"

    research_summary += f"\nResearch results summary:\n"
    for researcher in research_results:
        if isinstance(researcher, dict):
            subtopic = researcher.get("subtopic", "")
            results = researcher.get("research_results", {})
        else:
            subtopic = getattr(researcher, "subtopic", "")
            results = getattr(researcher, "research_results", {})

        source_count = len(results)
        research_summary += f"- {subtopic}: {source_count} sources\n"

    # Create structured output model for gaps
    GapsOutput = create_model(
        'GapsOutput',
        gaps=(list[dict], ...)  # List of gap dictionaries
    )

    structured_llm = llm.with_structured_output(GapsOutput)

    gap_analysis_prompt = f"""
    You are a research quality analyst. Analyze the following research and identify gaps.

    {research_summary}

    Current research depth: Layer {current_depth} of 3

    Identify research gaps in these categories:
    1. **missing_topic**: Important aspects of the main topic not covered by any subtopic
    2. **insufficient_depth**: Subtopics that need deeper investigation (few sources or shallow coverage)
    3. **conflicting_info**: Areas where sources might disagree and need clarification
    4. **outdated**: Time-sensitive topics that may need current information

    For each gap:
    - Provide a clear description
    - Classify the gap type
    - Assign priority (high/medium/low)
    - Generate a specific follow-up search query to address the gap

    Return a list of gap dictionaries with these exact keys:
    - gap_description: str
    - gap_type: str (one of: missing_topic, insufficient_depth, conflicting_info, outdated)
    - priority: str (one of: high, medium, low)
    - follow_up_query: str

    Identify 3-5 gaps maximum. Focus on the most important ones.
    If research seems comprehensive, you may return fewer gaps or an empty list.
    """

    messages = [
        SystemMessage(content="You are a research quality analyst that identifies gaps in research coverage."),
        HumanMessage(content=gap_analysis_prompt)
    ]

    try:
        response = await structured_llm.ainvoke(messages)
        gaps_data = response.gaps if hasattr(response, 'gaps') else []

        # Convert to ResearchGap objects
        gaps = []
        for gap_dict in gaps_data:
            try:
                gap = ResearchGap(
                    gap_description=gap_dict.get('gap_description', ''),
                    gap_type=gap_dict.get('gap_type', 'missing_topic'),
                    priority=gap_dict.get('priority', 'medium'),
                    follow_up_query=gap_dict.get('follow_up_query', '')
                )
                gaps.append(gap)
            except Exception as e:
                print(f"Error creating ResearchGap: {e}")
                continue

        print(f"identify_research_gaps: found {len(gaps)} gaps")
        return gaps

    except Exception as e:
        print(f"Error in identify_research_gaps: {e}")
        return []


async def analyze_report_gaps(
    topic: str,
    report_content: str,
    research_results: List[Dict],
    section_titles: List[str] = None
) -> List[ResearchGap]:
    """
    Analyze a draft report to identify gaps in coverage, logic, or evidence.
    Also identifies which sections are affected by each gap for targeted revision.
    """
    print(f"analyze_report_gaps: analyzing report of length {len(report_content)}")

    # Extract section titles from report if not provided
    if not section_titles:
        import re
        section_titles = re.findall(r'^## (.+)$', report_content, re.MULTILINE)
        print(f"analyze_report_gaps: extracted {len(section_titles)} section titles")

    # Build research summary
    research_summary = "Available research data:\n"
    for researcher in research_results[:5]:  # Limit to first 5 for brevity
        if isinstance(researcher, dict):
            subtopic = researcher.get("subtopic", "")
            results = researcher.get("research_results", {})
        else:
            subtopic = getattr(researcher, "subtopic", "")
            results = getattr(researcher, "research_results", {})

        research_summary += f"- {subtopic}: {len(results)} sources\n"

    sections_list = "\n".join(f"- {title}" for title in section_titles) if section_titles else "No sections found"

    gap_analysis_prompt = f"""
    You are a report quality analyst. Review this research report and identify gaps.

    Topic: {topic}

    {research_summary}

    Report sections:
    {sections_list}

    Report content:
    {report_content[:6000]}

    Identify gaps in the report:
    1. **missing_topic**: Important aspects mentioned in research but not covered in report
    2. **insufficient_depth**: Sections that lack detail or supporting evidence
    3. **conflicting_info**: Claims that may contradict available evidence
    4. **outdated**: Information that may be out of date

    For each gap, provide:
    - gap_description: Clear description of what's missing or problematic
    - gap_type: One of the four types above
    - priority: high, medium, or low
    - follow_up_query: Specific search query to fill this gap (or "N/A" if existing research suffices)
    - affected_sections: List of section titles that need revision to address this gap (use EXACT titles from the list above)

    Return 2-4 gaps maximum, focusing on the most critical issues.
    If the report seems comprehensive, return an empty list.

    Return as JSON array of objects with keys: gap_description, gap_type, priority, follow_up_query, affected_sections
    """

    messages = [
        SystemMessage(content="You are a report quality analyst that identifies gaps in research reports and maps them to specific sections."),
        HumanMessage(content=gap_analysis_prompt)
    ]

    try:
        response = await llm.ainvoke(messages)
        response_text = response.content if hasattr(response, 'content') else str(response)

        # Try to parse JSON
        import json
        import re as re_module

        # Extract JSON array if present
        json_match = re_module.search(r'\[.*\]', response_text, re_module.DOTALL)
        if json_match:
            gaps_data = json.loads(json_match.group())
        else:
            print("analyze_report_gaps: could not parse JSON, returning empty gaps")
            return []

        # Convert to ResearchGap objects
        gaps = []
        for gap_dict in gaps_data:
            try:
                affected = gap_dict.get('affected_sections', [])
                # Ensure affected_sections is a list
                if isinstance(affected, str):
                    affected = [affected]

                gap = ResearchGap(
                    gap_description=gap_dict.get('gap_description', ''),
                    gap_type=gap_dict.get('gap_type', 'missing_topic'),
                    priority=gap_dict.get('priority', 'medium'),
                    follow_up_query=gap_dict.get('follow_up_query', ''),
                    affected_sections=affected
                )
                gaps.append(gap)
            except Exception as e:
                print(f"Error creating ResearchGap from report analysis: {e}")
                continue

        print(f"analyze_report_gaps: found {len(gaps)} gaps")
        for gap in gaps:
            print(f"  - [{gap.priority}] {gap.gap_description[:50]}... -> sections: {gap.affected_sections}")
        return gaps

    except Exception as e:
        print(f"Error in analyze_report_gaps: {e}")
        return []


def prioritize_gaps(gaps: List[ResearchGap], max_gaps: int = 3) -> List[ResearchGap]:
    """
    Prioritize gaps to focus on the most important ones
    Returns top max_gaps gaps sorted by priority
    """
    if not gaps:
        return []

    # Sort by priority
    priority_order = {'high': 3, 'medium': 2, 'low': 1}

    sorted_gaps = sorted(
        gaps,
        key=lambda g: priority_order.get(g.priority, 0),
        reverse=True
    )

    return sorted_gaps[:max_gaps]


async def should_continue_research(
    gaps: List[ResearchGap],
    current_depth: int,
    max_depth: int = 3
) -> bool:
    """
    Determine if more research is needed based on gaps and current depth
    """
    # If at max depth, stop
    if current_depth >= max_depth:
        print(f"should_continue_research: max depth {max_depth} reached, stopping")
        return False

    # If no gaps, research is complete
    if not gaps:
        print(f"should_continue_research: no gaps found, research complete")
        return False

    # If only low-priority gaps remain, consider stopping
    high_priority_gaps = [g for g in gaps if g.priority == 'high']
    medium_priority_gaps = [g for g in gaps if g.priority == 'medium']

    if not high_priority_gaps and not medium_priority_gaps:
        print(f"should_continue_research: only low-priority gaps, stopping")
        return False

    print(f"should_continue_research: {len(high_priority_gaps)} high, {len(medium_priority_gaps)} medium priority gaps, continuing")
    return True