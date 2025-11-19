from typing import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.types import RunnableConfig, interrupt
from pydantic import BaseModel, create_model
from utils.model import llm
from utils.subresearcher import subresearcher_graph
import asyncio



# ============================================================================
# NODES
# ============================================================================

def check_initial_context(state: dict) -> dict:
    """
    Check if initial topic has enough context to proceed.
    If sufficient, mark as finalized. If not, prepare for clarification loop.
    """
    topic = state.get("topic", "")
    print(f"check_initial_context: topic='{topic[:50]}...'")

    # Basic validation checks
    if not topic or len(topic.strip()) == 0:
        print("check_initial_context: no topic found, returning is_finalized=False")
        return {
            "is_finalized": False
        }

    # Use LLM to evaluate if initial topic has enough context
    evaluation_prompt = f"""
    You are a strict research coordinator. Evaluate if the following is a valid and detailed research topic.

    Topic: "{topic}"

    IMPORTANT: Be strict. The topic must be:
    1. A clear research question or subject (NOT a greeting, command, or casual message)
    2. Specific enough to generate meaningful subtopics
    3. Contains enough detail to understand what to research
    4. Well-defined in scope

    Examples of INSUFFICIENT topics:
    - "hi", "hello", "help me"
    - "AI" (too broad, needs specifics)
    - "research this" (no subject)
    - Single words or very vague phrases

    Examples of SUFFICIENT topics:
    - "The impact of climate change on coastal cities in Southeast Asia"
    - "Recent advances in quantum computing applications for cryptography"
    - "Effectiveness of remote work on employee productivity in tech companies"

    Respond with ONLY "SUFFICIENT" or "INSUFFICIENT" followed by a brief reason (one sentence).
    """

    messages = [
        SystemMessage(content="You are a strict research coordinator that only accepts well-defined research topics. Reject greetings, commands, and vague phrases."),
        HumanMessage(content=evaluation_prompt)
    ]

    response = llm.invoke(messages)
    evaluation = response.content.strip() if hasattr(response, 'content') else str(response).strip()

    # Check if context is sufficient (must explicitly say SUFFICIENT)
    is_sufficient = evaluation.upper().startswith("SUFFICIENT")
    print(f"check_initial_context: evaluation={is_sufficient}, response='{evaluation[:150]}...'")

    return {
        "is_finalized": is_sufficient
    }


def generate_clarification_question(state: dict, config: RunnableConfig) -> dict:
    """
    Generate a clarification question based on current topic and previous responses.
    This node uses LLM to determine what information is still needed.
    """
    MAX_CLARIFICATION_ROUNDS = config.get("configurable", {}).get("max_clarification_rounds", 5)
    
    clarification_rounds = state.get("clarification_rounds", 0)
    topic = state.get("topic", "")
    clarification_questions = state.get("clarification_questions", [])
    user_responses = state.get("user_responses", [])
    
    print(f"generate_clarification_question: round={clarification_rounds}, topic='{topic[:50]}...'")
    
    # Safety check: if we've asked too many questions, proceed anyway
    if clarification_rounds >= MAX_CLARIFICATION_ROUNDS:
        print(f"generate_clarification_question: max rounds reached ({MAX_CLARIFICATION_ROUNDS}), finalizing")
        return {
            "is_finalized": True,
            "clarification_rounds": clarification_rounds
        }
    
    # Build context from previous interactions
    context = f"Topic: {topic}\n"
    if clarification_questions:
        context += "\nPrevious questions asked:\n"
        for i, (q, r) in enumerate(zip(clarification_questions, user_responses)):
            context += f"Q{i+1}: {q}\nA{i+1}: {r}\n"
    
    # Use LLM to generate a clarification question
    clarification_prompt = f"""
    You are a research assistant helping to clarify a research topic.
    
    Current topic: {topic}
    
    {context if clarification_rounds > 0 else "This is the initial topic."}
    
    Generate a specific clarification question to better understand what the user wants to research.
    The question should help narrow down the scope, focus, or specific aspects of the topic.
    
    If you have enough information to proceed with research, respond with "ENOUGH_CONTEXT" instead of a question.
    
    Return ONLY the question (or "ENOUGH_CONTEXT"), nothing else.
    """
    
    messages = [
        SystemMessage(content="You are a helpful research assistant that asks clarifying questions to better understand research topics."),
        HumanMessage(content=clarification_prompt)
    ]

    response = llm.invoke(messages)
    question = response.content.strip() if hasattr(response, 'content') else str(response).strip()

    # Check if LLM says we have enough context
    if question.upper() == "ENOUGH_CONTEXT" or "enough context" in question.lower():
        print("generate_clarification_question: LLM indicated enough context")
        return {
            "is_finalized": True,
            "clarification_rounds": clarification_rounds
        }
    
    # Add the question to the list
    new_questions = clarification_questions + [question]
    print(f"generate_clarification_question: generated question='{question[:100]}...'")
    clarification_question = AIMessage(content=question)
    
    return {
        "messages": [clarification_question],
        "clarification_questions": new_questions,
        "clarification_rounds": clarification_rounds + 1
    }


def collect_user_response(state: dict) -> dict:
    """
    Collect user's response to the clarification question.
    Uses interrupt() to pause execution and wait for user input via Command(resume=...).
    """
    messages = state.get("messages", [])
    topic = state.get("topic", "")
    user_responses = state.get("user_responses", [])

    print(f"collect_user_response: waiting for user input (current topic: '{topic[:50]}...')")

    # Get the last AI message (the clarification question)
    last_message = messages[-1] if messages else None
    question = last_message.content if last_message else "Please provide your response"

    # Call interrupt() - this pauses execution and waits for Command(resume=user_input)
    user_response = interrupt(question)

    print(f"collect_user_response: received user response '{user_response[:50]}...'")

    # Add response to the list
    new_responses = user_responses + [user_response]

    return {
        "user_responses": new_responses,
        "messages": [HumanMessage(content=user_response)]
    }



class TopicEvaluation(BaseModel):
    is_sufficient: bool
    finalized_topic: str


async def validate_context_after_clarification(state: dict, config: RunnableConfig) -> dict:
    """
    Validate if we have enough context after collecting clarification responses.
    Uses LLM to evaluate if the topic and clarifications provide sufficient info.
    """
    MAX_CLARIFICATION_ROUNDS = config.get("configurable", {}).get("max_clarification_rounds", 5)

    clarification_rounds = state.get("clarification_rounds", 0)
    topic = state.get("topic", "")
    clarification_questions = state.get("clarification_questions", [])
    user_responses = state.get("user_responses", [])

    # Build full context string
    context = f"Research Topic: {topic}\n\n"

    if clarification_questions:
        context += "Clarification Q&A:\n"
        for i, (q, r) in enumerate(zip(clarification_questions, user_responses)):
            context += f"Q{i + 1}: {q}\nA{i + 1}: {r}\n\n"

    structured_llm = llm.with_structured_output(TopicEvaluation)

    validation_prompt = f"""
You are evaluating whether there is enough information to conduct comprehensive research.

{context}

Your task:
1. Decide whether the information is sufficient to finalize the topic and proceed with research.
2. Provide your recommended finalized topic regardless of sufficiency.

Definition of sufficiency:
- Topic is specific and well-scoped.
- Clear research directions exist.
- Enough context is present to create subtopics, research steps, and a detailed report.

IMPORTANT:
Return ONLY the following fields in JSON form for structured parsing:
- `is_sufficient`: true/false
- `finalized_topic`: string
"""
    messages = [
        SystemMessage(content="You are a research coordinator evaluating the completeness of context."),
        HumanMessage(content=validation_prompt)
    ]

    response: TopicEvaluation = await structured_llm.ainvoke(messages)

    is_sufficient = bool(response.is_sufficient)
    finalized_topic = response.finalized_topic.strip()

    print(f"validate_context_after_clarification: is_sufficient={is_sufficient}, rounds={clarification_rounds}")

    # If we hit the round limit, finalize no matter what
    if clarification_rounds >= MAX_CLARIFICATION_ROUNDS:
        return {
            "is_finalized": True,
            "topic": finalized_topic,
            "clarification_rounds": clarification_rounds
        }

    # If sufficient, finalize
    if is_sufficient:
        return {
            "is_finalized": True,
            "topic": finalized_topic
        }

    # Otherwise request more clarification
    return {"is_finalized": False}


async def generate_subtopics(state: dict) -> dict:
    """Generate subtopics and create subresearchers using the subresearcher subgraph with multi-layer research"""
    # Read topic from state (works with unified state)
    topic = state.get("topic", "")
    print(f"generate_subtopics: starting for topic='{topic[:50]}...'")

    # Create structured output model for subtopics
    SubtopicsOutput = create_model(
        'SubtopicsOutput',
        subtopics=(list[str], ...)
    )

    structured_llm = llm.with_structured_output(SubtopicsOutput)

    prompt = f"""
    You are a helpful assistant that generates subtopics for a research report.
    Come up with 3-5 subtopics for the given topic based on the topic's scope and complexity.

    Topic: {topic}
    """

    llm_response = await structured_llm.ainvoke([
        SystemMessage(content="You are a helpful assistant that generates subtopics for a research report. Come up with 3-5 subtopics for the given topic based on the topic's scope and complexity."),
        HumanMessage(content=prompt)
    ])

    subtopics = llm_response.subtopics
    print(f"generate_subtopics: generated {len(subtopics)} subtopics")

    # Generate subresearchers for each subtopic using the enhanced subgraph in parallel
    async def process_subtopic(idx: int, subtopic: str):
        """Process a single subtopic through the multi-layer subresearcher subgraph"""
        subgraph_state = {
            "subtopic_id": idx,
            "subtopic": subtopic,
            "research_results": {},
            "research_depth": 1,  # Start at layer 1
            "source_credibilities": {},
            "follow_up_queries": []
        }

        # Invoke the enhanced subresearcher subgraph (now with multi-layer research)
        result = await subresearcher_graph.ainvoke(subgraph_state)

        # Return comprehensive research results with credibility info
        return {
            "subtopic_id": idx,
            "subtopic": subtopic,
            "research_results": result.get("research_results", {}),
            "source_credibilities": result.get("source_credibilities", {}),
            "research_depth": result.get("research_depth", 1)
        }

    # Process all subtopics in parallel
    tasks = [process_subtopic(idx, subtopic) for idx, subtopic in enumerate(subtopics)]
    print(f"generate_subtopics: processing {len(tasks)} subtopics in parallel with multi-layer research")
    sub_researchers = await asyncio.gather(*tasks)
    print(f"generate_subtopics: completed processing {len(sub_researchers)} sub_researchers")

    # Log research depth achieved
    for researcher in sub_researchers:
        depth = researcher.get("research_depth", 1)
        sources = len(researcher.get("research_results", {}))
        print(f"  - {researcher.get('subtopic', 'Unknown')}: {sources} sources, depth {depth}")

    return {
        "sub_researchers": [r for r in sub_researchers],
        "subtopics": subtopics,
    }


async def generate_outline(state: dict) -> dict:
    """
    Generate a structured outline for the report based on research
    Maps subtopics to sections with specific focus areas
    """
    topic = state.get("topic", "")
    sub_researchers = state.get("sub_researchers", [])

    print(f"generate_outline: creating outline for topic='{topic[:50]}...'")

    # Build research overview
    research_overview = f"Topic: {topic}\n\nSubtopics researched:\n"
    for researcher in sub_researchers:
        subtopic = researcher.get("subtopic", "")
        sources = len(researcher.get("research_results", {}))
        research_overview += f"- {subtopic} ({sources} sources)\n"

    outline_prompt = f"""
    You are a report outline specialist. Create a structured outline for a comprehensive research report.

    {research_overview}

    Create an outline with:
    1. Executive Summary
    2. Introduction (context and scope)
    3. Main sections (3-5 sections covering the subtopics)
    4. Analysis and Key Findings
    5. Conclusion and Recommendations

    For each main section, specify:
    - Section title
    - Which subtopics it covers
    - Key questions to address

    Return as a JSON object with this structure:
    {{
      "sections": [
        {{
          "title": "Section Title",
          "subtopics": ["subtopic1", "subtopic2"],
          "key_questions": ["question1", "question2"]
        }}
      ]
    }}
    """

    messages = [
        SystemMessage(content="You are a report outline specialist that creates structured outlines for research reports."),
        HumanMessage(content=outline_prompt)
    ]

    response = await llm.ainvoke(messages)
    response_text = response.content if hasattr(response, 'content') else str(response)

    # Parse JSON outline (simplified)
    import json
    import re
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            outline = json.loads(json_match.group())
        else:
            # Fallback: simple outline based on subtopics
            outline = {
                "sections": [
                    {"title": "Introduction", "subtopics": [], "key_questions": []},
                    *[{"title": r.get("subtopic", ""), "subtopics": [r.get("subtopic", "")], "key_questions": []}
                      for r in sub_researchers],
                    {"title": "Conclusion", "subtopics": [], "key_questions": []}
                ]
            }
    except Exception as e:
        print(f"generate_outline: error parsing JSON, using fallback: {e}")
        outline = {
            "sections": [
                {"title": "Introduction", "subtopics": [], "key_questions": []},
                *[{"title": r.get("subtopic", ""), "subtopics": [r.get("subtopic", "")], "key_questions": []}
                  for r in sub_researchers],
                {"title": "Conclusion", "subtopics": [], "key_questions": []}
            ]
        }

    print(f"generate_outline: created outline with {len(outline.get('sections', []))} sections")

    return {
        "report_outline": outline,
        "report_sections": []
    }


async def write_sections_with_citations(state: dict) -> dict:
    """
    Write each section of the report with proper inline citations
    Uses research results to build evidence-based sections
    """
    topic = state.get("topic", "")
    outline = state.get("report_outline", {})
    sub_researchers = state.get("sub_researchers", [])

    print(f"write_sections_with_citations: writing sections for topic='{topic[:50]}...'")

    sections = outline.get("sections", [])
    written_sections = []

    # Build research lookup by subtopic
    research_by_subtopic = {}
    for researcher in sub_researchers:
        subtopic = researcher.get("subtopic", "")
        research_by_subtopic[subtopic] = {
            "results": researcher.get("research_results", {}),
            "credibilities": researcher.get("source_credibilities", {})
        }

    # Write each section
    for section in sections:
        section_title = section.get("title", "")
        section_subtopics = section.get("subtopics", [])

        print(f"  Writing section: {section_title}")

        # Gather relevant research for this section
        relevant_research = ""
        sources_list = []
        for subtopic in section_subtopics:
            if subtopic in research_by_subtopic:
                results = research_by_subtopic[subtopic]["results"]
                credibilities = research_by_subtopic[subtopic]["credibilities"]

                for source, findings in list(results.items())[:3]:  # Top 3 sources per subtopic
                    credibility = credibilities.get(source, 0.5)
                    relevant_research += f"\nSource: {source} (credibility: {credibility:.2f})\n{findings}\n"
                    sources_list.append(source)

        # If no specific research, use general overview
        if not relevant_research:
            relevant_research = "General context based on overall research findings."

        section_prompt = f"""
        Write the "{section_title}" section of a research report on: {topic}

        Research findings:
        {relevant_research[:3000]}  # Limit to avoid token issues

        Write a well-structured section with:
        - Clear topic sentences
        - Evidence from sources with inline citations like [1], [2], etc.
        - Logical flow and transitions
        - 2-4 paragraphs depending on content

        Include inline citations referencing the sources by number.
        """

        messages = [
            SystemMessage(content="You are a research report writer that creates well-structured, evidence-based sections with proper citations."),
            HumanMessage(content=section_prompt)
        ]

        response = await llm.ainvoke(messages)
        section_content = response.content if hasattr(response, 'content') else str(response)

        written_sections.append({
            "title": section_title,
            "content": section_content,
            "sources": sources_list
        })

    print(f"write_sections_with_citations: completed {len(written_sections)} sections")

    # Combine sections into full report
    full_report = f"# {topic}\n\n"
    all_sources = []

    for section in written_sections:
        full_report += f"## {section['title']}\n\n{section['content']}\n\n"
        all_sources.extend(section['sources'])

    # Add references section
    unique_sources = list(dict.fromkeys(all_sources))  # Remove duplicates, preserve order
    full_report += "## References\n\n"
    for i, source in enumerate(unique_sources, 1):
        full_report += f"[{i}] {source}\n"

    final_report_message = AIMessage(
        content=full_report,
        contentType="text/markdown"
    )

    return {
        "report_sections": written_sections,
        "report_content": full_report,
        "report_references": unique_sources,
        "messages": [final_report_message],
        "current_report_id": state.get("current_report_id", 0) + 1,
        "report_history": state.get("report_history", []) + [state.get("current_report_id", 0) + 1]
    }


async def evaluate_report(state: dict) -> dict:
    """
    Evaluate a report based on content quality, structure, and evidence
    Provides detailed feedback for potential improvements
    """
    report_content = state.get("report_content", "")
    current_report_id = state.get("current_report_id", 0)
    topic = state.get("topic", "")

    print(f"evaluate_report: starting for report_id={current_report_id}")

    evaluation_prompt = f"""
    You are an expert research report evaluator. Evaluate the following report on: {topic}

    Report:
    {report_content[:5000]}  # Limit for token management

    Evaluate based on:
    1. **Coverage** (0-25): Does it comprehensively cover the topic?
    2. **Evidence** (0-25): Are claims well-supported with sources?
    3. **Structure** (0-25): Is it well-organized and logical?
    4. **Clarity** (0-25): Is it clear and well-written?

    Provide your evaluation in this exact format:
    COVERAGE: [score]/25
    EVIDENCE: [score]/25
    STRUCTURE: [score]/25
    CLARITY: [score]/25
    TOTAL: [sum of above]/100
    FEEDBACK: [One paragraph of constructive feedback on what could be improved]
    """

    messages = [
        SystemMessage(content="You are an expert research report evaluator that provides detailed, constructive feedback."),
        HumanMessage(content=evaluation_prompt)
    ]

    response = await llm.ainvoke(messages)
    response_text = response.content if hasattr(response, 'content') else str(response)

    # Extract total score
    import re
    total_match = re.search(r'TOTAL:\s*(\d+)', response_text)
    score = int(total_match.group(1)) if total_match else 75

    scores = state.get("scores", {})
    scores[current_report_id] = score

    print(f"evaluate_report: score={score}/100 for report_id={current_report_id}")

    return {
        "scores": scores,
        "final_score": score
    }


async def identify_report_gaps(state: dict) -> dict:
    """
    Analyze the report and identify gaps that need addressing
    Uses gap analysis utility to find issues
    """
    from utils.gap_analysis import analyze_report_gaps

    topic = state.get("topic", "")
    report_content = state.get("report_content", "")
    sub_researchers = state.get("sub_researchers", [])
    final_score = state.get("final_score", 0)

    print(f"identify_report_gaps: analyzing report with score={final_score}")

    # Use gap analysis to find issues
    gaps = await analyze_report_gaps(topic, report_content, sub_researchers)

    # Convert ResearchGap objects to dicts for state storage
    gaps_dicts = [
        {
            "gap_description": gap.gap_description,
            "gap_type": gap.gap_type,
            "priority": gap.priority,
            "follow_up_query": gap.follow_up_query
        }
        for gap in gaps
    ]

    print(f"identify_report_gaps: found {len(gaps_dicts)} gaps")

    return {
        "research_gaps": gaps_dicts
    }


# ============================================================================
# STATE TRANSFORMATION NODES
# ============================================================================

async def transform_to_subtopic_state(state: dict) -> dict:
    """
    Transformation node: Convert TopicInquiryState to SubtopicGenerationState structure.
    This node adds the fields needed for subtopic generation.
    """
    # Extract TopicInquiryState fields
    topic = state.get("topic", "")
    messages = state.get("messages", [])
    
    # Add SubtopicGenerationState fields
    return {
        "subtopics": [],
        "sub_researchers": []
    }


async def transform_to_report_state(state: dict) -> dict:
    """
    Transformation node: Convert SubtopicGenerationState to ReportWriterState structure.
    This node adds the fields needed for report writing.
    """
    # Extract research data
    sub_researchers = state.get("sub_researchers", [])
    
    # Build references from research results
    references = []
    for researcher in sub_researchers:
        if isinstance(researcher, dict):
            research_results = researcher.get("research_results", {})
        else:
            research_results = getattr(researcher, "research_results", {})
        
        for source in research_results.keys():
            references.append(source)
    
    # Add ReportWriterState fields
    return {
        "report_history": [],
        "current_report_id": 0,
        "report_content": "",
        "report_summary": "",
        "report_conclusion": "",
        "report_recommendations": [],
        "report_references": references,
        "report_citations": [],
        "report_footnotes": [],
        "report_endnotes": []
    }


async def transform_to_evaluator_state(state: dict) -> dict:
    """
    Transformation node: Convert ReportWriterState to ReportEvaluatorState structure.
    This node adds the fields needed for report evaluation.
    """
    report_history = state.get("report_history", [])
    current_report_id = state.get("current_report_id", 0)
    
    return {
        "scores": {}
    }


# ============================================================================
# CONDITIONAL EDGES
# ============================================================================

def route_after_initial_check(state: dict) -> str:
    """
    Conditional edge: Route after checking initial context.
    - If sufficient context: proceed to subtopic generation (skip clarification loop)
    - If insufficient: enter clarification loop
    """
    is_finalized = state.get("is_finalized", False)
    route = "continue" if is_finalized else "ask_clarification"
    print(f"route_after_initial_check: is_finalized={is_finalized}, routing to '{route}'")
    return route


def route_after_clarification(state: dict) -> str:
    """
    Conditional edge: Route after collecting clarification response.
    - If sufficient context: proceed to subtopic generation
    - If insufficient: loop back to ask another clarification question
    """
    is_finalized = state.get("is_finalized", False)
    route = "continue" if is_finalized else "ask_clarification"
    print(f"route_after_clarification: is_finalized={is_finalized}, routing to '{route}'")
    return route


async def should_continue_to_report(state: dict) -> bool:
    """Check if the subtopics have been generated"""
    subtopics_count = len(state.get("subtopics", []))
    result = subtopics_count > 0
    print(f"should_continue_to_report: subtopics_count={subtopics_count}, returning {result}")
    return result


def route_after_evaluation(state: dict) -> str:
    """
    Conditional edge: Route after report evaluation
    - If score >= 85: Report is good, finalize
    - If score < 85 and revision_count < 2: Identify gaps and revise
    - If revision_count >= 2: Accept report even if not perfect
    """
    final_score = state.get("final_score", 0)
    revision_count = state.get("revision_count", 0)
    MAX_REVISIONS = 2

    print(f"route_after_evaluation: score={final_score}, revisions={revision_count}")

    # If score is good enough, finalize
    if final_score >= 85:
        print(f"route_after_evaluation: score {final_score} >= 85, finalizing")
        return "finalize"

    # If we've hit max revisions, accept current version
    if revision_count >= MAX_REVISIONS:
        print(f"route_after_evaluation: max revisions ({MAX_REVISIONS}) reached, finalizing")
        return "finalize"

    # Otherwise, identify gaps and revise
    print(f"route_after_evaluation: score {final_score} < 85, identifying gaps for revision")
    return "revise"


def route_after_gap_identification(state: dict) -> str:
    """
    Conditional edge: Route after identifying gaps
    - If significant gaps exist: Go back to outline/rewrite
    - If minor/no gaps: Finalize
    """
    research_gaps = state.get("research_gaps", [])
    revision_count = state.get("revision_count", 0)

    high_priority_gaps = [g for g in research_gaps if g.get("priority") == "high"]
    medium_priority_gaps = [g for g in research_gaps if g.get("priority") == "medium"]

    print(f"route_after_gap_identification: {len(high_priority_gaps)} high, {len(medium_priority_gaps)} medium priority gaps")

    # If significant gaps exist, revise
    if high_priority_gaps or medium_priority_gaps:
        print("route_after_gap_identification: significant gaps found, regenerating outline")
        return "regenerate"

    # Otherwise, finalize
    print("route_after_gap_identification: no significant gaps, finalizing")
    return "finalize"


