from langgraph.types import RunnableConfig
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from utils.model import llm_quality
from utils.configuration import get_config_from_configurable
from utils.subresearcher import subresearcher_graph
from utils.db import save_report
from .helpers import format_source_as_markdown_link
from pydantic import create_model
import re, json, asyncio, uuid



async def generate_plan_and_research(state: dict, config: RunnableConfig) -> dict:
    """
    1. Generate outline with structured output (subtopics with queries)
    2. Immediately launch parallel subresearchers for each subtopic
    3. Return outline + research results
    """
    agent_config = get_config_from_configurable(config.get("configurable", {}))
    max_subtopics = agent_config.max_subtopics
    topic = state.get("topic", "")

    outline_prompt = f"""You are a research planning specialist. Create a comprehensive research plan for this topic.

MAIN RESEARCH QUESTION: {topic}

CRITICAL: First, determine what TYPE of question this is, then organize subtopics accordingly:

**TYPE A - ENTITY-FOCUSED** (listing/comparing specific items):
Examples: "Best mobile games 2024", "Top K-pop songs", "MMORPG recommendations"
→ Create THEMATIC subtopics that each cover MULTIPLE entities
→ Use RESEARCH QUESTIONS, not specific entity names
→ Example: "Established MMORPGs with Expansions" (not "WoW", "FFXIV")

**TYPE B - THEMATIC/ANALYTICAL** (explaining concepts, comparing philosophies):
Examples: "Investment philosophies of Buffett vs Munger", "Impact of AI on labor"
→ Create subtopics by KEY THEMES/ASPECTS
→ Subtopic titles = Themes/dimensions (e.g., "Risk Management", "Decision Process")

**TYPE C - MARKET/TECHNICAL ANALYSIS** (sizing markets, evaluating technologies):
Examples: "Elderly consumption Japan 2020-2050", "Scaling quantum computing"
→ Structure by ANALYTICAL COMPONENTS
→ Subtopic titles = Analysis dimensions (e.g., "Market Size", "Technology Roadmap")

**TYPE D - COMPREHENSIVE RESEARCH** (multi-faceted exploration):
Examples: "AI in K-12 education", "Gut microbiota and cancer"
→ Organize by LOGICAL FLOW (background → mechanisms → applications → implications)

For each subtopic, provide:
- subtopic: Subtopic title (theme/aspect/dimension)
- queries: List of dictionary with 3-5 specific search queries and their respective priority levels (high / medium / low) 

MAX NUMBER OF SUBTOPICS: {max_subtopics}

**QUERY SPECIFICITY RULES:**
1. Location specificity: Include country/region in EVERY query if mentioned
2. Data type specificity: Request quantitative data explicitly
3. Year/timeframe specificity: Include exact years or ranges
4. Official source targeting: Include keywords for authoritative data

Examples:
✅ GOOD: "Japan elderly population 2020 2030 2050 projections Ministry statistics"
❌ BAD: "elderly population trends"
"""

    # Define structured output schema with explicit types
    Query = create_model(
        'Query',
        query=(str, ...),
        priority=(str, ...),
        __config__={'extra': 'forbid'}
    )

    Subtopic = create_model(
        'Subtopic',
        subtopic=(str, ...),
        queries=(list[Query], ...),
        __config__={'extra': 'forbid'}
    )

    ResearchPlan = create_model(
        'ResearchPlan',
        subtopics=(list[Subtopic], ...),
        __config__={'extra': 'forbid'}
    )

    messages = [
        SystemMessage(content="You are an expert research planning specialist. Create comprehensive research plans with specific, actionable search queries."),
        HumanMessage(content=outline_prompt)
    ]

    # Generate outline with structured output
    llm_structured = llm_quality.with_structured_output(ResearchPlan)

    print(f"[OUTLINE] Generating research plan with LLM...")
    response = await llm_structured.ainvoke(messages)

    # Convert Pydantic models to dicts for easier processing
    subtopics = [s.dict() if hasattr(s, 'dict') else s for s in response.subtopics]

    print(f"[OUTLINE] Created plan with {len(subtopics)} subtopics:")
    for section in subtopics[:3]:
        subtopic_title = section.get("subtopic", "")
        queries = section.get("queries", [])
        print(f"  - {subtopic_title}: {len(queries)} queries")
    if len(subtopics) > 3:
        print(f"  ... and {len(subtopics) - 3} more subtopics")

    print(f"\n[RESEARCH] Starting parallel research for {len(subtopics)} subtopics...\n")

    async def process_subtopic(idx: int, subtopic: dict):
        # Process a single section through subresearcher
        subtopic_title = subtopic.get("subtopic", "")
        subtopic_queries = subtopic.get("queries", [])

        # Build research plan from queries
        research_plan = []
        if subtopic_queries and isinstance(subtopic_queries, list):
            for query_spec in subtopic_queries:
                if isinstance(query_spec, dict):
                    research_plan.append({
                        "query": query_spec.get("query", ""),
                        "priority": query_spec.get("priority", "medium")
                    })
                elif isinstance(query_spec, str):
                    research_plan.append({
                        "query": query_spec,
                        "priority": "medium"
                    })


        print(f"  [{idx + 1}/{len(subtopics)}] {subtopic_title}: {len(research_plan)} queries")

        # Launch subresearcher
        subgraph_state = {
            "subtopic_id": idx,
            "subtopic": subtopic_title,
            "main_topic": topic,
            "other_subtopics": [s.get("subtopic", "") for s in subtopics],
            "section_subtopics": [q.get("query", "") if isinstance(q, dict) else q for q in subtopic_queries],
            "research_results": {},
            "research_depth": 1,
            "source_credibilities": {},
            "source_relevance_scores": {},
            "entities": [],
            "research_plan": research_plan,
            "completed_searches": 0,
            "max_search_results": agent_config.max_search_results,
            "max_research_depth": agent_config.max_research_depth,
            "search_api": agent_config.search_api,
        }

        result = await subresearcher_graph.ainvoke(subgraph_state)

        return {
            "subtopic_id": idx,
            "subtopic": subtopic_title,
            "research_results": result.get("research_results", {}),
            "source_credibilities": result.get("source_credibilities", {}),
            "research_depth": result.get("research_depth", 1)
        }

    # Execute all subresearchers in parallel
    tasks = [process_subtopic(idx, subtopic) for idx, subtopic in enumerate(subtopics)]
    sub_researchers = await asyncio.gather(*tasks)

    print(f"\n{'='*60}")
    print(f"COMBINED NODE: Completed research for {len(sub_researchers)} subtopics")
    print(f"{'='*60}")

    total_sources = 0
    for researcher in sub_researchers:
        sources = len(researcher.get("research_results", {}))
        total_sources += sources
        print(f"  ✓ {researcher.get('subtopic', 'Unknown')}: {sources} sources")

    print(f"\nTotal sources gathered: {total_sources}\n")

    # Update subtopics with researched subtopic names for write_full_report
    # Each subtopic dict now has a "subtopics" field containing the actual researched subtopic name
    for idx, subtopic_dict in enumerate(subtopics):
        subtopic_dict["subtopics"] = [sub_researchers[idx].get("subtopic", "")]

    # Build final outline structure
    outline = {
        "subtopics": subtopics  # List of {subtopic: str, queries: list, subtopics: list}
    }

    return {
        "messages": [AIMessage(content=f"Generated research plan with {len(subtopics)} subtopics and gathered {total_sources} sources")],
        "report_outline": outline,
        "sub_researchers": sub_researchers,
    }


async def write_full_report(state: dict, config: RunnableConfig) -> dict:
    """
    Write the entire report in ONE LLM call for maximum coherence and efficiency.
    Uses all research results and outline to generate a comprehensive, well-structured report.
    """
    agent_config = get_config_from_configurable(config.get("configurable", {}))

    topic = state.get("topic", "")
    outline = state.get("report_outline", {})
    subtopics = outline.get("subtopics", [])  # List of {subtopic: str, queries: list, subtopics: list}
    sub_researchers = state.get("sub_researchers", [])
    revision_count = state.get("revision_count", 0)

    print(f"write_full_report: generating complete report for topic='{topic[:50]}...' (revision {revision_count})")
    print(f"write_full_report: using min_credibility_score={agent_config.min_credibility_score}")

    # Build comprehensive research context from all subtopics
    research_by_subtopic = {}
    all_sources = []
    total_sources = 0

    for researcher in sub_researchers:
        subtopic = researcher.get("subtopic", "")
        results = researcher.get("research_results", {})
        credibilities = researcher.get("source_credibilities", {})

        # Filter and sort by credibility
        credible_sources = [
            (source, findings, credibilities.get(source, 0.5))
            for source, findings in results.items()
            if credibilities.get(source, 0.5) >= agent_config.min_credibility_score
        ]
        credible_sources.sort(key=lambda x: x[2], reverse=True)

        research_by_subtopic[subtopic] = credible_sources[:25]  # Top 25 per subtopic
        total_sources += len(credible_sources[:25])

    print(f"write_full_report: {total_sources} total high-quality sources available")

    # DEBUG: Show what we have
    print(f"write_full_report: DEBUG - research_by_subtopic has {len(research_by_subtopic)} subtopics")
    for subtopic_name, sources in research_by_subtopic.items():
        print(f"  - {subtopic_name}: {len(sources)} sources")

    # Build comprehensive sources string - USE RESEARCHER DATA DIRECTLY
    all_sources_text = ""
    source_index = 1
    all_sources = []

    # FIXED: Iterate through sub_researchers directly instead of trying to map through outline
    for researcher in sub_researchers:
        subtopic_name = researcher.get("subtopic", "")

        if subtopic_name not in research_by_subtopic:
            print(f"write_full_report: WARNING - '{subtopic_name}' not in research_by_subtopic!")
            continue

        sources_for_subtopic = research_by_subtopic[subtopic_name]
        print(f"write_full_report: Processing '{subtopic_name}': {len(sources_for_subtopic)} sources")

        all_sources_text += f"\n\n=== SOURCES FOR SUBTOPIC: {subtopic_name} ===\n"

        for source, findings, credibility in sources_for_subtopic:
            all_sources_text += f"\n[{source_index}] {source} (credibility: {credibility:.2f})\n{findings}\n"
            all_sources.append(source)
            source_index += 1

    print(f"write_full_report: Built sources text with {source_index - 1} total sources")
    print(f"write_full_report: all_sources_text length: {len(all_sources_text)} characters")

    # Generate report with structured output for title and sources
    report_with_title_model = create_model(
        'ReportWithTitle',
        report_title=(str, ...),
        content=(str, ...),  # Full markdown report with ## headers
        sources=(list[int], ...)  # List of source numbers actually cited in the report
    )

    report_prompt = f"""
You are an expert research report writer. Write a comprehensive, professional research report on:

**TOPIC**: {topic}

**TOTAL SOURCES AVAILABLE**: {total_sources} high-quality sources

**ALL RESEARCH SOURCES** (cite using [1], [2], etc.):
{all_sources_text}

**CRITICAL REQUIREMENTS:**

1. **ANTI-HALLUCINATION (MANDATORY)**:
   - ONLY use information EXPLICITLY stated in the sources above
   - EVERY fact, statistic, figure, name, or claim MUST have a citation [1], [2], etc.
   - DO NOT use your general knowledge - if it's not in the sources, don't include it
   - If you can't find a source for a claim, LEAVE IT OUT entirely

2. **COMPREHENSIVE SOURCE UTILIZATION**:
   - You MUST use AT LEAST 70% of available sources (aim for {int(total_sources * 0.7)}+ citations)
   - Each major section should cite 10-20 different sources minimum
   - Synthesize information from MULTIPLE sources for each key point
   - Extract ALL quantitative data, statistics, and specific examples from sources
   - Create detailed, data-rich sections using the extensive research provided

3. **QUANTITATIVE DATA (REQUIRED for market/economic topics)**:
   - Include EVERY specific number found in sources: revenue, market size, percentages, projections, dates
   - Present data with full context: time periods, geographical scope, source organization
   - Example: "According to the Ministry of Internal Affairs, elderly population reached 36 million in 2020, projected to rise to 39 million by 2050 [3]"
   - Example: "The adaptive clothing market is growing at 9.8% CAGR from 2025-2033 [15]"
   - Compare data points across sources when available

4. **DETAILED SECTIONS - BALANCED COVERAGE**:
   - Each major section MUST be 400-600 words minimum (enforce this strictly)
   - All sections should have similar depth and detail - avoid having some sections more developed than others
   - Include multiple subsections (use ### for subsections)
   - Present information in depth, not superficially
   - Use specific examples, case studies, and detailed explanations from sources
   - Build comprehensive narrative by weaving together information from many sources

5. **CITATION DISCIPLINE (STRICTLY ENFORCED)**:
   - EVERY sentence containing factual information MUST end with a citation
   - Citations MUST be placed immediately after the fact: "X happened in 2020 [5]"
   - Multi-source citations: "Population grew 15% [3][7][12]"
   - ONLY cite sources whose information you actually used in the text
   - Each paragraph should have 3-5 citations minimum
   - General statements without citations will be considered hallucinations
   - Format: Always use [X] format, never footnotes or other styles

6. **REPORT STRUCTURE**:
   - Generate a professional, refined title (in report_title field)
   - Use markdown headers (## for main sections, ### for subsections)
   - Write cohesive narrative that flows naturally
   - Each section should be substantial and evidence-rich
   - Organize logically with clear progression of ideas

7. **QUALITY CHECKS**:
   - Count your citations before submitting - you should have {int(total_sources * 0.7)}+ unique source citations
   - Verify every citation number corresponds to actual information you used from that source
   - Remove any placeholder citations or citations to sources not actually used
   - Ensure no unsupported claims remain in the text

8. **LANGUAGE**:
   - Write in the same language as the topic so the user understands
   - Use clear, professional language appropriate for a research report

Return in 'content' field: Complete markdown report WITHOUT the title header or References section.
Start directly with the first main section as a ## header.
I will add the title and References section separately.

Return in 'report_title' field: A clear, professional title for this report.

Return in 'sources' field: A list of ALL source numbers you actually cited in the report (e.g., [1, 3, 5, 7, 12]).
This should include EVERY source number that appears in your citations [X].
CRITICAL: Only include sources you actually used. Do NOT include sources you didn't cite.
"""

    messages = [
        SystemMessage(content="You are an expert research report writer specializing in comprehensive, evidence-based reports. CRITICAL: Only use information from provided sources. Every claim must be cited."),
        HumanMessage(content=report_prompt)
    ]

    llm_with_structured = llm_quality.with_structured_output(report_with_title_model)

    print(f"write_full_report: invoking LLM for complete report generation...")
    response = await llm_with_structured.ainvoke(messages)

    report_content = response.content
    report_title = response.report_title
    cited_source_numbers = response.sources  # Get sources from structured output

    print(f"write_full_report: generated report with title='{report_title}'")

    # Use sources from structured output (primary method)
    unique_citations = sorted(set(cited_source_numbers))

    # Fallback: Extract citations from report text if structured output is empty
    if not unique_citations:
        print("write_full_report: WARNING - No sources in structured output, falling back to regex extraction")
        citations_in_content = re.findall(r'\[(\d+)\]', report_content)
        unique_citations = sorted(set(int(c) for c in citations_in_content if c.isdigit()))

    citation_count = len(unique_citations)

    print(f"write_full_report: Total sources available: {len(all_sources)}")
    print(f"write_full_report: Citations from structured output: {unique_citations}")
    print(f"write_full_report: Number of unique citations: {citation_count}")

    # Build references section with ONLY cited sources
    full_report = f"# {report_title}\n\n{report_content}\n\n"
    full_report += "## References\n\n"

    # Only include sources that were actually cited
    cited_sources = []
    for citation_num in unique_citations:
        if citation_num <= len(all_sources):
            source = all_sources[citation_num - 1]  # citation_num is 1-indexed
            formatted_source = format_source_as_markdown_link(source)
            full_report += f"[{citation_num}] {formatted_source}\n"
            cited_sources.append(source)

    print(f"write_full_report: Number of sources added to references: {len(cited_sources)}")

    full_report_message = AIMessage(content=full_report)

    # Generate report_id if not exists
    report_id = state.get("report_id", "")
    if not report_id:
        report_id = f"{uuid.uuid4().hex[:12]}"
        print(f"write_full_report: generated new report_id={report_id}")

    new_version_id = state.get("version_id", 0) + 1

    # Build search_results for MongoDB (maps subtopic → research data)
    mongodb_search_results = {}
    for researcher in sub_researchers:
        subtopic_name = researcher.get("subtopic", "")
        mongodb_search_results[subtopic_name] = {
            "results": researcher.get("research_results", {}),
            "credibilities": researcher.get("source_credibilities", {})
        }

    # Save report to MongoDB
    try:
        await save_report(
            report_id=report_id,
            version_id=new_version_id,
            full_report=full_report,
            report_title=report_title,
            search_results=mongodb_search_results,
            report_sections=[
                {
                    "title": subtopic_item.get("subtopic", ""),
                    "subtopics": subtopic_item.get("subtopics", []),
                    "source_count": len(research_by_subtopic.get(subtopic_item.get("subtopics", [""])[0], []))
                }
                for subtopic_item in subtopics
            ]
        )
        print(f"write_full_report: saved report {report_id} version {new_version_id} to MongoDB")
    except Exception as e:
        print(f"write_full_report: failed to save report to MongoDB: {e}")

    return {
        "report_id": report_id,
        "report_content": full_report,
        "version_id": new_version_id,
        "report_history": state.get("report_history", []) + [new_version_id],
        "messages": [full_report_message]
    }

