from langgraph.types import RunnableConfig
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from utils.model import configurable_model, get_fast_model_config, get_quality_model_config
from utils.configuration import get_config_from_configurable, tavily_client
from utils.subresearcher import subresearcher_graph
from utils.db import save_report
from .helpers import format_source_as_markdown_link
from pydantic import create_model
import re, json, asyncio, uuid



async def write_full_report(state: dict, config: RunnableConfig) -> dict:
    """
    Write the entire report in ONE LLM call for maximum coherence and efficiency.
    Uses all research results and outline to generate a comprehensive, well-structured report.
    """
    agent_config = get_config_from_configurable(config.get("configurable", {}))

    topic = state.get("topic", "")
    outline = state.get("report_outline", {})
    sub_researchers = state.get("sub_researchers", [])
    revision_count = state.get("revision_count", 0)

    print(f"write_full_report: generating complete report for topic='{topic[:50]}...' (revision {revision_count})")
    print(f"write_full_report: using min_credibility_score={agent_config.min_credibility_score}")

    sections = outline.get("sections", [])

    # Build comprehensive research context from all sections
    research_by_section = {}
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

        research_by_section[subtopic] = credible_sources[:25]  # Top 25 per section
        total_sources += len(credible_sources[:25])

    print(f"write_full_report: {total_sources} total high-quality sources available")

    # Build comprehensive sources string
    all_sources_text = ""
    source_index = 1
    source_mapping = {}  # Maps section → source index

    for section in sections:
        section_title = section.get("title", "")
        section_subtopics = section.get("subtopics", [])

        all_sources_text += f"\n\n=== SOURCES FOR SECTION: {section_title} ===\n"

        for subtopic in section_subtopics:
            if subtopic in research_by_section:
                for source, findings, credibility in research_by_section[subtopic]:
                    all_sources_text += f"\n[{source_index}] {source} (credibility: {credibility:.2f})\n{findings}\n"
                    all_sources.append(source)
                    source_index += 1

    # Generate report with structured output for title
    ReportWithTitle = create_model(
        'ReportWithTitle',
        report_title=(str, ...),
        content=(str, ...)  # Full markdown report with ## headers
    )

    report_prompt = f"""
You are an expert research report writer. Write a comprehensive, professional research report on:

**TOPIC**: {topic}

**OUTLINE** (follow this structure):
{json.dumps(sections, indent=2)}

**ALL RESEARCH SOURCES** (cite using [1], [2], etc.):
{all_sources_text}

**CRITICAL REQUIREMENTS:**

1. **ANTI-HALLUCINATION (MANDATORY)**:
   - ONLY use information EXPLICITLY in the sources above
   - EVERY fact, name, date, statistic MUST have a citation [1], [2], etc.
   - DO NOT use your knowledge to add information not in sources
   - If you can't find a source for a claim, LEAVE IT OUT

2. **QUANTITATIVE DATA (REQUIRED for market/economic topics)**:
   - Include ALL specific numbers: revenue, market size, percentages, dates
   - Example: "Market reached ¥18.5 trillion ($125B) in 2024 [3]"
   - Example: "Grew from 3.1M to 3.8M users (+22.6%) Jan-Nov 2025 [7]"

3. **COMPREHENSIVE COVERAGE**:
   - Address ALL sections in the outline
   - Use MULTIPLE sources per claim when available
   - Include specific examples, not just general statements
   - Present balanced viewpoints (include counterarguments if topic is debated)

4. **REPORT STRUCTURE**:
   - Generate a professional, refined title (in report_title field)
   - Use markdown headers (## for sections)
   - Write cohesive narrative that flows naturally
   - Each section builds on previous ones
   - Maintain consistent terminology throughout

5. **CITATIONS**:
   - Cite sources as [1], [2], etc. corresponding to source numbers above
   - Use at least 50% of available sources (aim for {total_sources // 2}+ citations)
   - More citations = better evidence quality

Return in 'content' field: Complete markdown report WITHOUT the title header or References section.
Start directly with "## Executive Summary" (or first section name).
I will add the title and References section separately.

Return in 'report_title' field: A clear, professional title for this report.
"""

    messages = [
        SystemMessage(content="You are an expert research report writer specializing in comprehensive, evidence-based reports. CRITICAL: Only use information from provided sources. Every claim must be cited."),
        HumanMessage(content=report_prompt)
    ]

    # Get quality model and generate report
    model_config = get_quality_model_config(agent_config)
    llm_with_structured = configurable_model.with_config(model_config).with_structured_output(ReportWithTitle)

    print(f"write_full_report: invoking LLM for complete report generation...")
    response = await llm_with_structured.ainvoke(messages)

    report_content = response.content
    report_title = response.report_title

    print(f"write_full_report: generated report with title='{report_title}'")

    # Assemble final report with title and references
    full_report = f"# {report_title}\n\n{report_content}\n\n"

    # Add references section
    unique_sources = list(dict.fromkeys(all_sources))
    full_report += "## References\n\n"
    for i, source in enumerate(unique_sources, 1):
        formatted_source = format_source_as_markdown_link(source)
        full_report += f"[{i}] {formatted_source}\n"

    # Extract citations from report
    citations_in_content = re.findall(r'\[(\d+)\]', report_content)
    unique_citations = sorted(set(int(c) for c in citations_in_content if c.isdigit()))
    citation_count = len(unique_citations)

    print(f"write_full_report: {citation_count} sources cited out of {len(unique_sources)} available")

    full_report_message = AIMessage(content=full_report)

    # Generate report_id if not exists
    report_id = state.get("report_id", "")
    if not report_id:
        report_id = f"{uuid.uuid4().hex[:12]}"
        print(f"write_full_report: generated new report_id={report_id}")

    new_version_id = state.get("version_id", 0) + 1

    # Build research_by_subtopic for MongoDB (using original structure)
    research_by_subtopic = {}
    for researcher in sub_researchers:
        subtopic = researcher.get("subtopic", "")
        research_by_subtopic[subtopic] = {
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
            search_results=research_by_subtopic,
            report_sections=[
                {
                    "title": s.get("title"),
                    "subtopics": s.get("subtopics", []),
                    "source_count": len(research_by_section.get(s.get("subtopics", [""])[0], []))
                }
                for s in sections
            ]
        )
        print(f"write_full_report: saved report {report_id} version {new_version_id} to MongoDB")
    except Exception as e:
        print(f"write_full_report: failed to save report to MongoDB: {e}")

    return {
        "report_id": report_id,
        "report_content": full_report,
        "report_references": unique_sources,
        "version_id": new_version_id,
        "report_history": state.get("report_history", []) + [new_version_id],
        "messages": [full_report_message]
    }


async def research_sections(state: dict, config: RunnableConfig) -> dict:
    """
    Research each section from the outline with dedicated subresearchers.
    Flow: outline sections → 1 subresearcher per section → research results
    """
    agent_config = get_config_from_configurable(config.get("configurable", {}))

    topic = state.get("topic", "")
    outline = state.get("report_outline", {})
    sections = outline.get("sections", [])

    print(f"research_sections: researching {len(sections)} sections")

    # Create 1 subresearcher per section
    async def process_section(idx: int, section: dict):
        """Process a single section through subresearcher"""
        section_title = section.get("title", "")
        section_subtopics = section.get("subtopics", [])

        # OPTIMIZATION: Extract pre-generated research plan from outline (if exists)
        pregenerated_plan = section.get("research_plan", {})
        research_plan = []
        if pregenerated_plan:
            primary_query = pregenerated_plan.get("primary_query", "")
            targeted_searches = pregenerated_plan.get("targeted_searches", [])

            if primary_query:
                research_plan.append({"query": primary_query, "type": "primary"})

            for ts in targeted_searches:
                research_plan.append({
                    "query": ts.get("query", ""),
                    "type": "targeted",
                    "priority": ts.get("priority", "medium")
                })

        print(f"\n  === Researching section {idx + 1}/{len(sections)}: {section_title} ===")
        print(f"      Subtopics: {section_subtopics[:3]}")
        if research_plan:
            print(f"      Pre-generated plan: {len(research_plan)} searches")

        subgraph_state = {
            "subtopic_id": idx,
            "subtopic": section_title,
            "main_topic": topic,
            "other_subtopics": [s.get("title", "") for s in sections],
            "section_subtopics": section_subtopics,
            "research_results": {},
            "research_depth": 1,
            "source_credibilities": {},
            "source_relevance_scores": {},
            "entities": [],
            "research_plan": research_plan,  # OPTIMIZATION: Pass pre-generated plan
            "completed_searches": 0,
            "max_search_results": agent_config.max_search_results,
            "max_research_depth": agent_config.max_research_depth,
            "search_api": agent_config.search_api,
        }

        result = await subresearcher_graph.ainvoke(subgraph_state)

        return {
            "subtopic_id": idx,
            "subtopic": section_title,  # Use section title as subtopic key
            "research_results": result.get("research_results", {}),
            "source_credibilities": result.get("source_credibilities", {}),
            "research_depth": result.get("research_depth", 1)
        }

    tasks = [process_section(idx, section) for idx, section in enumerate(sections)]
    sub_researchers = await asyncio.gather(*tasks)

    print(f"research_sections: completed {len(sub_researchers)} subresearchers")
    for researcher in sub_researchers:
        sources = len(researcher.get("research_results", {}))
        print(f"  - {researcher.get('subtopic', 'Unknown')}: {sources} sources")

    # Update outline sections with subtopic keys
    for idx, section in enumerate(sections):
        section["subtopics"] = [sub_researchers[idx].get("subtopic", "")]

    return {
        "sub_researchers": sub_researchers,
        "report_outline": outline,  # Return updated outline with subtopics
        "subtopics": [r.get("subtopic", "") for r in sub_researchers]
    }


async def write_outline(state: dict, config: RunnableConfig) -> dict:
    # Get agent configuration
    agent_config = get_config_from_configurable(config.get("configurable", {}))

    topic = state.get("topic", "")

    print(f"generate_outline: creating outline for topic='{topic}'")

    outline_prompt = f"""
        You are a report outline specialist. Create a structured outline that DIRECTLY ANSWERS the main research question.

        MAIN RESEARCH QUESTION: {topic}

        CRITICAL: First, determine what TYPE of question this is, then organize sections accordingly:

        **TYPE A - ENTITY-FOCUSED** (listing/comparing specific items):
        Examples: "Best mobile games 2024", "Top K-pop songs", "MMORPG recommendations", "Leading AI companies"

        **CRITICAL FOR TYPE A - USE THEMATIC GROUPING + RESEARCH QUESTIONS:**
        - DO NOT create one section per entity (this leads to incomplete coverage)
        - Instead, create 2-4 THEMATIC sections that each cover MULTIPLE entities
        - Example for "MMORPGs in 2025":
          ❌ BAD: "WoW", "FFXIV", "GW2" (individual sections = incomplete coverage)
          ✅ GOOD: "Established MMORPGs with Expansions", "Emerging MMORPGs", "Gameplay Innovation Trends"
        - Each thematic section should discuss 3-5+ entities with comparisons
        - This ensures comprehensive coverage and prevents cherry-picking

        **CRITICAL FOR TYPE A - SUBTOPICS MUST BE RESEARCH QUESTIONS, NOT SPECIFIC ENTITIES:**
        - DO NOT list specific entity names as subtopics (e.g., "World of Warcraft", "Final Fantasy XIV")
        - Instead, use BROAD RESEARCH QUESTIONS that will allow discovery of ALL relevant entities
        - Example for "Established MMORPGs with Expansions":
          ❌ BAD subtopics: ["World of Warcraft: Dragonflight Expansion", "Final Fantasy XIV: Dawntrail Expansion", "Guild Wars 2: The Icebrood Saga"]
          ✅ GOOD subtopics: ["Which established MMORPGs released major expansions in 2024-2025?", "What are the player retention trends for long-running MMORPGs?", "How do expansion features compare across top MMORPGs?", "What are the subscription and monetization models?"]
        - Research questions allow the research system to discover ALL relevant entities dynamically
        - Specific entity names constrain research to only those entities and miss others
        - Each section should have 3-5 research questions that guide comprehensive discovery

        **TYPE B - THEMATIC/ANALYTICAL** (explaining concepts, comparing philosophies, analyzing impacts):
        Examples: "Investment philosophies of Buffett vs Munger", "Impact of AI on labor", "HGT in eukaryotes"
        → Create sections by KEY THEMES/ASPECTS that answer the question
        → Section titles = Themes/dimensions (e.g., "Risk Management", "Portfolio Construction", "Decision Process")
        → Synthesize findings across subtopics for each theme

        **TYPE C - MARKET/TECHNICAL ANALYSIS** (sizing markets, evaluating technologies, comparing methods):
        Examples: "Elderly consumption Japan 2020-2050", "Scaling quantum computing", "SRAM stability methods"
        → Structure by ANALYTICAL COMPONENTS
        → Section titles = Analysis dimensions (e.g., "Market Size", "Technology Roadmap", "Implementation")

        **TYPE D - COMPREHENSIVE RESEARCH** (multi-faceted exploration of complex topics):
        Examples: "AI in K-12 education", "Gut microbiota and cancer", "Bird migration navigation"
        → Organize by LOGICAL FLOW (background → mechanisms → applications → implications)
        → Build from foundational concepts to specific findings

        STRUCTURE (adapt to type):
        1. Executive Summary - Key findings/direct answer
        2. Main sections (2-5):
           - TYPE A: One per discovered entity
           - TYPE B/C/D: One per theme/aspect/component
        3. Conclusion - Synthesis and implications

        For each section:
        - Title: Entity name OR theme/aspect
        - Subtopics: Which research contains relevant data
        - Key questions: What this section should answer

        Return JSON:
        {{
          "sections": [
            {{
              "title": "Section Title",
              "subtopics": ["subtopic1", "subtopic2"],
              "research_plan": {{
                "primary_query": "main search query for this section",
                "targeted_searches": [
                  {{"query": "specific search 1", "priority": "high"}},
                  {{"query": "specific search 2", "priority": "medium"}}
                ]
              }}
            }}
          ]
        }}

        OPTIMIZATION: For each section, also provide a research_plan:
        - primary_query: One broad search combining the topic + section title
        - targeted_searches: 3-5 focused searches for the subtopics (with priority: high/medium/low)
        - This saves an additional LLM call during research phase

        **CRITICAL QUERY SPECIFICITY RULES (for high-quality research):**
        When creating research queries, ALWAYS include:
        1. **Location specificity**: If the topic mentions a country/region, include it in EVERY query
           - ✅ GOOD: "Japan elderly consumption housing 2024 statistics"
           - ❌ BAD: "elderly housing consumption patterns"
        2. **Data type specificity**: Explicitly request quantitative data
           - ✅ GOOD: "Japan elderly population 2020-2050 projections statistics"
           - ❌ BAD: "Japan aging population trends"
        3. **Market size specificity**: For market analysis, request dollar/yen amounts
           - ✅ GOOD: "Japan elderly clothing market size revenue 2024 yen"
           - ❌ BAD: "elderly clothing market Japan"
        4. **Year/timeframe specificity**: Include exact years or ranges
           - ✅ GOOD: "Japan elderly consumption 2020 2025 2030 projections"
           - ❌ BAD: "Japan future elderly consumption"
        5. **Official source targeting**: Include keywords that find authoritative data
           - ✅ GOOD: "Japan Ministry Internal Affairs elderly expenditure survey 2024"
           - ✅ GOOD: "OECD Japan elderly spending data statistics"
           - ❌ BAD: "Japan elderly spending information"

        Examples of GOOD vs BAD research queries:
        Topic: "Market size analysis for elderly consumption in Japan 2020-2050"

        ❌ BAD queries (too generic, get irrelevant results):
        - "elderly consumption patterns"
        - "aging population market opportunities"
        - "senior consumer behavior"

        ✅ GOOD queries (specific, get targeted data):
        - "Japan elderly population 2020 2030 2050 projections Ministry statistics"
        - "Japan elderly housing expenditure market size 2024 yen revenue"
        - "Japan elderly food consumption spending statistics 2023 2024"
        - "Japan transportation elderly market size accessible vehicles 2024"
        - "Japan Ministry Internal Affairs elderly household spending survey 2024"
        """

    messages = [
        SystemMessage(
            content="You are an expert report outline specialist. Create comprehensive, well-structured outlines that ensure complete coverage, logical flow, and clear organization. Focus on creating outlines that will result in high-quality reports with excellent coverage, evidence, structure, and clarity."),
        HumanMessage(content=outline_prompt)
    ]

    # Get fast model config (outline generation is a simpler task)
    model_config = get_fast_model_config(agent_config)
    llm = configurable_model.with_config(model_config)

    response = await llm.ainvoke(messages)
    response_text = response.content if hasattr(response, 'content') else str(response)

    # Parse JSON outline (simplified)
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            outline = json.loads(json_match.group())
        else:
            # Create generic outline
            outline = {
                "sections": [
                    {"title": "Executive Summary", "subtopics": []},
                    {"title": "Main Analysis", "subtopics": []},
                    {"title": "Conclusion", "subtopics": []}
                ]
            }
    except Exception as e:
        print(f"generate_outline: error parsing JSON, using fallback: {e}")
        outline = {
            "sections": [
                {"title": "Executive Summary", "subtopics": []},
                {"title": "Main Analysis", "subtopics": []},
                {"title": "Conclusion", "subtopics": []}
            ]
        }

    print(f"generate_outline: created outline with {len(outline.get('sections', []))} sections")

    for section in outline.get("sections", []):
        print(f"  Section: '{section.get('title', '')}' -> Subtopics: {section.get('subtopics', [])}")

    return {
        "messages": [AIMessage(content=outline.get("sections", []))],
        "report_outline": outline,
        "report_sections": []
    }


async def research_and_write(state: dict, config: RunnableConfig) -> dict:
    """
    Research sections and write complete report in one LLM call.
    """
    print("research_and_write: Starting combined research and writing process")

    # Research Sections (1 subresearcher per outline section)
    print("research_and_write: Step 1 - Researching sections")
    research_result = await research_sections(state, config)
    state = {**state, **research_result}

    # Write Complete Report (single LLM call)
    print("research_and_write: Step 2 - Writing full report")
    write_result = await write_full_report(state, config)

    return {
        **research_result,
        **write_result,
    }