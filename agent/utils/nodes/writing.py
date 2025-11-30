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
            "research_depth": result.get("research_depth", 1),
            "summarized_findings": result.get("summarized_findings", "")
        }

    # Execute all subresearchers in parallel
    tasks = [process_subtopic(idx, subtopic) for idx, subtopic in enumerate(subtopics)]
    sub_researchers = await asyncio.gather(*tasks)

    print(f"\n{'='*60}")
    print(f"WRITER: Completed research for {len(sub_researchers)} subtopics")
    print(f"{'='*60}")

    total_sources = 0
    for researcher in sub_researchers:
        sources = len(researcher.get("research_results", {}))
        total_sources += sources
        print(f"  ✓ {researcher.get('subtopic', 'Unknown')}: {sources} sources")

    print(f"\nTotal sources gathered: {total_sources}\n")

    # Update subtopics with researched subtopic names for write_full_report
    for idx, subtopic_dict in enumerate(subtopics):
        subtopic_dict["subtopics"] = [sub_researchers[idx].get("subtopic", "")]

    # Build final outline structure
    outline = {
        "subtopics": subtopics  # List of {subtopic: str, queries: list, subtopics: list}
    }

    return {
        "messages": [AIMessage(content=f"Finished research on {len(subtopics)} subtopics and gathered {total_sources} sources. Generating report now...")],
        "research_outline": outline,
        "sub_researchers": sub_researchers,
    }



async def write_full_report(state: dict, config: RunnableConfig) -> dict:
    """
    Write the entire report in ONE LLM call for maximum coherence and efficiency.
    Uses all research results and outline to generate a comprehensive, well-structured report.
    """

    topic = state.get("topic", "")
    outline = state.get("report_outline", {})
    subtopics = outline.get("subtopics", [])  # List of {subtopic: str, queries: list, subtopics: list}
    sub_researchers = state.get("sub_researchers", [])
    revision_count = state.get("revision_count", 0)

    print(f"write_full_report: generating complete report for topic='{topic[:50]}...' (revision {revision_count})")
    print(f"write_full_report: using summarized findings from subresearchers")

    all_summarized_findings = ""
    total_reports = 0

    for researcher in sub_researchers:
        subtopic = researcher.get("subtopic", "")
        summarized_findings = researcher.get("summarized_findings", "")

        if not summarized_findings:
            print(f"write_full_report: WARNING - No summarized findings for '{subtopic}'!")
            continue

        all_summarized_findings += f"\n\n{'='*60}\n"
        all_summarized_findings += f"RESEARCH FOR: {subtopic}\n"
        all_summarized_findings += f"{'='*60}\n\n"
        all_summarized_findings += summarized_findings
        total_reports += 1
        print(f"write_full_report: Including summarized findings for '{subtopic}' ({len(summarized_findings)} chars)")

    print(f"write_full_report: Compiled {total_reports} summarized research reports")
    print(f"write_full_report: Total context length: {len(all_summarized_findings)} characters")

    # Generate report with structured output for title and sources
    report_with_title_model = create_model(
        'ReportWithTitle',
        report_title=(str, ...),
        content=(str, ...),  # Full markdown report with ## headers
    )

    report_prompt = f"""
Based on all the research conducted, create a comprehensive, cohesive, professional research report that answers the main topic.
Provide a balanced, thorough analysis. Be as comprehensive as possible, and include all information that is relevant to the overall research question. People are using you for deep research and will expect detailed, comprehensive answers.

**CRITICAL: LANGUAGE REQUIREMENT**
The main topic below is written in a specific language. You MUST write your ENTIRE report in the SAME language as the main topic.
If the topic is in Chinese (中文), write the entire report in Chinese.
If the topic is in English, write the entire report in English.
Match the topic's language EXACTLY. This is MANDATORY.

**MAIN TOPIC**: {topic}

**RESEARCH RESULTS** (each has citations and sources):
{all_summarized_findings}

**CRITICAL REQUIREMENTS:**

1. **SOURCE FIDELITY & CITATION DISCIPLINE***:
   - ONLY use information EXPLICITLY in the sources above - NO general knowledge
   - EVERY factual sentence MUST end with citations [1], [2], etc.
   - Assign each source/url a single citation number in your tex
   - End with ### Sources that lists each source with corresponding numbers
   - IMPORTANT: Number sources sequentially without gaps (1,2,3,4...) in the final list regardless of which sources you choose
   - Example format:
      [1] Source Title: URL
      [2] Source Title: URL

2. **COMPREHENSIVE EXPANSION (NOT SUMMARY)**:
   - This is a FULL RESEARCH REPORT - EXPAND the subtopic reports, don't condense
   - Feel free to combine research for different subtopics
   - Each major section (##): 600-1000 words MINIMUM (strictly enforce)
   - Each subsection (###): 200-300 words
   - Final report should be LONGER than combined inputs
   - Structure: 5-7 major sections (##), each with 3-5 subsections (###)

3. **DETAILED, SPECIFIC CONTENT**:
   - Include ALL specifics: named entities, exact numbers, dates, percentages, methodologies
   - Use 10 sentences per paragraph MINIMUM
   - Never use vague terms ("increased significantly") - use exact data with context
   - Example: Instead of "AI improves communication [1]" → "Research by Smith et al. (2023) found AI-powered translation tools increased cross-cultural communication efficiency by 47% among multinational teams, with technical documentation translation reaching 89% accuracy and real-time video interpretation 76% accuracy [1]"

4. **QUALITY STANDARDS**:
   - Write in same language as topic using clear, academic style
   - Use topic sentences and transition paragraphs to build progressive arguments
   - Before submitting: verify 600+ words per section, 1 citation per sentence average, zero unsupported claims

Return in 'content' field: Complete markdown report WITHOUT title or References section. Start with first ## section.
Return in 'report_title' field: Clear, professional title.

CRITICAL: Only cite sources you actually used in the text.
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

    print(f"write_full_report: generated report with title='{report_title}' ({len(report_content)} chars)")

    full_report = f"# {report_title}\n\n{report_content}"

    full_report_message = AIMessage(content=full_report)

    # Generate report_id if not exists
    report_id = state.get("report_id", "")
    if not report_id:
        report_id = f"{uuid.uuid4().hex[:12]}"
        print(f"write_full_report: generated new report_id={report_id}")

    new_version_id = state.get("version_id", 0) + 1

    # Save report to MongoDB
    try:
        await save_report(
            report_id=report_id,
            version_id=new_version_id,
            full_report=full_report,
            report_title=report_title,
            search_results=all_summarized_findings,
            report_sections=[
                {
                    "title": subtopic_item.get("subtopic", ""),
                    "subtopics": subtopic_item.get("subtopics", []),
                    "source_count": len(sub_researchers[idx].get("research_results", {}))
                }
                for idx, subtopic_item in enumerate(subtopics)
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

