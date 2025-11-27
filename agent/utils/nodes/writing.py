from langgraph.types import RunnableConfig
from langchain_core.messages import SystemMessage, AIMessage, HumanMessage
from utils.model import llm, llm_quality
from utils.configuration import get_config_from_configurable, tavily_client, serper_client
from utils.subresearcher import subresearcher_graph
from utils.db import save_report
from .helpers import format_source_as_markdown_link
import re, json, asyncio, uuid



async def search_for_section_sources(section_title: str, topic: str, search_api: str = "tavily",
                                     num_results: int = 3) -> list[dict]:
    """
    Perform a web search to find additional sources for a section.
    Used when existing research doesn't provide enough sources.
    Respects the configured search API (tavily or serper).
    """
    search_query = f"{section_title}"
    print(f"    Searching for additional sources: '{search_query[:50]}...'")

    try:
        loop = asyncio.get_event_loop()
        if search_api == "serper":
            search_results = await loop.run_in_executor(
                None,
                lambda: serper_client.search(query=search_query, max_results=num_results)
            )
            # SerperClient returns Tavily-compatible format
            return search_results.get("results", [])
        else:
            search_results = await loop.run_in_executor(
                None,
                lambda: tavily_client.search(query=search_query, max_results=num_results)
            )
            return search_results.get("results", [])
    except Exception as e:
        print(f"    Search failed: {e}")
        return []


async def write_single_section(
        section: dict,
        topic: str,
        research_by_subtopic: dict,
        min_credibility_score: float,
        all_sections: list = None,
        section_index: int = 0,
        previously_written_sections: list = None,
        search_api: str = "serper"
) -> dict:
    """
    Write a single section of the report with citations.

    SEQUENTIAL WRITING: Receives previously written sections to maintain context.
    If insufficient sources exist, performs additional web search.
    """
    section_title = section.get("title", "")
    section_subtopics = section.get("subtopics", [])
    all_sections = all_sections or []
    previously_written_sections = previously_written_sections or []
    MIN_SOURCES_THRESHOLD = 3  # Increased threshold for better coverage

    print(f"  Writing section {section_index + 1}/{len(all_sections)}: {section_title}")

    # Gather relevant research for this section with ENHANCED source processing
    relevant_research = ""
    sources_list = []
    research_depth_info = {}
    source_type_counts = {"academic": 0, "news": 0, "general": 0}

    for subtopic in section_subtopics:
        if subtopic in research_by_subtopic:
            results = research_by_subtopic[subtopic]["results"]
            credibilities = research_by_subtopic[subtopic]["credibilities"]

            # Get research depth if available (from subresearcher state)
            research_depth = research_by_subtopic[subtopic].get("research_depth", 1)
            research_depth_info[subtopic] = research_depth

            # Sort sources by credibility score (highest first)
            credible_sources = [
                (source, findings, credibilities.get(source, 0.5))
                for source, findings in results.items()
                if credibilities.get(source, 0.5) >= min_credibility_score
            ]
            credible_sources.sort(key=lambda x: x[2], reverse=True)  # Sort by credibility descending

            # Categorize sources by type for better context
            def categorize_source(source_url: str) -> str:
                """Categorize source type based on URL"""
                academic_domains = ['edu', 'scholar.google', 'arxiv.org', 'researchgate.net', 'ieee.org',
                                    'springer.com', 'sciencedirect.com']
                news_domains = ['bbc.com', 'reuters.com', 'nytimes.com', 'theguardian.com', 'cnn.com', 'wsj.com',
                                'bloomberg.com']

                source_lower = source_url.lower()
                if any(domain in source_lower for domain in academic_domains):
                    return "academic"
                elif any(domain in source_lower for domain in news_domains):
                    return "news"
                else:
                    return "general"

            # Expand to top 25 sources for comprehensive coverage
            # Prioritize diverse source types for balanced perspective
            # With 45-56 sources available per section, using 25 ensures we capture
            # the full depth of multi-layer research while staying within token limits
            for source, findings, credibility in credible_sources[:25]:
                source_type = categorize_source(source)
                source_type_counts[source_type] += 1

                # Include source type in metadata for LLM context
                type_label = f"[{source_type.upper()}]" if source_type in ["academic", "news"] else ""
                relevant_research += f"\nSource {type_label}: {source} (credibility: {credibility:.2f})\n{findings}\n"
                sources_list.append(source)

    # If insufficient sources, search for more
    if len(sources_list) < MIN_SOURCES_THRESHOLD:
        print(f"    Section '{section_title}' has only {len(sources_list)} sources, searching for more...")
        # Use configured search API for fallback searches
        additional_results = await search_for_section_sources(section_title, topic, search_api=search_api)

        for result in additional_results:
            source_url = result.get("url", "")
            source_title = result.get("title", "Untitled")
            source_content = result.get("content", "")

            if source_content:
                source_key = f"{source_title} ({source_url})"
                relevant_research += f"\nSource: {source_key} (credibility: 0.70)\n{source_content[:500]}...\n"
                sources_list.append(source_key)

        print(f"    Now have {len(sources_list)} sources for section")

    # If still no research, note it
    if not relevant_research:
        relevant_research = "Limited sources available. Provide general analysis based on the topic."

    # Build sequential context - include actual content from previously written sections
    sequential_context = ""
    if previously_written_sections:
        sequential_context = "\n**PREVIOUSLY WRITTEN SECTIONS - Build on this content:**\n"
        sequential_context += f"You are writing Section {section_index + 1} of {len(all_sections)}. The reader has already read:\n\n"

        for i, prev_section in enumerate(previously_written_sections):
            prev_title = prev_section.get("title", "")
            prev_content = prev_section.get("content", "")

            # Include a summary of previous section content (first 800 chars to maintain context)
            content_preview = prev_content[:800] + "..." if len(prev_content) > 800 else prev_content

            sequential_context += f"--- Section {i + 1}: {prev_title} ---\n{content_preview}\n\n"

        sequential_context += """**YOUR SECTION MUST:**
- Reference and build upon specific facts, names, and details mentioned in previous sections
- Continue the narrative flow - don't start from scratch as if nothing was written before
- Add NEW information that extends or deepens the analysis (not repeat what was already said)
- Use phrases like "As mentioned in [previous section]...", "Building on the [X] discussed above...", "These [entities] face..."
- If previous sections introduced specific entities (games, companies, people, products), reference them by name in your section
- Focus on answering the MAIN RESEARCH QUESTION with specific, concrete details from your sources
"""

    # Build outline context for remaining sections
    outline_context = ""
    if all_sections and section_index < len(all_sections) - 1:
        sections_after = all_sections[section_index + 1:]
        outline_context = "\n**SECTIONS COMING AFTER YOURS (don't cover their content):**\n"
        for i, s in enumerate(sections_after):
            s_title = s.get("title", "")
            s_questions = s.get("key_questions", [])
            outline_context += f"  {section_index + 2 + i}. \"{s_title}\""
            if s_questions:
                outline_context += f" - will cover: {', '.join(s_questions[:2])}"
            outline_context += "\n"

    # IMPROVEMENT 4: Add research depth context for LLM awareness
    research_quality_note = ""
    if research_depth_info:
        max_depth = max(research_depth_info.values())
        if max_depth >= 2:
            research_quality_note = f"\n**RESEARCH QUALITY NOTE:**\nYou have access to DEEP, multi-layered research including entity-specific investigations from academic, news, and general sources. This research went beyond initial broad searches to focus on specific topics/entities.\n\n**CRITICAL:** You have {len(sources_list)} high-quality sources available. USE this detailed information to write comprehensive, specific content. The more sources you cite, the better the coverage.\n"
        else:
            research_quality_note = f"\n**RESEARCH QUALITY NOTE:**\nYou have access to {len(sources_list)} sources on this topic. Focus on comprehensive coverage using all available sources.\n"

    # IMPROVEMENT 5: Add source diversity context
    source_diversity_note = f"\n**SOURCE DIVERSITY:**\nYour sources include: {source_type_counts['academic']} academic, {source_type_counts['news']} news, {source_type_counts['general']} general sources. Use this diversity to provide a balanced, well-rounded perspective.\n"

    section_prompt = f"""
    Write the "{section_title}" section of a research report on: {topic}

    {sequential_context}
    {outline_context}
    {research_quality_note}
    {source_diversity_note}

    SOURCES (numbered for citation):
    {relevant_research}

    **CRITICAL ANTI-HALLUCINATION RULES - VIOLATIONS ARE UNACCEPTABLE:**
    1. ONLY write about information that EXPLICITLY appears in the SOURCES above - word for word verification
    2. Every factual claim (names, dates, numbers, titles, events, statistics) MUST have a citation [1], [2], etc.
    3. DO NOT combine information from different sources to create new facts
    4. DO NOT infer or assume connections between facts that aren't stated in sources
    5. DO NOT use your knowledge to add information not in the sources
    6. DO NOT mention specific entities, dates, rankings, or statistics unless they appear EXACTLY in the sources
    7. If a source mentions "Entity X" but doesn't mention it ranked #1, DO NOT say it was #1
    8. If sources mention different entities separately, DO NOT combine them into a single narrative unless sources explicitly do so
    9. If you're unsure whether something is in the sources, LEAVE IT OUT
    10. It's better to write less content that is accurate than more content that is fabricated

    **VERIFICATION PROCESS:**
    - Before writing each sentence, check: "Is this EXACTLY stated in the sources?"
    - For specific claims (entity names, dates, rankings, statistics): Quote or closely paraphrase the source
    - If you can't find a source for a claim, DO NOT include it

    **WHAT TO DO IF SOURCES ARE LIMITED:**
    - Focus only on what IS explicitly mentioned in the sources
    - Use phrases like "According to [source]..." or "Source [X] states..."
    - Acknowledge gaps: "Available research focuses on X, while Y remains less documented"
    - DO NOT fill gaps with invented information, even if you think you know the answer

    Write a high-quality, well-structured section that excels in:

    **COVERAGE:**
    - Comprehensively address all aspects of the section topic
    - Cover key questions and subtopics assigned to this section
    - Include relevant details, examples, and context from sources
    - Don't leave important aspects unaddressed
    
    **CONCRETE EXAMPLES (REQUIRED):**
    - DO NOT just describe concepts abstractly - provide SPECIFIC EXAMPLES from sources
    - For morphological similarities: Give actual word examples showing the pattern
      Example: ❌ "Both languages use agglutination"
      Example: ✅ "Turkish forms 'eve' (to the house) by adding '-e' to 'ev' (house), while Korean uses '-로' in '집으로' (to the house) from '집' (house) [8]"
    - For phonological features: Include specific sound examples or rules
    - For lexical borrowings: List actual shared words if mentioned in sources
    - Every major claim should have at least one concrete example from sources

    **COMPREHENSIVE COVERAGE - THIS IS CRITICAL FOR HIGH SCORES:**

    **ANTI-VAGUENESS RULES (STRICTLY ENFORCED):**
    ❌ FORBIDDEN PHRASES - These will result in LOW EVIDENCE SCORES:
    - "several", "various", "multiple", "many", "some", "a number of"
    - "emerging titles", "new games" (without naming them)
    - "industry trends show", "research indicates" (without specific numbers)
    - "significant growth", "considerable impact" (without percentages/data)
    - "players appreciate", "users prefer" (without survey data/numbers)

    ✅ REQUIRED SPECIFICITY:
    - Name ALL games, companies, people mentioned
    - Include ALL numbers: player counts, revenue, percentages, dates, rankings
    - Use EXACT quotes from sources for claims
    - Include specific features/mechanics, not general descriptions

    **Entity Coverage Requirements:**
    - List ALL specific entities mentioned in sources (games, companies, products, people, etc.)
    - If sources mention "top 5", "best 10", etc., list ALL items with names
    - Example: ❌ "Several MMORPGs like WoW remain popular"
              ✅ "Top MMORPGs in 2025 include World of Warcraft (5.2M active players), Final Fantasy XIV (3.8M players), Guild Wars 2 (2.1M players), Elder Scrolls Online (1.9M players), and Lost Ark (1.5M players) [1][2][3]"
    - Example: ❌ "New expansions introduce exciting features"
              ✅ "WoW's Midnight expansion (Q2 2025) adds housing system with 50+ customization options and revamped talent trees affecting all 13 classes [5]"

    **Numerical/Statistical Requirements (MANDATORY):**
    - EVERY claim about popularity, growth, or trends MUST include:
      * Exact numbers (not "many" or "significant")
      * Dates (not "recently" or "in 2025")
      * Percentages (not "increased" but "increased 23%")
      * Comparisons with specific values ("from 2.1M to 3.4M players")
    - Example: ❌ "FFXIV saw substantial growth in 2025"
              ✅ "FFXIV grew from 3.1M to 3.8M active players (+22.6%) between Jan-Nov 2025 [7]"
    - Example: ❌ "Mobile MMORPGs are gaining traction"
              ✅ "Mobile MMORPGs generated $12.3B revenue in 2025 (+18% YoY), led by Genshin Impact ($3.2B) and Tower of Fantasy ($1.8B) [9]"

    **Feature/Mechanic Specificity:**
    - DON'T say: "innovative gameplay mechanics"
    - DO say: "cross-server dungeons supporting 40-player raids, dynamic weather affecting combat stats, and player-owned guild halls with crafting stations [12]"
    - DON'T say: "improved graphics"
    - DO say: "upgraded to Unreal Engine 5.3 with ray tracing, DLSS 3.5 support, and 4K textures reducing load times by 35% [15]"

    **Comparison Requirements:**
    - ALWAYS include specific data points when comparing
    - Example: ❌ "WoW is more popular than GW2"
              ✅ "WoW maintains 5.2M active players vs GW2's 2.1M (2.5x larger playerbase), but GW2 has higher retention rate (78% vs 65% 6-month retention) [1][4]"

    **EVIDENCE:**
    - Every factual claim must have a citation [1], [2], etc.
    - Use multiple sources to support important claims when available
    - Integrate citations naturally into the text
    - Provide specific examples and data from sources
    
    **BALANCE AND COUNTERARGUMENTS (CRITICAL FOR HIGH SCORES):**
    - If sources present DEBATED or CONTROVERSIAL topics, you MUST present BOTH sides
    - Include counterarguments, criticisms, and alternative viewpoints from sources
    - Example: If discussing "Altaic hypothesis", include:
      * Arguments FOR the hypothesis (similarities, shared features)
      * Arguments AGAINST the hypothesis (lack of evidence, alternative explanations)
      * Specific data/challenges mentioned in sources
    - Use phrases like: "However, critics argue...", "Alternative theories suggest...", "Some researchers challenge this view..."
    - DO NOT present only one side of a debate - this reduces credibility and comprehensiveness
    - If sources mention skepticism or controversy, explicitly address it

    **STRUCTURE:**
    - Start with a clear topic sentence that introduces the section's focus
    - Organize content logically (chronological, thematic, or by importance)
    - Use smooth transitions between paragraphs
    - Build arguments progressively, leading to key insights
    - End paragraphs with sentences that connect to the next idea

    **CLARITY:**
    - Write in clear, professional academic style
    - Use precise language and avoid vague statements
    - Explain technical terms when first introduced
    - Write concise, well-crafted sentences
    - Ensure each paragraph has a single clear focus
    - Use active voice when appropriate for readability

    **LENGTH AND DEPTH:**
    - Provide sufficient detail to comprehensively cover the topic
    - Balance depth with readability
    - CRITICAL: You have access to {len(sources_list)} high-quality sources
    - Aim to cite at least 50% of available sources (minimum {max(len(sources_list) // 2, 5)} citations)
    - More citations = better evidence = higher quality score
    - Use the full depth of research available to you

    IMPORTANT: Do NOT include the section title. Start directly with the content.
    """

    # Special handling for Executive Summary and Conclusion sections
    is_executive_summary = "executive summary" in section_title.lower()
    is_conclusion = "conclusion" in section_title.lower() or "recommendations" in section_title.lower()

    if is_executive_summary:
        section_prompt += "\n\n**EXECUTIVE SUMMARY REQUIREMENTS:**\n- Provide a concise, high-level overview of the entire report\n- Highlight the most important findings and conclusions\n- Summarize key insights from all sections\n- Keep it brief but comprehensive (2-3 paragraphs)\n- Write for a busy executive audience - clear and impactful"
    elif is_conclusion:
        section_prompt += "\n\n**CONCLUSION REQUIREMENTS:**\n- Synthesize findings from all sections\n- Draw clear, evidence-based conclusions\n- Provide actionable recommendations when applicable\n- Connect back to the research objectives\n- End with forward-looking statements or implications"

    messages = [
        SystemMessage(
            content="You are an expert research report writer. CRITICAL: You MUST only write information that explicitly appears in the provided sources. DO NOT use your knowledge to add facts, names, dates, or statistics. Every factual claim must be directly traceable to a source. Hallucination is unacceptable. Additionally, focus on writing high-quality, well-structured content that excels in coverage, evidence, structure, and clarity."),
        HumanMessage(content=section_prompt)
    ]


    response = await llm_quality.ainvoke(messages)
    section_content = response.content if hasattr(response, 'content') else str(response)

    citations_in_content = re.findall(r'\[(\d+)\]', section_content)
    unique_citation_numbers = sorted(set(int(c) for c in citations_in_content))

    # Map citation numbers to actual sources (1-indexed)
    actually_cited_sources = []
    for citation_num in unique_citation_numbers:
        if 1 <= citation_num <= len(sources_list):
            actually_cited_sources.append(sources_list[citation_num - 1])

    print(f"      Citations: {len(sources_list)} sources available, {len(actually_cited_sources)} actually cited")

    return {
        "title": section_title,
        "content": section_content,
        "sources": actually_cited_sources,  # Only sources actually used
        "all_available_sources": sources_list,
        "subtopics": section_subtopics
    }


async def write_sections_with_citations(state: dict, config: RunnableConfig) -> dict:
    """
    Write each section of the report with proper inline citations in parallel.
    Uses research results to build evidence-based sections.
    """
    agent_config = get_config_from_configurable(config.get("configurable", {}))

    topic = state.get("topic", "")
    outline = state.get("report_outline", {})
    sub_researchers = state.get("sub_researchers", [])
    revision_count = state.get("revision_count", 0)

    print(f"write_sections_with_citations: writing sections for topic='{topic[:50]}...' (revision {revision_count})")
    print(f"write_sections_with_citations: using min_credibility_score={agent_config.min_credibility_score}")

    sections = outline.get("sections", [])

    # Build research lookup by subtopic
    research_by_subtopic = {}
    for researcher in sub_researchers:
        subtopic = researcher.get("subtopic", "")
        research_by_subtopic[subtopic] = {
            "results": researcher.get("research_results", {}),
            "credibilities": researcher.get("source_credibilities", {})
        }

    print(f"write_sections_with_citations: Available research keys:")
    for key in research_by_subtopic.keys(): # Key -> Subtopic
        results = research_by_subtopic[key]["results"]
        credibilities = research_by_subtopic[key]["credibilities"]
        num_sources = len(results)

        # Count sources that meet credibility threshold
        credible_count = sum(1 for source in results.keys()
                             if credibilities.get(source, 0.5) >= agent_config.min_credibility_score)

        print(
            f"  '{key}': {num_sources} sources ({credible_count} meet credibility threshold >= {agent_config.min_credibility_score})")

    # Write sections one at a time so each has context from previous sections
    print(f"write_sections_with_citations: writing {len(sections)} sections SEQUENTIALLY for context flow")
    written_sections = []

    for idx, section in enumerate(sections):
        # Pass all previously written sections so this section can reference them
        section_result = await write_single_section(
            section=section,
            topic=topic,
            research_by_subtopic=research_by_subtopic,
            min_credibility_score=agent_config.min_credibility_score,
            all_sections=sections,
            section_index=idx,
            previously_written_sections=written_sections,
            search_api=agent_config.search_api
        )
        written_sections.append(section_result)
        print(f"  Completed section {idx + 1}/{len(sections)}: {section_result.get('title', 'Unknown')}")

    print(f"write_sections_with_citations: completed all {len(written_sections)} sections sequentially")

    # Combine sections into full report
    full_report = f"# {topic}\n\n"
    all_sources = []

    for section in written_sections:
        full_report += f"## {section['title']}\n\n{section['content']}\n\n"
        all_sources.extend(section['sources'])

    # Add references section with properly formatted markdown links
    unique_sources = list(dict.fromkeys(all_sources))  # Remove duplicates, preserve order
    full_report += "## References\n\n"
    for i, source in enumerate(unique_sources, 1):
        formatted_source = format_source_as_markdown_link(source)
        full_report += f"[{i}] {formatted_source}\n"


    full_report_message = AIMessage(content=full_report)

    # Generate report_id if not exists
    report_id = state.get("report_id", "")
    if not report_id:
        report_id = f"{uuid.uuid4().hex[:12]}"
        print(f"write_sections_with_citations: generated new report_id={report_id}")

    new_version_id = state.get("version_id", 0) + 1

    # Save report to MongoDB
    try:
        await save_report(report_id, new_version_id, full_report)
        print(f"write_sections_with_citations: saved report {report_id} version {new_version_id} to MongoDB")
    except Exception as e:
        print(f"write_sections_with_citations: failed to save report to MongoDB: {e}")

    return {
        "report_id": report_id,
        "report_sections": written_sections,
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

        print(f"\n  === Researching section {idx + 1}/{len(sections)}: {section_title} ===")
        print(f"      Subtopics: {section_subtopics[:3]}")

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
            "research_plan": [],
            "completed_searches": 0,
            "max_search_results": agent_config.max_search_results,
            "max_research_depth": agent_config.max_research_depth,
            "search_api": agent_config.search_api,
            "enable_mcp_fetch": agent_config.enable_mcp_fetch,
            "max_mcp_fetches": agent_config.max_mcp_fetches
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
            }}
          ]
        }}
        """

    messages = [
        SystemMessage(
            content="You are an expert report outline specialist. Create comprehensive, well-structured outlines that ensure complete coverage, logical flow, and clear organization. Focus on creating outlines that will result in high-quality reports with excellent coverage, evidence, structure, and clarity."),
        HumanMessage(content=outline_prompt)
    ]

    response = await llm.ainvoke(messages)
    response_text = response.content if hasattr(response, 'content') else str(response)

    # Parse JSON outline (simplified)
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            outline = json.loads(json_match.group())
        else:
            # Fallback: create generic outline
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
    Research sections, and write.
    """
    print("research_and_write: Starting combined research and writing process")

    # Research Sections (1 subresearcher per outline section)
    print("research_outline_and_write: Step 2 - Researching sections")
    research_result = await research_sections(state, config)
    state = {**state, **research_result}

    # Write Sections
    print("research_outline_and_write: Step 3 - Writing sections")
    write_result = await write_sections_with_citations(state, config)


    return {
        **research_result,
        **write_result,
    }
