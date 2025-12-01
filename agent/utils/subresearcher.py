from typing import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from pydantic import create_model
from utils.model import llm_quality
from utils.configuration import tavily_client, exa_client, global_tavily_semaphore, global_exa_semaphore
from utils.nodes.helpers import filter_quality_sources
import asyncio
from urllib.parse import urlparse





class SubResearcherGraphState(TypedDict):

    subtopic_id: int
    subtopic: str  # The broad query
    main_topic: str
    other_subtopics: list[str]

    section_subtopics: list[str]  # Specific subtopics from research plan

    # Final outputs
    research_results: dict[str, str]  # All combined results
    source_credibilities: dict[str, float]
    source_relevance_scores: dict[str, float]  # Track search API relevance scores
    research_depth: int
    summarized_findings: str  # Synthesized summary of key findings with source attribution

    # Config
    max_search_results: int
    max_research_depth: int
    search_api: str

    # Internal state for iterative deepening
    entities: list[str]
    research_plan: list[dict]
    discovered_gaps: list[str]
    discovered_entities: list[str]
    coverage_score: float
    needs_deepening: bool

    # Shared entities and findings from other sections
    shared_research_pool: dict


def quick_domain_quality_score(url: str) -> float:
    """
    Quick, cheap domain-based quality assessment without LLM.
    Returns a score from 0.0 (low quality) to 1.0 (high quality).
    """
    try:
        parsed = urlparse(url)
        domain = parsed.netloc.lower()

        # High-quality domains (academic, government, established news)
        high_quality_patterns = [
            '.edu', '.gov', '.org',
            'scholar.google', 'arxiv.org', 'pubmed', 'nature.com', 'science.org',
            'ieee.org', 'acm.org', 'springer.com', 'sciencedirect.com',
            'nytimes.com', 'wsj.com', 'bloomberg.com', 'reuters.com',
            'bbc.com', 'theguardian.com', 'economist.com', 'forbes.com',
            'wikipedia.org', 'britannica.com'
        ]

        # Medium-quality domains (reputable general sources)
        medium_quality_patterns = [
            'medium.com', 'substack.com', 'news.ycombinator.com',
            'techcrunch.com', 'wired.com', 'arstechnica.com',
            'stackoverflow.com', 'github.com', 'reddit.com'
        ]

        # Low-quality patterns (spam, ads, low-trust)
        low_quality_patterns = [
            'pinterest.com', 'quora.com', 'yahoo.answers',
            'clickbait', 'viral', 'buzz', 'gossip',
            'ads', 'promo', 'sale', 'buy'
        ]

        for pattern in high_quality_patterns:
            if pattern in domain:
                return 0.9

        for pattern in medium_quality_patterns:
            if pattern in domain:
                return 0.6

        for pattern in low_quality_patterns:
            if pattern in domain or pattern in url.lower():
                return 0.2

        # Default to neutral quality
        return 0.5

    except Exception:
        return 0.5  # Default on parsing error


def filter_results_by_domain_quality(results: list[dict], min_score: float = 0.3) -> list[dict]:
    """
    Filter search results by domain quality before deep processing.
    Removes low-quality domains early to save API quota.
    """
    filtered = []
    for result in results:
        url = result.get("url", "")
        if not url:
            continue

        quality = quick_domain_quality_score(url)
        if quality >= min_score:
            filtered.append(result)

    return filtered


async def parallel_search_with_rate_limit(
    queries: list[tuple[str, dict]],
    search_api: str,
    max_results: int,
    max_concurrent: int = 2
) -> list[tuple[str, list[dict]]]:
    """
    Execute multiple search queries in parallel with rate limiting.
    """
    # Exa has stricter rate limits (5 req/sec) -> Reduce concurrency further
    if search_api == "exa":
        max_concurrent = 1  # Only 1 concurrent request for Exa

    semaphore = asyncio.Semaphore(max_concurrent)

    async def throttled_search(query: str, metadata: dict) -> tuple[str, list[dict], dict]:
        """Execute a single search with GLOBAL + LOCAL rate limiting and retry logic"""
        # Truncate query to 400 characters (API limit)
        original_query = query
        if len(query) > 400:
            # Try to cut at last space before 400 chars
            truncated = query[:400]
            last_space = truncated.rfind(' ')
            if last_space > 320:  # If space is reasonably close to end (80% of 400)
                query = truncated[:last_space].strip()
            else:
                query = truncated.strip()
            print(f"    Query truncated from {len(original_query)} to {len(query)} chars")

        # Select global semaphore based on API
        global_sem = global_exa_semaphore if search_api == "exa" else global_tavily_semaphore

        # Use both global (cross-subresearcher) and local (per-subresearcher) semaphores
        async with global_sem:  # Global limit
            async with semaphore:  # Then local
                await asyncio.sleep(0.5 if search_api == "exa" else 0.4)

                max_retries = 3
                retry_delay = 2.0

                for attempt in range(max_retries):
                    try:
                        loop = asyncio.get_event_loop()

                        if search_api == "tavily":
                            search_results = await loop.run_in_executor(
                                None,
                                lambda: tavily_client.search(query=query, max_results=max_results, include_raw_content=True)
                            )
                            results_list = search_results.get("results", [])[:max_results]

                        elif search_api == "exa":
                            search_results = await loop.run_in_executor(
                                None,
                                lambda: exa_client.search_and_contents(query, text=True, type="auto", num_results=max_results)
                            )
                            # Exa returns SearchResponse object with .results attribute
                            results_list = [
                                {
                                    "url": r.url,
                                    "title": r.title,
                                    "content": r.text,
                                    "score": r.score if hasattr(r, 'score') else 0.0
                                }
                                for r in search_results.results
                            ][:max_results]

                        return (query, results_list, metadata)

                    except Exception as e:
                        error_str = str(e).lower()
                        # Check if it's a rate limit error
                        if "rate" in error_str or "excessive" in error_str or "blocked" in error_str:
                            if attempt < max_retries - 1:
                                wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                                print(f"    Rate limit hit for '{query[:60]}...', retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})")
                                await asyncio.sleep(wait_time)
                                continue
                            else:
                                print(f"    Search failed for '{query[:60]}...': {e} (max retries exceeded)")
                                return (query, [], metadata)
                        else:
                            # Non-rate-limit error
                            print(f"    Search failed for '{query[:60]}...': {e}")
                            return (query, [], metadata)

                # If all retries exhausted
                return (query, [], metadata)

    # Execute all searches in parallel with rate limiting
    tasks = [throttled_search(query, meta) for query, meta in queries]
    results = await asyncio.gather(*tasks)

    return results


async def execute_research(state: SubResearcherGraphState) -> dict:
    """
    Execute research using pre-generated plan from outline creation.
    The research plan is pre-generated during outline creation, saving 1 LLM call per section.
    """
    research_plan = state.get("research_plan", [])
    search_api = state.get("search_api", "tavily")
    max_results = state.get("max_search_results", 3)
    shared_pool = state.get("shared_research_pool", {})
    section_title = state.get("subtopic", "")

    if not research_plan:
        print(f"[RESEARCH] ERROR: No research plan provided for section '{section_title}'")
        return {
            "research_results": {},
            "source_relevance_scores": {},
            "research_depth": 1
        }

    print(f"\n[RESEARCH] Executing {len(research_plan)} searches in parallel for: {section_title}")

    queries_with_meta = []
    cached_searches = shared_pool.get("search_cache", {})
    cache_hits = [] # Cache for entity-based search

    for idx, search_spec in enumerate(research_plan):
        query = search_spec.get("query", "")
        priority = search_spec.get("priority", "medium")

        # Check cache
        if query in cached_searches:
            cache_hits.append((query, cached_searches[query], {"priority": priority, "index": idx}))
        else:
            queries_with_meta.append((query, {"priority": priority, "index": idx}))

    if cache_hits:
        print(f"[RESEARCH] Cache hits: {len(cache_hits)} searches (saved API calls)")

    # Execute uncached searches in parallel
    search_results = []
    if queries_with_meta:
        search_results = await parallel_search_with_rate_limit(
            queries=queries_with_meta,
            search_api=search_api,
            max_results=max_results,
            max_concurrent=2  # Rate limiting - reduced to avoid API limits
        )

    # Combine cache hits with new results
    all_search_results = cache_hits + search_results

    # Process all results with quality filtering
    all_results = {}
    relevance_scores = {}
    total_before_filter = 0
    total_after_filter = 0
    updated_cache = {**cached_searches}

    for query, results_list, metadata in all_search_results:
        priority = metadata.get("priority", "medium")
        query_index = metadata.get("index", 0)

        total_before_filter += len(results_list)

        # Quality filtering (use uniform threshold)
        min_quality = 0.4
        results_list = filter_results_by_domain_quality(results_list, min_score=min_quality)
        total_after_filter += len(results_list)

        print(f"  [{priority.upper()}] {query[:60]}... → {len(results_list)} sources")

        # Cache new searches
        if query not in cached_searches and results_list:
            updated_cache[query] = results_list

        # Store results
        for result_idx, result in enumerate(results_list):
            source_url = result.get("url", "")
            source_title = result.get("title", "Untitled")
            source_content = result.get("content", "")
            search_score = result.get("score", 0.0)

            if source_content:
                source_key = f"{source_title} ({source_url})" if source_url else source_title
                all_results[source_key] = source_content

                # Relevance scoring based on query position and priority
                # Earlier queries in plan = more important
                if priority == "high":
                    base_relevance = 0.9
                elif priority == "medium":
                    base_relevance = 0.7
                else:
                    base_relevance = 0.5

                # Adjust by position in plan
                position_penalty = query_index * 0.02  # Small penalty for later queries
                base_relevance = max(0.3, base_relevance - position_penalty)

                # Use search API score if available, otherwise position-based
                if search_score is not None and search_score > 0:
                    relevance_scores[source_key] = search_score
                else:
                    relevance_scores[source_key] = base_relevance - (result_idx * 0.05)

    print(f"[RESEARCH] Total: {len(all_results)} unique sources")
    print(f"[RESEARCH] Quality filtering: {total_before_filter} → {total_after_filter} sources")

    # Update shared pool with new cache entries
    update_dict = {
        "research_results": all_results,
        "source_relevance_scores": relevance_scores,
        "research_depth": 1
    }

    if len(updated_cache) > len(cached_searches):
        updated_pool = {**shared_pool, "search_cache": updated_cache}
        update_dict["shared_research_pool"] = updated_pool
        print(f"[RESEARCH] Updated cache: {len(updated_cache) - len(cached_searches)} new searches cached")

    return update_dict


async def assess_coverage_and_gaps(state: SubResearcherGraphState) -> dict:
    """
    Analyze research coverage and discover gaps/entities.

    After initial research, this node:
    1. Extracts entities/topics mentioned in results
    2. Identifies knowledge gaps (questions raised but not answered)
    3. Estimates coverage score
    4. Decides if deeper research is needed
    """
    research_results = state.get("research_results", {})
    section_subtopics = state.get("section_subtopics", [])
    research_depth = state.get("research_depth", 1)
    max_depth = state.get("max_research_depth", 3)
    main_topic = state.get("main_topic", "")
    section_title = state.get("subtopic", "")

    if not research_results or research_depth >= max_depth:
        print(f"[COVERAGE ASSESSMENT] Skipping (depth={research_depth}/{max_depth}, results={len(research_results)})")
        return {
            "coverage_score": 1.0,
            "needs_deepening": False,
            "discovered_gaps": [],
            "discovered_entities": []
        }

    print(f"[COVERAGE ASSESSMENT] Analyzing {len(research_results)} sources for gaps and entities...")

    # Combine all research content for analysis
    # Sort by relevance/credibility if available, otherwise use all sources
    all_sources = list(research_results.items())

    num_sources_to_analyze = min(25, len(all_sources))
    combined_content = "\n\n".join([
        f"Source: {k}\n{v[:1500]}" 
        for k, v in all_sources[:num_sources_to_analyze]
    ])

    CoverageAnalysis = create_model(
        'CoverageAnalysis',
        coverage_score=(float, ...),  # 0.0-1.0
        discovered_entities=(list[str], ...),  # Important entities/topics mentioned
        knowledge_gaps=(list[str], ...),  # Questions raised but not answered
        needs_deeper_research=(bool, ...),
        reasoning=(str, ...)
    )

    # Use configurable_model with default settings (subresearcher doesn't have access to agent config)
    llm_structured = llm_quality.with_structured_output(CoverageAnalysis)

    analysis_prompt = f"""Analyze research coverage and identify gaps for deeper investigation.

TOPIC: {main_topic}
SECTION: {section_title}
EXPECTED SUBTOPICS: {', '.join(section_subtopics[:5])}

RESEARCH SO FAR:
{combined_content[:8000]}

ANALYZE:
1. **Coverage Score** (0.0-1.0): How well does this research cover the expected subtopics?
   - 0.85-1.0 = Excellent: All subtopics comprehensively covered with multiple specific examples, data, statistics, and detailed explanations
   - 0.75-0.84 = Good: Most subtopics well-covered with specific examples and some data points, minor gaps acceptable
   - 0.60-0.74 = Moderate: Subtopics addressed but lacking specific examples, data, or depth; needs targeted improvement
   - 0.40-0.59 = Weak: Major gaps exist, missing key details, examples, or entire subtopics
   - 0.0-0.39 = Poor: Minimal coverage, mostly off-topic or very superficial
   - BE FAIR BUT THOROUGH: Don't penalize for missing obscure details, but do require specific examples and data for main subtopics

2. **Discovered Entities**: Extract 2-4 SPECIFIC, NAMED entities that warrant deeper investigation:
   - ONLY include entities that are PROPER NOUNS or SPECIFIC NAMED THINGS:
     ✅ Organizations: "International Weightlifting Federation (IWF)", "USA Powerlifting (USAPL)", "IPF"
     ✅ Named techniques: "Westside Barbell conjugate method", "Smolov squat program", "Bulgarian training system"
     ✅ Specific studies: "Haff et al. 2003 periodization study", "Stone biomechanics research 2015"
     ✅ Named individuals: "Dr. Mike Israetel", "Boris Sheiko", "Greg Everett"
     ✅ Specific competitions: "Arnold Classic", "World Weightlifting Championships", "IPF Worlds"
   - NEVER include generic concepts or abstract ideas:
     ❌ "snatch technique" (too generic - this is already covered)
     ❌ "injury rates" (concept, not entity)
     ❌ "biomechanics" (too vague)
     ❌ "community culture" (abstract idea)
   - ONLY include entities if they would significantly improve the report with specific information
   - Test: Can you Google this exact phrase and find a Wikipedia page or official website? If YES → include. If NO → exclude.
   - PREFER FEWER HIGH-VALUE entities over many low-value ones

3. **Knowledge Gaps**: Identify 1-3 HIGHLY SPECIFIC, ACTIONABLE search queries that are < 400 characters:
   - ONLY include gaps that address MISSING CRITICAL INFORMATION for the expected subtopics
   - Each gap must be:
     ✅ Specific enough to find targeted data (include year, location, organization names when relevant)
     ✅ Different from existing research queries (check what was already searched)
     ✅ Actually answerable via web search (not opinion-based or requiring expert interviews)
   - Examples of GOOD gaps:
     ✅ "USA Powerlifting USAPL membership statistics 2023 2024"
     ✅ "Olympic weightlifting vs powerlifting injury rate comparison study"
     ✅ "IPF World Championships prize money athlete earnings 2024"
   - Examples of BAD gaps:
     ❌ "powerlifting community culture" (too vague, already covered)
     ❌ "how athletes feel about training" (opinion-based, not fact-based)
     ❌ "general biomechanics research" (not specific enough)
   - Format as search queries optimized for finding specific data, NOT as questions

4. **Needs Deeper Research**: Should we do another research round?
   - TRUE if: coverage < 0.60 (weak/poor) OR critical gaps exist (missing core subtopic data/statistics)
   - MAYBE (treat as TRUE) if: coverage 0.60-0.74 (moderate) AND high-value entities/gaps identified
   - FALSE if: coverage >= 0.75 (good) OR no actionable gaps/entities remain
   - BE BALANCED: Moderate coverage (0.60-0.74) with 1-2 targeted improvements is often sufficient

5. **Reasoning**: Brief explanation (1-2 sentences)

Return structured output.
"""

    try:
        messages = [
            SystemMessage(content="You are a research coverage analyst. Identify gaps and entities for deeper investigation."),
            HumanMessage(content=analysis_prompt)
        ]

        response = await llm_structured.ainvoke(messages)

        # Limit entities to top 5
        discovered_entities = response.discovered_entities[:5]
        knowledge_gaps = response.knowledge_gaps[:4]

        print(f"[COVERAGE ASSESSMENT] Results:")
        print(f"  Coverage score: {response.coverage_score:.2f}")
        print(f"  Needs deepening: {response.needs_deeper_research}")
        if discovered_entities:
            print(f"  Discovered entities ({len(discovered_entities)}): {', '.join(discovered_entities)}")
        if knowledge_gaps:
            print(f"  Knowledge gaps ({len(knowledge_gaps)}):")
            for gap in knowledge_gaps:
                print(f"    - {gap[:80]}")
        print(f"  Reasoning: {response.reasoning}")

        return {
            "coverage_score": response.coverage_score,
            "needs_deepening": response.needs_deeper_research and research_depth < max_depth,
            "discovered_gaps": knowledge_gaps,
            "discovered_entities": discovered_entities
        }

    except Exception as e:
        print(f"[COVERAGE ASSESSMENT] Analysis failed: {e}")
        # Conservative fallback: assume decent coverage, don't deepen
        return {
            "coverage_score": 0.7,
            "needs_deepening": False,
            "discovered_gaps": [],
            "discovered_entities": []
        }


async def deep_dive_research(state: SubResearcherGraphState) -> dict:
    """
    This runs AFTER assess_coverage_and_gaps and performs searches for:
    1. Discovered entities (specific items that need investigation)
    2. Knowledge gaps (unanswered questions)

    CROSS-SECTION LEARNING: Checks shared_research_pool first to avoid redundant research.
    """
    discovered_entities = state.get("discovered_entities", [])
    knowledge_gaps = state.get("discovered_gaps", [])
    search_api = state.get("search_api", "tavily")
    max_results = state.get("max_search_results", 3)
    shared_pool = state.get("shared_research_pool", {})
    research_plan = state.get("research_plan", [])

    if not discovered_entities and not knowledge_gaps:
        print(f"[DEEP DIVE] No gaps or entities to research")
        return {}

    # Check shared pool for entities already researched by other sections
    already_researched = shared_pool.get("researched_entities", {})
    new_entities = []
    reused_entities = []

    for entity in discovered_entities:
        if entity in already_researched:
            reused_entities.append(entity)
        else:
            new_entities.append(entity)

    if reused_entities:
        print(f"[DEEP DIVE - SHARED POOL] Reusing research for {len(reused_entities)} entities: {', '.join(reused_entities[:3])}")

    # Check if gaps are similar to already executed searches
    existing_queries = set()
    for plan_item in research_plan:
        query = plan_item.get("query", "").lower()
        existing_queries.add(query)

    # Check shared pool for previous searches too
    cached_searches = shared_pool.get("search_cache", {})
    for cached_query in cached_searches.keys():
        existing_queries.add(cached_query.lower())

    deduplicated_gaps = []
    duplicate_gaps = []

    for gap in knowledge_gaps:
        gap_lower = gap.lower()
        # Check if this gap is too similar to existing queries
        is_duplicate = False
        for existing_query in existing_queries:
            # Simple similarity check: if gap contains most words from existing query, it's likely duplicate
            gap_words = set(gap_lower.split())
            query_words = set(existing_query.split())
            overlap = len(gap_words & query_words)
            total = len(gap_words | query_words)

            if overlap / max(total, 1) > 0.6:  # 60% word overlap = duplicate
                is_duplicate = True
                duplicate_gaps.append(gap)
                break

        if not is_duplicate:
            deduplicated_gaps.append(gap)

    if duplicate_gaps:
        print(f"[DEEP DIVE - DEDUP] Skipping {len(duplicate_gaps)} redundant gap searches (already covered by previous searches)")
        print(f"  Examples: {', '.join(duplicate_gaps[:2])}")

    knowledge_gaps = deduplicated_gaps  # Only research unique gaps

    if new_entities:
        print(f"[DEEP DIVE] Researching {len(new_entities)} NEW entities + {len(knowledge_gaps)} unique gaps...")
    elif knowledge_gaps:
        print(f"[DEEP DIVE] Researching {len(knowledge_gaps)} unique gaps (all entities already researched)...")
    else:
        print(f"[DEEP DIVE] All entities already researched, no unique gaps to fill")
        return {}

    discovered_entities = new_entities  # Only research new entities

    # Build queries for parallel execution
    queries_with_meta = []

    for entity in discovered_entities:
        queries_with_meta.append((entity, {"type": "entity", "subject": entity, "priority": "high"}))

    # Add gap-filling queries - ensure they're search-friendly
    for gap in knowledge_gaps:
        queries_with_meta.append((gap, {"type": "gap", "subject": gap, "priority": "medium"}))

    search_results = await parallel_search_with_rate_limit(
        queries=queries_with_meta,
        search_api=search_api,
        max_results=max_results,
        max_concurrent=2  # Rate limiting - reduced to avoid API limits
    )

    # Process results with early quality filtering
    deep_dive_results = {}
    relevance_scores = state.get("source_relevance_scores", {}).copy()

    for query, results_list, metadata in search_results:
        search_type = metadata.get("type", "unknown")
        subject = metadata.get("subject", "")[:30]

        print(f"  [{search_type.upper()}] {subject}... → {len(results_list)} results")

        # Early quality filtering
        results_list = filter_results_by_domain_quality(results_list, min_score=0.3)

        # Store filtered results
        for result_idx, result in enumerate(results_list):
            source_url = result.get("url", "")
            source_title = result.get("title", "Untitled")
            source_content = result.get("content", "")

            if source_content:
                source_key = f"{source_title} ({source_url})" if source_url else source_title
                deep_dive_results[source_key] = source_content

                # High relevance for entity searches, medium for gap fills
                base_relevance = 0.85 if search_type == "entity" else 0.7
                relevance_scores[source_key] = base_relevance - (result_idx * 0.05)

    print(f"[DEEP DIVE] Found {len(deep_dive_results)} new sources from deep dive")

    # Merge with existing results
    combined_results = state.get("research_results", {}).copy()
    combined_results.update(deep_dive_results)

    return {
        "research_results": combined_results,
        "source_relevance_scores": relevance_scores,
        "research_depth": state.get("research_depth", 1) + 1,  # Increment depth
        "entities": []  # Clear entities to avoid legacy entity research
    }


async def synthesize_findings(state: SubResearcherGraphState) -> dict:
    """
    CRITICAL REQUIREMENTS:
    1. Include ALL information from ALL sources (nothing should be lost)
    2. Organize information thematically for better readability
    3. Preserve ALL sources with inline citations
    4. Return a complete report that can be merged with other subtopic reports
    5. The output should be LONGER and more comprehensive, not shorter

    Benefits:
    - Better organization for the writer LLM
    - All key information is extracted and cited
    - ALL sources are preserved with proper attribution
    - Ready to be merged into the final report
    """
    research_results = state.get("research_results", {})
    credibilities = state.get("source_credibilities", {})
    subtopic = state.get("subtopic", "Unknown Subtopic")
    main_topic = state.get("main_topic", "")

    if not research_results:
        print(f"[SUMMARIZE] No results to summarize for '{subtopic}'")
        return {
            "summarized_findings": f"No research findings available for {subtopic}."
        }

    print(f"[SUMMARIZE] Creating comprehensive findings report for '{subtopic}' from {len(research_results)} sources...")

    # Sort sources by credibility
    sorted_sources = sorted(
        research_results.items(),
        key=lambda x: credibilities.get(x[0], 0.5),
        reverse=True
    )

    # Build source list using source keys (not numbers yet)
    # Create a mapping from short IDs to full source keys for easier citation
    sources_text = ""
    source_key_to_id = {}  # Maps full source_key to short ID like "src1", "src2"
    id_to_source_key = {}  # Reverse mapping

    for idx, (source_key, content) in enumerate(sorted_sources, start=1):
        credibility = credibilities.get(source_key, 0.5)
        short_id = f"src{idx}"
        source_key_to_id[source_key] = short_id
        id_to_source_key[short_id] = source_key

        # Include full content (or substantial portion)
        sources_text += f"\n[{short_id}] {source_key} (credibility: {credibility:.2f})\n{content}\n"

    # Create comprehensive summarization prompt
    summarization_prompt = f"""You are a research synthesis specialist. Create a COMPREHENSIVE, organized report from ALL research findings.

**CRITICAL: LANGUAGE REQUIREMENT**
The main topic below is written in a specific language. You MUST write your ENTIRE report in the SAME language as the main topic.
If the topic is in Chinese (中文), write the entire report in Chinese.
If the topic is in English, write the entire report in English.
Match the topic's language EXACTLY. This is MANDATORY.

MAIN TOPIC: {main_topic}
SUBTOPIC: {subtopic}

RESEARCH SOURCES - ALL {len(sorted_sources)} SOURCES (cite using [src1], [src2], etc.):
{sources_text}

CRITICAL REQUIREMENTS:

1. **COMPREHENSIVE COVERAGE - USE ALL SOURCES**:
   - You MUST incorporate information from ALL {len(sorted_sources)} sources
   - Do NOT summarize or condense - EXPAND and organize
   - Include ALL data, statistics, examples, and details from every source (Every number from the search result should be kept)
   - Repeat key information verbatim from sources when important
   - This report should be LONGER than the raw sources, not shorter
   
2. **CONTENT RICHNESS**:
   - Extract and include ALL quantitative data: numbers, percentages, dates, projections
   - Include ALL named entities: organizations, people, specific programs, studies
   - Include ALL specific examples and case studies mentioned
   - Include ALL context: time periods, geographical scope, methodologies

3. **ORGANIZATION**:
   - Organize findings into 4-8 clear thematic sections
   - Use descriptive headers (## format)
   - Within each section, include ALL relevant information from sources
   - Use bullet points or paragraphs as appropriate for readability

4. **CITATION DISCIPLINE**:
   - Assign each source/url a single citation number in your tex
   - End with ### Sources that lists each source with corresponding numbers
   - IMPORTANT: Number sources sequentially without gaps (1,2,3,4...) in the final list regardless of which sources you choose
   - Example format:
      [1] Source Title: URL
      [2] Source Title: URL

5. **LENGTH EXPECTATION**:
   - Report should be as long as it needs to be to cover all sources
   - Be thorough, not brief

REMEMBER: A later LLM will merge this with other subtopic reports, so having ALL sources and ALL information is CRITICAL. Don't lose anything!

Return the complete organized report with inline citations using [srcX] format.
"""

    try:
        messages = [
            SystemMessage(content="You are a research synthesis specialist. Create comprehensive, well-organized reports that preserve ALL information from ALL sources with proper citations."),
            HumanMessage(content=summarization_prompt)
        ]

        response = await llm_quality.ainvoke(messages)
        summarized_findings = response.content

        print(f"[SUMMARIZE] Generated comprehensive report for '{subtopic}' ({len(summarized_findings)} chars)")

        return {
            "summarized_findings": summarized_findings
        }

    except Exception as e:
        print(f"[SUMMARIZE] Summarization failed: {e}")
        # Fallback: return all sources in a simple format
        fallback_summary = f"## Research Findings for {subtopic}\n\n"
        fallback_summary += f"Comprehensive findings from {len(research_results)} sources:\n\n"
        fallback_summary += "## Sources\n\n"
        for idx, (source_key, _) in enumerate(sorted_sources, start=1):
            fallback_summary += f"[{idx}] {source_key}\n"

        return {
            "summarized_findings": fallback_summary
        }


def should_deepen_research(state: SubResearcherGraphState) -> str:
    """
    Routing function: Decide if we need another round of deep dive research.

    Returns:
        "deepen" - Coverage is low, do deep dive research
        "finish" - Coverage is good, proceed to quality assessment
    """
    needs_deepening = state.get("needs_deepening", False)
    research_depth = state.get("research_depth", 1)
    max_depth = state.get("max_research_depth", 3)

    if needs_deepening and research_depth < max_depth:
        return "deepen"
    else:
        return "finish"


def create_subresearcher_graph():
    """
    Create the subresearcher graph with UNIFIED ITERATIVE RESEARCH and DYNAMIC DEEPENING.

    Flow:
    1. execute_research: Execute ALL searches from pre-generated plan in parallel
    2. assess_coverage_and_gaps: Analyze coverage, discover entities/gaps
    3. Routing decision:
       - If coverage low OR important entities discovered → deep_dive_research
       - If coverage good → assess_quality
    4. deep_dive_research: Research discovered entities and fill gaps
       → Loop back to assess_coverage_and_gaps (iterative deepening)
    5. assess_quality: Filter and score final sources
    6. summarize_findings: Create comprehensive organized report with ALL sources

    The graph can loop through steps 2-4 multiple times until max_depth is reached
    or coverage is satisfactory.
    """
    workflow = StateGraph(SubResearcherGraphState)

    workflow.add_node("research", execute_research)
    workflow.add_node("assess_coverage", assess_coverage_and_gaps)
    workflow.add_node("deep_dive", deep_dive_research)
    workflow.add_node("synthesize", synthesize_findings)

    workflow.set_entry_point("research")
    workflow.add_edge("research", "assess_coverage")

    # Either deepen or finish
    workflow.add_conditional_edges(
        "assess_coverage",
        should_deepen_research,
        {
            "deepen": "deep_dive",      # Coverage low → research more
            "finish": "synthesize"  # Coverage good → finish
        }
    )

    # After deep dive, re-assess coverage (may trigger another round)
    workflow.add_edge("deep_dive", "assess_coverage")

    # After quality assessment, summarize findings
    workflow.add_edge("synthesize", END)

    return workflow.compile()


# Create the graph
subresearcher_graph = create_subresearcher_graph()