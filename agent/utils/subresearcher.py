from typing import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from pydantic import create_model
from utils.model import llm_quality, llm
from utils.configuration import tavily_client
from utils.nodes.helpers import filter_quality_sources
import asyncio
from urllib.parse import urlparse
import re


class SubResearcherGraphState(TypedDict):
    # State for the subresearcher subgraph with iterative deepening
    subtopic_id: int
    subtopic: str  # The broad query
    main_topic: str
    other_subtopics: list[str]

    # Section-specific research guidance
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
    entities: list[str]  # Extracted entities to research
    research_plan: list[dict]  # Planned searches based on subtopics/questions
    completed_searches: int  # Track progress
    discovered_gaps: list[str]  # Knowledge gaps found during research
    discovered_entities: list[str]  # New entities/topics discovered in results
    coverage_score: float  # Estimated coverage of the topic (0.0-1.0)
    needs_deepening: bool  # Flag to trigger additional research round

    # Shared entities and findings from other sections
    shared_research_pool: dict


def quick_domain_quality_score(url: str) -> float:
    """
    Quick domain-based quality assessment without LLM.
    Returns a score from 0.0 (low quality) to 1.0 (high quality).

    This enables early filtering before expensive API calls.
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

        # Check high quality
        for pattern in high_quality_patterns:
            if pattern in domain:
                return 0.9

        # Check medium quality
        for pattern in medium_quality_patterns:
            if pattern in domain:
                return 0.6

        # Check low quality
        for pattern in low_quality_patterns:
            if pattern in domain or pattern in url.lower():
                return 0.2

        # Default: neutral quality
        return 0.5

    except Exception:
        return 0.5  # Default on parsing error


def filter_results_by_domain_quality(results: list[dict], min_score: float = 0.3) -> list[dict]:
    """
    Filter search results by domain quality before deep processing.
    Removes low-quality domains early to save API quota.

    Args:
        results: List of search results with 'url' field
        min_score: Minimum quality score to keep (0.0-1.0)

    Returns:
        Filtered list of results
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
    max_concurrent: int = 3
) -> list[tuple[str, list[dict]]]:
    """
    Execute multiple search queries in parallel with rate limiting.

    Args:
        queries: List of (query_string, metadata) tuples
        search_api: "tavily" or "serper"
        max_results: Max results per query
        max_concurrent: Max concurrent searches (default 3 for rate limits)

    Returns:
        List of (query_string, results_list) tuples
    """
    semaphore = asyncio.Semaphore(max_concurrent)

    async def throttled_search(query: str, metadata: dict) -> tuple[str, list[dict], dict]:
        """Execute a single search with rate limiting"""
        async with semaphore:
            try:
                loop = asyncio.get_event_loop()

                if search_api == "tavily":
                    search_results = await loop.run_in_executor(
                        None,
                        lambda: tavily_client.search(query=query, max_results=max_results, include_raw_content=True)
                    )
                    results_list = search_results.get("results", [])[:max_results]

                    # Smaller delay for Tavily (more permissive rate limits)
                    await asyncio.sleep(0.1)

                return (query, results_list, metadata)

            except Exception as e:
                print(f"    Search failed for '{query[:60]}...': {e}")
                return (query, [], metadata)

    # Execute all searches in parallel with rate limiting
    tasks = [throttled_search(query, meta) for query, meta in queries]
    results = await asyncio.gather(*tasks)

    return results


# ============================================================================
# ITERATIVE DEEPENING NODES
# ============================================================================

async def plan_research_strategy(state: SubResearcherGraphState) -> dict:
    """
    OPTIMIZATION: Research plans are now pre-generated during outline creation.
    This node simply validates and returns the pre-generated plan.

    This saves 1 LLM call per section (typically 3-5 calls per report).
    """
    pre_generated_plan = state.get("research_plan", [])
    section_title = state.get("subtopic", "")

    # All research plans should be pre-generated by write_outline()
    if not pre_generated_plan or len(pre_generated_plan) == 0:
        raise ValueError(
            f"No pre-generated research plan found for section '{section_title}'. "
            "Outline generation should create research plans for all sections."
        )

    print(f"[RESEARCH PLANNING] Using pre-generated plan from outline (saved 1 LLM call)")
    print(f"  Section: {section_title}")
    print(f"  Plan has {len(pre_generated_plan)} searches:")
    for search in pre_generated_plan[:3]:
        priority = search.get("priority", "unknown")
        query = search.get("query", "")[:60]
        print(f"    - [{priority.upper()}] {query}...")

    if len(pre_generated_plan) > 3:
        print(f"    ... and {len(pre_generated_plan) - 3} more")

    return {
        "research_plan": pre_generated_plan,
        "completed_searches": 0
    }


async def execute_unified_research(state: SubResearcherGraphState) -> dict:
    """
    UNIFIED ITERATIVE RESEARCH: Execute ALL searches from research plan in parallel.

    This replaces the old two-step approach (initial_broad_search + execute_targeted_searches)
    with a single unified execution that runs all queries in parallel for maximum efficiency.

    Uses pre-generated research plan from generate_research_plan (outline node).
    """
    research_plan = state.get("research_plan", [])
    search_api = state.get("search_api", "tavily")
    max_results = state.get("max_search_results", 3)
    shared_pool = state.get("shared_research_pool", {})
    section_title = state.get("subtopic", "")

    if not research_plan:
        print(f"[UNIFIED RESEARCH] ERROR: No research plan provided for section '{section_title}'")
        return {
            "research_results": {},
            "source_relevance_scores": {},
            "research_depth": 1
        }

    print(f"\n[UNIFIED RESEARCH] Executing {len(research_plan)} searches in parallel for: {section_title}")

    # Build queries for parallel execution from research plan
    queries_with_meta = []
    cached_searches = shared_pool.get("search_cache", {})
    cache_hits = []

    for idx, search_spec in enumerate(research_plan):
        query = search_spec.get("query", "")
        priority = search_spec.get("priority", "medium")

        # Check cache
        if query in cached_searches:
            cache_hits.append((query, cached_searches[query], {"priority": priority, "index": idx}))
        else:
            queries_with_meta.append((query, {"priority": priority, "index": idx}))

    if cache_hits:
        print(f"[UNIFIED RESEARCH] Cache hits: {len(cache_hits)} searches (saved API calls)")

    # Execute uncached searches in parallel
    search_results = []
    if queries_with_meta:
        print(f"[UNIFIED RESEARCH] Executing {len(queries_with_meta)} new searches...")
        search_results = await parallel_search_with_rate_limit(
            queries=queries_with_meta,
            search_api=search_api,
            max_results=max_results,
            max_concurrent=5  # Rate limiting
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

        # Quality filtering - use uniform threshold
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
                if search_score > 0:
                    relevance_scores[source_key] = search_score
                else:
                    relevance_scores[source_key] = base_relevance - (result_idx * 0.05)

    print(f"[UNIFIED RESEARCH] Total: {len(all_results)} unique sources")
    print(f"[UNIFIED RESEARCH] Quality filtering: {total_before_filter} → {total_after_filter} sources")

    # Update shared pool with new cache entries
    update_dict = {
        "research_results": all_results,
        "source_relevance_scores": relevance_scores,
        "research_depth": 1
    }

    if len(updated_cache) > len(cached_searches):
        updated_pool = {**shared_pool, "search_cache": updated_cache}
        update_dict["shared_research_pool"] = updated_pool
        print(f"[UNIFIED RESEARCH] Updated cache: {len(updated_cache) - len(cached_searches)} new searches cached")

    return update_dict


async def deep_research_entities(state: SubResearcherGraphState) -> dict:
    """
    Step 3: Deep research on each extracted entity with PARALLEL SEARCH STRATEGIES.
    For each entity, searches academic + news + social sources simultaneously.
    Combines with initial results.
    """
    entities = state.get("entities", [])
    main_topic = state.get("main_topic", "")
    search_api = state.get("search_api", "tavily")
    max_results = state.get("max_search_results", 3)
    initial_results = state.get("research_results", {})

    if not entities:
        print(f"[Deep research] No entities to research, keeping initial results")
        return {}

    print(f"[Deep research] Deep research on {len(entities)} entities with parallel search strategies...")

    async def search_academic(entity: str):
        """Search academic/research sources for entity"""
        # Trust search API's ranking - no site restrictions
        academic_query = f"{entity} {main_topic} academic research papers"
        try:
            if search_api == "tavily":
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    None,
                    lambda: tavily_client.search(query=academic_query, max_results=max_results)
                )
                return results.get("results", [])[:max_results]
        except Exception as e:
            print(f"[Deep research]   - Academic search failed for {entity}: {e}")
            return []

    async def search_news(entity: str):
        """Search news sources for entity"""
        # Trust search API's ranking - no site restrictions
        news_query = f"{entity} {main_topic} latest news updates"
        try:
            if search_api == "tavily":
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    None,
                    lambda: tavily_client.search(query=news_query, max_results=max_results)
                )
                return results.get("results", [])[:max_results]
        except Exception as e:
            print(f"[Deep research]   - News search failed for {entity}: {e}")
            return []

    async def search_social_web(entity: str):
        """Search general web and social sources for entity"""
        general_query = f"{entity} {main_topic}"
        try:
            if search_api == "tavily":
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    None,
                    lambda: tavily_client.search(query=general_query, max_results=max_results)
                )
                return results.get("results", [])[:max_results]
        except Exception as e:
            print(f"[Deep research]   - Social/web search failed for {entity}: {e}")
            return []

    async def research_entity_parallel(entity: str):
        """Research a single entity with parallel search strategies"""
        print(f"[Deep research]   - {entity}: parallel searches (academic + news + social)")

        # Run all 3 searches in parallel
        academic_results, news_results, social_results = await asyncio.gather(
            search_academic(entity),
            search_news(entity),
            search_social_web(entity)
        )

        # Combine all results
        all_results = academic_results + news_results + social_results

        # Convert to dict
        entity_results = {}
        for result in all_results:
            source_url = result.get("url", "")
            source_title = result.get("title", "Untitled")
            source_content = result.get("content", "")

            if source_content:
                source_key = f"{source_title} ({source_url})"
                entity_results[source_key] = source_content

        print(f"[Deep research]   - {entity}: found {len(entity_results)} sources (academic={len(academic_results)}, news={len(news_results)}, social={len(social_results)})")
        return entity_results

    # Research all entities with rate limiting
    all_entity_results = []
    if search_api == "tavily":
        tasks = [research_entity_parallel(entity) for entity in entities]
        all_entity_results = await asyncio.gather(*tasks)

    # Combine all results (initial + entity research) with relevance tracking
    combined_results = {**initial_results}
    combined_relevance = state.get("source_relevance_scores", {}).copy()

    entity_idx = 0
    for entity_results in all_entity_results:
        for source_key, content in entity_results.items():
            combined_results[source_key] = content
            # Deep research sources: relevance based on entity order (earlier entities = more important)
            if source_key not in combined_relevance:
                combined_relevance[source_key] = 0.7 - (entity_idx * 0.1)  # Start at 0.7, decrease by entity
        entity_idx += 1

    print(f"[Deep research] Combined total: {len(combined_results)} sources from parallel searches")

    # Print sample of deep research findings
    if all_entity_results:
        print(f"[Deep research] Deep research sample findings:")
        for idx, entity_results in enumerate(all_entity_results[:2]):
            entity_name = entities[idx] if idx < len(entities) else "Unknown"
            if entity_results:
                sample_key = list(entity_results.keys())[0]
                content_preview = entity_results[sample_key][:150].replace('\n', ' ')
                print(f"    Entity '{entity_name}': {len(entity_results)} sources | Sample: {content_preview}...")

    return {
        "research_results": combined_results,
        "source_relevance_scores": combined_relevance,
        "research_depth": 2
    }


async def assess_coverage_and_gaps(state: SubResearcherGraphState) -> dict:
    """
    DYNAMIC DEEPENING: Analyze research coverage and discover gaps/entities.

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
        needs_deeper_research=(bool, ...),  # Should we go deeper?
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

3. **Knowledge Gaps**: Identify 1-3 HIGHLY SPECIFIC, ACTIONABLE search queries:
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
    DYNAMIC DEEPENING: Execute additional research based on discovered gaps/entities.

    This runs AFTER assess_coverage_and_gaps and performs targeted searches for:
    1. Discovered entities (specific items that need investigation)
    2. Knowledge gaps (unanswered questions)

    CROSS-SECTION LEARNING: Checks shared_research_pool first to avoid redundant research.
    """
    discovered_entities = state.get("discovered_entities", [])
    knowledge_gaps = state.get("discovered_gaps", [])
    search_api = state.get("search_api", "tavily")
    max_results = state.get("max_search_results", 3)
    main_topic = state.get("main_topic", "")
    shared_pool = state.get("shared_research_pool", {})
    research_plan = state.get("research_plan", [])

    if not discovered_entities and not knowledge_gaps:
        print(f"[DEEP DIVE] No gaps or entities to research")
        return {}

    # DEDUPLICATION #1: Check shared pool for entities already researched by other sections
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

    # DEDUPLICATION #2: Check if gaps are similar to already executed searches
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

    # Add entity-specific queries - make them more specific and actionable
    for entity in discovered_entities:
        # Filter out low-value entities:
        # - Skip very long names (likely not useful)
        # - Skip organization/brand names (unless they're key concepts)
        # - Skip generic terms
        entity_lower = entity.lower()
        skip_indicators = [
            len(entity.split()) > 4,  # Too long
            any(word in entity_lower for word in ["inc", "llc", "company", "organization", "association"]),
            entity_lower in ["usa weightlifting", "barbell medicine", "westside barbell", "elitefts"]  # Common orgs
        ]
        
        if any(skip_indicators):
            continue
            
        # Create more specific query that focuses on the comparison aspect
        query = f"{entity} {main_topic}"
        queries_with_meta.append((query, {"type": "entity", "subject": entity, "priority": "high"}))

    # Add gap-filling queries - ensure they're search-friendly
    for gap in knowledge_gaps:
        # If gap is already a search query (no question mark, action-oriented), use it directly
        # Otherwise, convert question to search query
        if "?" in gap:
            # Convert question to search query: remove question words, make it action-oriented
            query = gap.replace("What", "").replace("How", "").replace("Are there", "").replace("?", "").strip()
            query = f"{query} {main_topic}" if main_topic not in query else query
        else:
            query = gap if len(gap) > 15 else f"{gap} {main_topic}"
        queries_with_meta.append((query, {"type": "gap", "subject": gap, "priority": "medium"}))

    # Execute all deep dive searches in parallel
    max_concurrent = 3 if search_api == "serper" else 5
    search_results = await parallel_search_with_rate_limit(
        queries=queries_with_meta,
        search_api=search_api,
        max_results=max_results,
        max_concurrent=max_concurrent
    )

    # Process results with early quality filtering
    deep_dive_results = {}
    relevance_scores = state.get("source_relevance_scores", {}).copy()

    for query, results_list, metadata in search_results:
        search_type = metadata.get("type", "unknown")
        priority = metadata.get("priority", "medium")
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


async def assess_quality(state: SubResearcherGraphState) -> dict:
    """
    Final step: Assess quality and assign credibility scores to sources.
    """
    research_results = state.get("research_results", {})


    if not research_results:
        print(f"[Subresearcher] No results to assess")
        return {
            "source_credibilities": {}
        }

    print(f"[Subresearcher] Assessing quality of {len(research_results)} sources...")

    # Convert dict to list format for quality filter
    search_results_list = []
    for source_key, content in research_results.items():
        # Parse source_key which is in format "Title (URL)"
        if '(' in source_key and ')' in source_key:
            title = source_key.split('(')[0].strip()
            url = source_key.split('(')[1].split(')')[0]
        else:
            title = source_key
            url = ""

        search_results_list.append({
            "url": url,
            "title": title,
            "content": content
        })

    # Use existing quality filter
    filtered = await filter_quality_sources(
        search_results=search_results_list,
        min_credibility=0.
    )

    credibilities = {}
    filtered_results = {}

    for result, credibility in filtered:
        # Reconstruct source_key
        title = result.get("title", "")
        url = result.get("url", "")
        content = result.get("content", "")
        source_key = f"{title} ({url})" if url else title

        credibilities[source_key] = credibility
        filtered_results[source_key] = content

    print(f"[Subresearcher] Quality assessment complete: {len(filtered_results)} sources passed")

    # Show credibility distribution
    if credibilities:
        high_quality = sum(1 for c in credibilities.values() if c >= 0.7)
        medium_quality = sum(1 for c in credibilities.values() if 0.5 <= c < 0.7)
        low_quality = sum(1 for c in credibilities.values() if c < 0.5)
        print(f"[Subresearcher] Credibility distribution: High (≥0.7)={high_quality}, Medium (0.5-0.7)={medium_quality}, Low (<0.5)={low_quality}")

    return {
        "research_results": filtered_results,
        "source_credibilities": credibilities
    }


async def summarize_findings(state: SubResearcherGraphState) -> dict:
    """
    NEW FINAL NODE: Synthesize research findings into a COMPREHENSIVE, organized report.

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
        content_to_use = content[:2500] if len(content) > 2500 else content
        sources_text += f"\n[{short_id}] {source_key} (credibility: {credibility:.2f})\n{content_to_use}\n"

    # Create comprehensive summarization prompt
    summarization_prompt = f"""You are a research synthesis specialist. Create a COMPREHENSIVE, organized report from ALL research findings.

MAIN TOPIC: {main_topic}
SUBTOPIC: {subtopic}

RESEARCH SOURCES - ALL {len(sorted_sources)} SOURCES (cite using [src1], [src2], etc.):
{sources_text}

CRITICAL REQUIREMENTS:

1. **COMPREHENSIVE COVERAGE - USE ALL SOURCES**:
   - You MUST incorporate information from ALL {len(sorted_sources)} sources
   - Do NOT summarize or condense - EXPAND and organize
   - Include ALL specific data, statistics, examples, and details from every source
   - Repeat key information verbatim from sources when important
   - This report should be LONGER than the raw sources, not shorter

2. **ORGANIZATION**:
   - Organize findings into 4-8 clear thematic sections
   - Use descriptive headers (## format)
   - Within each section, include ALL relevant information from sources
   - Use bullet points or paragraphs as appropriate for readability

3. **CITATION DISCIPLINE**:
   - EVERY fact, statistic, claim, or piece of information MUST have a citation using the source IDs
   - Use inline citations immediately after each fact: "X happened [src3]"
   - Multiple sources for the same fact: "Y increased 15% [src2][src7][src12]"
   - Every source ID from [src1] to [src{len(sorted_sources)}] should appear at least once
   - If a source doesn't fit naturally, create an "Additional Findings" section

4. **CONTENT RICHNESS**:
   - Extract and include ALL quantitative data: numbers, percentages, dates, projections
   - Include ALL named entities: organizations, people, specific programs, studies
   - Include ALL specific examples and case studies mentioned
   - Include ALL context: time periods, geographical scope, methodologies

5. **NO SOURCES SECTION YET**:
   - Do NOT include a "## Sources" section in your output
   - I will add the sources section separately with proper numbering
   - Just use the [srcX] citations in your text

6. **LENGTH EXPECTATION**:
   - Target length: 800-2000 words (or more if needed to cover all sources)
   - Each major section: 200-400 words minimum
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

        # Extract which source IDs were actually cited (e.g., src1, src5, src12)
        citation_matches = re.findall(r'\[src(\d+)\]', summarized_findings)
        cited_src_ids = set(f"src{c}" for c in citation_matches if c.isdigit())

        print(f"[SUMMARIZE] Found {len(cited_src_ids)} cited sources out of {len(sorted_sources)} available")

        # Now renumber citations sequentially and build sources list
        # Map old src IDs to new sequential numbers
        src_id_to_number = {}
        number_to_source_key = {}
        current_number = 1

        # First pass: assign numbers to cited sources in order of appearance
        for src_id in sorted(cited_src_ids, key=lambda x: int(x.replace('src', ''))):
            src_id_to_number[src_id] = current_number
            source_key = id_to_source_key.get(src_id, "Unknown Source")
            number_to_source_key[current_number] = source_key
            current_number += 1

        # Replace all [srcX] with [numbered] citations
        final_report = summarized_findings
        # Sort by src number descending to avoid replacing src1 before src10
        for src_id in sorted(cited_src_ids, key=lambda x: int(x.replace('src', '')), reverse=True):
            new_number = src_id_to_number[src_id]
            final_report = re.sub(rf'\[{src_id}\]', f'[{new_number}]', final_report)

        # Add Sources section with sequential numbering
        final_report += "\n\n## Sources\n\n"
        for number in sorted(number_to_source_key.keys()):
            source_key = number_to_source_key[number]
            final_report += f"[{number}] {source_key}\n"

        print(f"[SUMMARIZE] ✓ Renumbered {len(cited_src_ids)} sources sequentially (1-{len(cited_src_ids)})")

        return {
            "summarized_findings": final_report
        }

    except Exception as e:
        print(f"[SUMMARIZE] Summarization failed: {e}")
        # Fallback: return all sources in a simple format
        fallback_summary = f"## Research Findings for {subtopic}\n\n"
        fallback_summary += f"Comprehensive findings from {len(research_results)} sources:\n\n"

        for idx, (source_key, content) in enumerate(sorted_sources, start=1):
            fallback_summary += f"### Source [{idx}]: {source_key[:150]}\n\n"
            fallback_summary += f"{content}\n\n"

        fallback_summary += "## Sources\n\n"
        for idx, source_key in enumerate(all_source_keys, start=1):
            fallback_summary += f"[{idx}] {source_key}\n"

        return {
            "summarized_findings": fallback_summary
        }


# ============================================================================
# GRAPH CREATION
# ============================================================================

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
    1. plan_research_strategy: Validate pre-generated research plan from outline
    2. execute_unified_research: Execute ALL searches (primary + targeted) in parallel
    3. assess_coverage_and_gaps: Analyze coverage, discover entities/gaps
    4. Routing decision:
       - If coverage low OR important entities discovered → deep_dive_research
       - If coverage good → assess_quality
    5. deep_dive_research: Research discovered entities and fill gaps
       → Loop back to assess_coverage_and_gaps (iterative deepening)
    6. assess_quality: Filter and score final sources
    7. summarize_findings: Create comprehensive organized report with ALL sources

    The graph can loop through steps 3-5 multiple times until max_depth is reached
    or coverage is satisfactory.

    IMPROVEMENTS:
    - Unified research execution (no separate broad/targeted steps)
    - All planned searches run in parallel for maximum efficiency
    - Uses pre-generated plan from generate_research_plan
    - Iterative deepening based on coverage assessment
    - Final summarization node organizes findings for writer LLM
    """
    workflow = StateGraph(SubResearcherGraphState)

    # Add nodes
    workflow.add_node("plan_research", plan_research_strategy)
    workflow.add_node("unified_research", execute_unified_research)
    workflow.add_node("assess_coverage", assess_coverage_and_gaps)
    workflow.add_node("deep_dive", deep_dive_research)
    workflow.add_node("assess_quality", assess_quality)
    workflow.add_node("summarize", summarize_findings)

    # Unified research flow
    workflow.set_entry_point("plan_research")
    workflow.add_edge("plan_research", "unified_research")
    workflow.add_edge("unified_research", "assess_coverage")

    # Either deepen or finish
    workflow.add_conditional_edges(
        "assess_coverage",
        should_deepen_research,
        {
            "deepen": "deep_dive",      # Coverage low → research more
            "finish": "assess_quality"  # Coverage good → finish
        }
    )

    # After deep dive, re-assess coverage (may trigger another round)
    workflow.add_edge("deep_dive", "assess_coverage")

    # After quality assessment, summarize findings
    workflow.add_edge("assess_quality", "summarize")
    workflow.add_edge("summarize", END)

    return workflow.compile()


# Create the graph
subresearcher_graph = create_subresearcher_graph()