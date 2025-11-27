from typing import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from pydantic import create_model
from utils.model import llm
from utils.configuration import tavily_client, serper_client, fetch_full_content
from utils.nodes.helpers import filter_quality_sources
import asyncio
from urllib.parse import urlparse


class SubResearcherGraphState(TypedDict):
    """State for the subresearcher subgraph with iterative deepening"""
    subtopic_id: int
    subtopic: str  # The broad query
    main_topic: str
    other_subtopics: list[str]

    # Section-specific research guidance
    section_subtopics: list[str]  # Specific subtopics from outline

    # Final outputs
    research_results: dict[str, str]  # All combined results
    source_credibilities: dict[str, float]
    source_relevance_scores: dict[str, float]  # NEW: Track search API relevance scores
    research_depth: int

    # Config
    max_search_results: int
    max_research_depth: int
    search_api: str
    enable_mcp_fetch: bool
    max_mcp_fetches: int

    # Internal state for iterative deepening
    entities: list[str]  # Extracted entities to research
    research_plan: list[dict]  # Planned searches based on subtopics/questions
    completed_searches: int  # Track progress

    # DYNAMIC DEEPENING STATE
    discovered_gaps: list[str]  # Knowledge gaps found during research
    discovered_entities: list[str]  # New entities/topics discovered in results
    coverage_score: float  # Estimated coverage of the topic (0.0-1.0)
    needs_deepening: bool  # Flag to trigger additional research round

    # SHARED KNOWLEDGE POOL (cross-section learning)
    shared_research_pool: dict  # Shared entities and findings from other sections


# ============================================================================
# HELPER FUNCTIONS - QUALITY FILTERING & PARALLELIZATION
# ============================================================================

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

                if search_api == "serper":
                    search_results = await loop.run_in_executor(
                        None,
                        lambda: serper_client.search(query=query, max_results=max_results)
                    )
                    results_list = search_results.get("results", [])[:max_results]

                    # Small delay for Serper to respect rate limits
                    await asyncio.sleep(0.3)
                else:  # tavily
                    search_results = await loop.run_in_executor(
                        None,
                        lambda: tavily_client.search(query=query, max_results=max_results)
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
    NEW Step 1: Analyze subtopics and key questions to plan targeted research.

    Uses LLM to:
    1. Assess relevance of each subtopic/question to main topic
    2. Determine scope and complexity
    3. Prioritize what to research deeply vs broadly
    4. Create a research plan
    """
    main_topic = state.get("main_topic", "")
    section_title = state.get("subtopic", "")
    section_subtopics = state.get("section_subtopics", [])
    max_depth = state.get("max_research_depth", 3)

    # If no guidance provided, fall back to simple broad search
    if not section_subtopics:
        print(f"[RESEARCH PLANNING] No subtopics provided, will do broad search only")
        return {
            "research_plan": [],
            "completed_searches": 0
        }

    print(f"[RESEARCH PLANNING] Planning research for '{section_title}'")
    print(f"  Main topic: {main_topic}")
    print(f"  Subtopics: {section_subtopics}")

    TargetedSearch = create_model(
        'TargetedSearch',
        query=(str, ...),
        priority=(str, ...),
        reasoning=(str, ...)
    )

    ResearchPlanOutput = create_model(
        'ResearchPlanOutput',
        primary_query=(str, ...),
        targeted_searches=(list[TargetedSearch], ...),
        reasoning=(str, ...)
    )

    llm_structured = llm.with_structured_output(ResearchPlanOutput)

    prompt = f"""You are a research strategist. Plan an efficient, targeted research strategy.

MAIN TOPIC: {main_topic}
SECTION TO RESEARCH: {section_title}

SUBTOPICS TO COVER:
{chr(10).join(f"- {st}" for st in section_subtopics)}

MAX RESEARCH DEPTH: {max_depth} (1=broad only, 2=broad+deep, 3=broad+deep+targeted)

CREATE A RESEARCH PLAN:

1. PRIMARY QUERY: One broad search query that captures the section's main focus
   - Should combine main topic + section title
   - Example: "most popular MMORPGs 2025 player statistics"

2. TARGETED SEARCHES: List of focused searches for subtopics/questions
   - Only include subtopics/questions that are:
     a) HIGHLY RELEVANT to the main topic
     b) SPECIFIC enough to warrant targeted research
     c) COMPLEX enough to need deep investigation

   For each targeted search, provide:
   - query: The search query
   - priority: "high", "medium", or "low"
   - reasoning: Why this needs targeted research (1 sentence)

   IMPORTANT:
   - Limit to 3-5 targeted searches (quality over quantity)
   - Skip subtopics that are too broad or off-topic
   - Prioritize questions over subtopics (questions are more specific)

3. REASONING: Explain your research strategy (2-3 sentences)

Return structured output with:
- primary_query: str
- targeted_searches: list of dicts with {{query, priority, reasoning}}
- reasoning: str
"""

    try:
        messages = [
            SystemMessage(content="You are a research planning expert. Create efficient, targeted research plans."),
            HumanMessage(content=prompt)
        ]

        response = await llm_structured.ainvoke(messages)
        all_targeted = response.targeted_searches

        # Limit based on research depth
        if max_depth <= 1:
            targeted_searches = []  # Broad search only
        elif max_depth == 2:
            targeted_searches = [s for s in all_targeted if s.priority == "high"][:3]
        else:  # max_depth >= 3
            targeted_searches = [s for s in all_targeted if s.priority in ["high", "medium"]][:5]

        print(f"[RESEARCH PLANNING] Strategy created:")
        print(f"  Primary query: {response.primary_query}")
        print(f"  Targeted searches: {len(targeted_searches)} planned")
        for ts in targeted_searches:
            print(f"    - [{ts.priority.upper()}] {ts.query[:60]}...")
        print(f"  Reasoning: {response.reasoning}")

        return {
            "research_plan": [
                {"query": response.primary_query, "type": "primary"},
                *[{"query": s.query, "type": "targeted", "priority": s.priority} for s in targeted_searches]
            ],
            "completed_searches": 0
        }

    except Exception as e:
        print(f"[RESEARCH PLANNING] Failed to create plan: {e}")
        # Fallback: use first question or section title
        fallback_query = section_subtopics[0] if section_subtopics else f"{main_topic} {section_title}"
        return {
            "research_plan": [{"query": fallback_query, "type": "primary"}],
            "completed_searches": 0
        }


async def initial_broad_search(state: SubResearcherGraphState) -> dict:
    """
    Step 2: Execute initial broad search using research plan.
    Uses primary query from research plan, or falls back to subtopic.
    """
    research_plan = state.get("research_plan", [])
    search_api = state.get("search_api", "tavily")
    max_results = state.get("max_search_results", 3)

    # Use primary query from research plan, or fallback to subtopic
    if research_plan:
        primary_search = next((p for p in research_plan if p.get("type") == "primary"), None)
        broad_query = primary_search.get("query") if primary_search else state.get("subtopic", "")
    else:
        broad_query = state.get("subtopic", "")

    print(f"[DEPTH 1: Initial Search] Query: {broad_query[:100]}...")

    # Single search on the broad query
    try:
        if search_api == "serper":
            loop = asyncio.get_event_loop()
            search_results = await loop.run_in_executor(
                None,
                lambda: serper_client.search(query=broad_query, max_results=max_results)
            )
            # SerperClient already returns Tavily-compatible format {"results": [...]}
            search_results_list = search_results.get("results", [])[:max_results]
        else:  # tavily (default)
            loop = asyncio.get_event_loop()
            search_results = await loop.run_in_executor(
                None,
                lambda: tavily_client.search(query=broad_query, max_results=max_results)
            )
            search_results_list = search_results.get("results", [])[:max_results]

    except Exception as e:
        print(f"[DEPTH 1: Initial Search] Initial search failed: {e}")
        search_results_list = []

    # Early quality filtering - filter by domain quality BEFORE processing
    initial_count = len(search_results_list)
    search_results_list = filter_results_by_domain_quality(search_results_list, min_score=0.5)
    filtered_count = len(search_results_list)

    if filtered_count < initial_count:
        print(f"[DEPTH 1: Initial Search] Early filtering: {initial_count} → {filtered_count} sources (removed {initial_count - filtered_count} low-quality domains)")

    # Store initial results with relevance scores
    initial_results = {}
    relevance_scores = {}
    initial_domain_scores = {}  # Track domain quality scores

    for idx, result in enumerate(search_results_list):
        source_url = result.get("url", "")
        source_title = result.get("title", "Untitled")
        source_content = result.get("content", "")
        search_score = result.get("score", 0.0)  # Search API relevance score

        if source_content:
            source_key = f"{source_title} ({source_url})"
            initial_results[source_key] = source_content

            # Store domain quality for later use
            initial_domain_scores[source_key] = quick_domain_quality_score(source_url)

            # Relevance: Use search score if available, otherwise use position (earlier = more relevant)
            if search_score > 0:
                relevance_scores[source_key] = search_score
            else:
                # Position-based relevance: first result = 1.0, last = 0.5
                relevance_scores[source_key] = 1.0 - (idx * 0.5 / max(len(search_results_list), 1))

    print(f"[DEPTH 1: Initial Search] Found {len(initial_results)} sources")

    # Print sample of what was found
    if initial_results:
        sample_keys = list(initial_results.keys())[:2]
        for key in sample_keys:
            content = initial_results[key]
            preview = content[:150].replace('\n', ' ')
            print(f"    Sample: {key[:50]}... | {preview}...")

    return {
        "research_results": initial_results,
        "source_relevance_scores": relevance_scores,
        "research_depth": 1
    }


async def execute_targeted_searches(state: SubResearcherGraphState) -> dict:
    """
    Step 3: Execute targeted searches from research plan with PARALLEL EXECUTION.

    IMPROVEMENT #3: Uses parallel_search_with_rate_limit for faster, rate-limited searches.
    """
    research_plan = state.get("research_plan", [])
    search_api = state.get("search_api", "tavily")
    max_results = state.get("max_search_results", 3)
    max_depth = state.get("max_research_depth", 1)

    # Filter for targeted searches only
    targeted_searches = [p for p in research_plan if p.get("type") == "targeted"]

    # If max_depth is 1 or no targeted searches, skip
    if max_depth <= 1 or not targeted_searches:
        print(f"[DEPTH 1→2: Targeted Searches] Skipping (max_depth={max_depth}, targeted={len(targeted_searches)})")
        return {"entities": []}

    print(f"[DEPTH 1→2: Targeted Searches] Executing {len(targeted_searches)} searches in PARALLEL...")

    # IMPROVEMENT #3: Execute all searches in parallel with rate limiting
    # Prepare queries with metadata
    queries_with_meta = [
        (search_spec.get("query", ""), {"priority": search_spec.get("priority", "medium"), "index": idx})
        for idx, search_spec in enumerate(targeted_searches)
    ]

    # Execute parallel searches with semaphore-based rate limiting
    # Serper: max 3 concurrent, Tavily: max 5 concurrent
    max_concurrent = 3 if search_api == "serper" else 5
    search_results = await parallel_search_with_rate_limit(
        queries=queries_with_meta,
        search_api=search_api,
        max_results=max_results,
        max_concurrent=max_concurrent
    )

    # Process results with early quality filtering
    all_targeted_results = {}
    relevance_scores = state.get("source_relevance_scores", {}).copy()
    total_before_filter = 0
    total_after_filter = 0

    for query, results_list, metadata in search_results:
        priority = metadata.get("priority", "medium")
        print(f"  [{priority.upper()}] {query[:80]}... → {len(results_list)} results")

        # IMPROVEMENT #2: Early quality filtering
        total_before_filter += len(results_list)
        results_list = filter_results_by_domain_quality(results_list, min_score=0.3)
        total_after_filter += len(results_list)

        # Store filtered results
        for result_idx, result in enumerate(results_list):
            source_url = result.get("url", "")
            source_title = result.get("title", "Untitled")
            source_content = result.get("content", "")

            if source_content:
                source_key = f"{source_title} ({source_url})" if source_url else source_title
                all_targeted_results[source_key] = source_content

                # Higher relevance for high-priority searches
                base_relevance = 0.8 if priority == "high" else 0.6
                relevance_scores[source_key] = base_relevance - (result_idx * 0.1)

    if total_before_filter > total_after_filter:
        print(f"[DEPTH 1→2: Targeted Searches] Early filtering: {total_before_filter} → {total_after_filter} sources (removed {total_before_filter - total_after_filter})")

    print(f"[DEPTH 1→2: Targeted Searches] Total: {len(all_targeted_results)} unique sources from parallel searches")

    # Merge with existing results
    combined_results = state.get("research_results", {}).copy()
    combined_results.update(all_targeted_results)

    return {
        "research_results": combined_results,
        "source_relevance_scores": relevance_scores,
        "research_depth": 2,
        "entities": []  # No entities needed - we used planned searches instead
    }


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
        print(f"[Subresearcher] No entities to research, keeping initial results")
        return {}

    print(f"[Subresearcher] Deep research on {len(entities)} entities with parallel search strategies...")

    async def search_academic(entity: str):
        """Search academic/research sources for entity"""
        # Trust search API's ranking - no site restrictions
        academic_query = f"{entity} {main_topic} academic research papers"
        try:
            if search_api == "serper":
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    None,
                    lambda: serper_client.search(query=academic_query, max_results=max_results)
                )
                # SerperClient returns {"results": [...]} in Tavily-compatible format
                return results.get("results", [])[:max_results]
            else:
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    None,
                    lambda: tavily_client.search(query=academic_query, max_results=max_results)
                )
                return results.get("results", [])[:max_results]
        except Exception as e:
            print(f"[Subresearcher]   - Academic search failed for {entity}: {e}")
            return []

    async def search_news(entity: str):
        """Search news sources for entity"""
        # Trust search API's ranking - no site restrictions
        news_query = f"{entity} {main_topic} latest news updates"
        try:
            if search_api == "serper":
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    None,
                    lambda: serper_client.search(query=news_query, max_results=max_results)
                )
                # SerperClient returns {"results": [...]} in Tavily-compatible format
                return results.get("results", [])[:max_results]
            else:
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    None,
                    lambda: tavily_client.search(query=news_query, max_results=max_results)
                )
                return results.get("results", [])[:max_results]
        except Exception as e:
            print(f"[Subresearcher]   - News search failed for {entity}: {e}")
            return []

    async def search_social_web(entity: str):
        """Search general web and social sources for entity"""
        general_query = f"{entity} {main_topic}"
        try:
            if search_api == "serper":
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    None,
                    lambda: serper_client.search(query=general_query, max_results=max_results)
                )
                # SerperClient returns {"results": [...]} in Tavily-compatible format
                return results.get("results", [])[:max_results]
            else:
                loop = asyncio.get_event_loop()
                results = await loop.run_in_executor(
                    None,
                    lambda: tavily_client.search(query=general_query, max_results=max_results)
                )
                return results.get("results", [])[:max_results]
        except Exception as e:
            print(f"[Subresearcher]   - Social/web search failed for {entity}: {e}")
            return []

    async def research_entity_parallel(entity: str):
        """Research a single entity with parallel search strategies"""
        print(f"[Subresearcher]   - {entity}: parallel searches (academic + news + social)")

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

        print(f"[Subresearcher]   - {entity}: found {len(entity_results)} sources (academic={len(academic_results)}, news={len(news_results)}, social={len(social_results)})")
        return entity_results

    # Research all entities with rate limiting for Serper API
    # Serper has stricter rate limits, so we add delays between entities
    all_entity_results = []
    if search_api == "serper":
        # Sequential with delay to avoid 429 errors
        for entity in entities:
            result = await research_entity_parallel(entity)
            all_entity_results.append(result)
            # Small delay between entities to respect rate limits
            if len(entities) > 1:
                await asyncio.sleep(1)  # 1 second delay between entities
    else:
        # Tavily can handle parallel requests better
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

    print(f"[Subresearcher] Combined total: {len(combined_results)} sources from parallel searches")

    # Print sample of deep research findings
    if all_entity_results:
        print(f"[Subresearcher] Deep research sample findings:")
        for idx, entity_results in enumerate(all_entity_results[:2]):
            entity_name = entities[idx] if idx < len(entities) else "Unknown"
            if entity_results:
                sample_key = list(entity_results.keys())[0]
                content_preview = entity_results[sample_key][:150].replace('\n', ' ')
                print(f"    Entity '{entity_name}': {len(entity_results)} sources | Sample: {content_preview}...")

    return {
        "research_results": combined_results,
        "source_relevance_scores": combined_relevance,
        "research_depth": 2  # We went deeper
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

    # Combine all research content for analysis - use MORE sources and MORE content per source
    # Sort by relevance/credibility if available, otherwise use all sources
    all_sources = list(research_results.items())
    
    # Use up to 25 sources (instead of 10) and up to 1500 chars per source (instead of 500)
    # This gives much better coverage understanding
    num_sources_to_analyze = min(25, len(all_sources))
    combined_content = "\n\n".join([
        f"Source: {k}\n{v[:1500]}" 
        for k, v in all_sources[:num_sources_to_analyze]
    ])

    # Use LLM to analyze coverage
    CoverageAnalysis = create_model(
        'CoverageAnalysis',
        coverage_score=(float, ...),  # 0.0-1.0
        discovered_entities=(list[str], ...),  # Important entities/topics mentioned
        knowledge_gaps=(list[str], ...),  # Questions raised but not answered
        needs_deeper_research=(bool, ...),  # Should we go deeper?
        reasoning=(str, ...)
    )

    llm_structured = llm.with_structured_output(CoverageAnalysis)

    analysis_prompt = f"""Analyze research coverage and identify gaps for deeper investigation.

TOPIC: {main_topic}
SECTION: {section_title}
EXPECTED SUBTOPICS: {', '.join(section_subtopics[:5])}

RESEARCH SO FAR:
{combined_content[:8000]}

ANALYZE:
1. **Coverage Score** (0.0-1.0): How well does this research cover the expected subtopics?
   - 1.0 = Comprehensive, all subtopics well-covered with specific examples, data, and details
   - 0.8-0.9 = Good coverage, most subtopics addressed with some specifics
   - 0.6-0.7 = Adequate coverage, subtopics mentioned but lacking depth/specifics
   - 0.4-0.5 = Partial coverage, major gaps exist, missing key details/examples
   - 0.0-0.3 = Minimal coverage, mostly off-topic or very superficial
   - BE STRICT: Coverage requires SPECIFIC examples, data points, and detailed explanations, not just general statements

2. **Discovered Entities**: Extract 3-5 SPECIFIC, RESEARCHABLE entities mentioned that warrant deeper investigation:
   - Focus on: Specific studies, research papers, named methodologies, key researchers, specific techniques/exercises
   - AVOID: Generic organizations (unless central to topic), brand names, websites, unless they represent key concepts
   - Only include entities that are:
     a) Directly relevant to the main topic AND comparison
     b) Will yield NEW information when researched (not just general info)
     c) Specific enough to generate targeted search results
   - Example for "powerlifting vs Olympic lifting": ["snatch technique", "squat biomechanics", "periodization models"]
   - NOT: ["USA Weightlifting", "Westside Barbell", "Barbell Medicine"] (these are organizations, not researchable concepts)
   - Prefer: Specific techniques, studies, methodologies, or key concepts that need deeper investigation

3. **Knowledge Gaps**: Identify 2-4 specific, ACTIONABLE search queries (not questions):
   - Convert questions into search-friendly queries
   - Focus on missing data, statistics, or specific comparisons
   - Make queries specific enough to find targeted information
   - Example: Instead of "What are the exact player counts?" use "powerlifting Olympic lifting player statistics 2024"
   - Example: Instead of "How do retention rates compare?" use "powerlifting vs Olympic lifting injury rates statistics comparison"
   - Format as search queries, not questions

4. **Needs Deeper Research**: Should we do another research round?
   - TRUE if: coverage < 0.75 OR significant gaps exist OR important entities need investigation OR missing specific data/statistics
   - FALSE if: coverage >= 0.75 AND all subtopics well-addressed AND sufficient specific examples/data present
   - Be STRICT: Coverage of 0.5-0.7 means significant gaps still exist and more research is needed

5. **Reasoning**: Brief explanation (1-2 sentences)

Return structured output.
"""

    try:
        messages = [
            SystemMessage(content="You are a research coverage analyst. Identify gaps and entities for deeper investigation."),
            HumanMessage(content=analysis_prompt)
        ]

        response = await llm_structured.ainvoke(messages)

        # Limit entities to top 5 (quality over quantity)
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

    if not discovered_entities and not knowledge_gaps:
        print(f"[DEEP DIVE] No gaps or entities to research")
        return {}

    # CHECK SHARED POOL: Filter out entities already researched by other sections
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

    if new_entities:
        print(f"[DEEP DIVE] Researching {len(new_entities)} NEW entities + {len(knowledge_gaps)} gaps...")
    elif knowledge_gaps:
        print(f"[DEEP DIVE] Researching {len(knowledge_gaps)} gaps (all entities already researched)...")
    else:
        print(f"[DEEP DIVE] All entities already researched, no gaps to fill")
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
        query = f"{entity} {main_topic} comparison differences statistics"
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
        min_credibility=0.3  # Low threshold, we filter later in writing
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


async def fetch_full_content_for_top_sources(state: SubResearcherGraphState) -> dict:
    """
    ENHANCEMENT: Fetch full article content for top credible sources using Firecrawl.
    This enriches research with detailed information beyond search snippets.
    """
    enable_mcp_fetch = state.get("enable_mcp_fetch", False)
    max_mcp_fetches = state.get("max_mcp_fetches", 5)

    if not enable_mcp_fetch or max_mcp_fetches == 0:
        print(f"[Subresearcher] Firecrawl fetch disabled, skipping full content extraction")
        return {}

    research_results = state.get("research_results", {})
    credibilities = state.get("source_credibilities", {})
    relevance_scores = state.get("source_relevance_scores", {})

    if not research_results:
        return {}

    print(f"[Subresearcher] FIRECRAWL: Extracting full content for most RELEVANT sources...")

    # PRIORITY: Relevance to main topic (what user is searching for)
    # Sort by: 60% relevance + 40% credibility
    source_scores = []
    for source_key in research_results.keys():
        relevance = relevance_scores.get(source_key, 0.5)  # Search API relevance
        credibility = credibilities.get(source_key, 0.5)    # Quality/credibility

        # Combined score: prioritize relevance (what user wants) over credibility
        combined_score = (0.6 * relevance) + (0.4 * credibility)
        source_scores.append((source_key, combined_score, relevance, credibility))

    # Sort by combined score (highest first)
    sorted_sources = sorted(source_scores, key=lambda x: x[1], reverse=True)

    # Extract URLs from top RELEVANT sources
    urls_to_fetch = []
    for source_key, combined_score, relevance, credibility in sorted_sources[:max_mcp_fetches]:
        # Extract URL from source_key format "Title (URL)"
        if '(' in source_key and ')' in source_key:
            url = source_key.split('(')[1].split(')')[0]
            urls_to_fetch.append(url)
            print(f"      Top source: {source_key[:60]}... (relevance={relevance:.2f}, cred={credibility:.2f})")

    if not urls_to_fetch:
        print(f"[Subresearcher] No valid URLs found for Firecrawl fetch")
        return {}

    print(f"[Subresearcher] Fetching full content for top {len(urls_to_fetch)} sources...")

    # Fetch full content using Firecrawl
    try:
        full_content_map = await fetch_full_content(urls_to_fetch, max_urls=max_mcp_fetches)

        # Merge full content back into research_results
        # Strategy: Replace snippet with full content if available
        enhanced_results = {}
        for source_key, snippet_content in research_results.items():
            # Extract URL to check if we have full content
            if '(' in source_key and ')' in source_key:
                url = source_key.split('(')[1].split(')')[0]
                if url in full_content_map:
                    # Replace snippet with full content
                    full_content = full_content_map[url]
                    enhanced_results[source_key] = f"[FULL ARTICLE]\n{full_content}"
                    print(f"[Subresearcher]   ✓ Enhanced with full content: {source_key[:60]}...")
                else:
                    # Keep snippet
                    enhanced_results[source_key] = snippet_content
            else:
                enhanced_results[source_key] = snippet_content

        print(f"[Subresearcher] FIRECRAWL: Enhanced {len(full_content_map)}/{len(urls_to_fetch)} sources with full content")

        # Show sample of enhanced content
        if full_content_map:
            sample_url = list(full_content_map.keys())[0]
            sample_content = full_content_map[sample_url]
            char_count = len(sample_content)
            # Check if it has numerical data
            import re
            numbers_found = len(re.findall(r'\d+', sample_content[:2000]))
            print(f"[Subresearcher] Firecrawl sample: {char_count} chars, {numbers_found} numbers in first 2000 chars")

        return {
            "research_results": enhanced_results
        }

    except Exception as e:
        print(f"[Subresearcher] Firecrawl fetch failed: {e}")
        return {}


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
    Create the subresearcher graph with DYNAMIC DEEPENING.

    Flow:
    1. plan_research_strategy: Create initial research plan from outline
    2. initial_broad_search: Execute primary search
    3. execute_targeted_searches: Execute planned targeted searches
    4. assess_coverage_and_gaps: **NEW** Analyze coverage, discover entities/gaps
    5. Routing decision:
       - If coverage low OR important entities discovered → deep_dive_research (go to step 6)
       - If coverage good → assess_quality (skip to step 7)
    6. deep_dive_research: **NEW** Research discovered entities and fill gaps
       → Loop back to assess_coverage_and_gaps (iterative deepening)
    7. assess_quality: Filter and score final sources
    8. fetch_full_content: Fetch full articles for top sources

    The graph can loop through steps 4-6 multiple times until max_depth is reached
    or coverage is satisfactory.
    """
    workflow = StateGraph(SubResearcherGraphState)

    # Add nodes
    workflow.add_node("plan_research", plan_research_strategy)
    workflow.add_node("initial_broad_search", initial_broad_search)
    workflow.add_node("targeted_searches", execute_targeted_searches)
    workflow.add_node("assess_coverage", assess_coverage_and_gaps)  # NEW: Dynamic analysis
    workflow.add_node("deep_dive", deep_dive_research)              # NEW: Adaptive research
    workflow.add_node("assess_quality", assess_quality)
    workflow.add_node("fetch_full_content", fetch_full_content_for_top_sources)

    # Initial research flow
    workflow.set_entry_point("plan_research")
    workflow.add_edge("plan_research", "initial_broad_search")
    workflow.add_edge("initial_broad_search", "targeted_searches")

    # DYNAMIC DEEPENING: After targeted searches, assess coverage
    workflow.add_edge("targeted_searches", "assess_coverage")

    # Routing: Either deepen or finish
    workflow.add_conditional_edges(
        "assess_coverage",
        should_deepen_research,
        {
            "deepen": "deep_dive",      # Coverage low → research more
            "finish": "assess_quality"  # Coverage good → finish
        }
    )

    # ITERATIVE LOOP: After deep dive, re-assess coverage (may trigger another round)
    workflow.add_edge("deep_dive", "assess_coverage")

    # Final steps
    workflow.add_edge("assess_quality", "fetch_full_content")
    workflow.add_edge("fetch_full_content", END)

    return workflow.compile()


# Create the graph
subresearcher_graph = create_subresearcher_graph()