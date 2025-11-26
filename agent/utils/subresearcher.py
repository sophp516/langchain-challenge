from typing import TypedDict, Literal
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from utils.model import llm, llm_quality
from utils.configuration import tavily_client, serper_client
from utils.verification import filter_quality_sources
import asyncio
from functools import partial
from urllib.parse import urlparse
from collections import Counter


class SubResearcherGraphState(TypedDict):
    """State for the subresearcher subgraph"""
    subtopic_id: int
    subtopic: str
    main_topic: str  # The parent topic for context in searches
    other_subtopics: list[str]  # Other subtopics being researched (for scope context)
    research_results: dict[str, str]
    research_depth: int  # Current depth layer (1, 2, or 3)
    source_credibilities: dict[str, float]  # Track credibility scores
    follow_up_queries: list[str]  # Queries for deeper research
    # Config values from AgentConfig
    max_search_results: int  # From frontend config
    max_research_depth: int  # From frontend config
    search_api: str  # From frontend config (tavily/serper)
    # New fields for enhanced features
    quality_assessment: str  # "excellent" | "good" | "poor"
    academic_results: dict[str, str]  # Results from academic search
    news_results: dict[str, str]  # Results from news search
    social_results: dict[str, str]  # Results from social/web search
    parallel_search_complete: bool  # Track if parallel searches are done


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def extract_domain(source_key: str) -> str:
    """Extract domain from source key like 'Title (https://example.com)'"""
    try:
        if '(' in source_key and ')' in source_key:
            url = source_key.split('(')[1].split(')')[0]
            parsed = urlparse(url)
            return parsed.netloc.replace('www.', '')
        return source_key
    except:
        return source_key


def classify_source_types(sources: list[str]) -> set[str]:
    """Classify sources into types: academic, news, blog, social, etc."""
    types = set()
    for source in sources:
        domain = extract_domain(source).lower()

        if any(x in domain for x in ['edu', 'scholar', 'researchgate', 'arxiv', 'pubmed']):
            types.add('academic')
        elif any(x in domain for x in ['bbc', 'cnn', 'reuters', 'nytimes', 'guardian', 'news']):
            types.add('news')
        elif any(x in domain for x in ['twitter', 'reddit', 'facebook', 'instagram', 'tiktok']):
            types.add('social')
        elif any(x in domain for x in ['gov', '.org']):
            types.add('official')
        else:
            types.add('web')

    return types


async def process_source(
    result: dict,
    subtopic: str,
    main_topic: str,
    credibility_score: float
) -> tuple[str, str, float]:
    """
    Process a single source and return (source_key, findings, credibility) tuple
    Enhanced with credibility tracking and main topic context
    Uses cheaper llm model instead of llm_quality for 25-30% cost savings
    """
    source_url = result.get("url", "")
    source_title = result.get("title", "Untitled Source")
    source_content = result.get("content", "")

    summary_prompt = f"""
    Extract specific, concrete information from this source relevant to the research.

    Main Research Topic: {main_topic}
    Current Subtopic: {subtopic}

    Source Title: {source_title}
    Source URL: {source_url}

    Content:
    {source_content}

    PRIORITY EXTRACTION (include ALL that appear in the source):
    - Specific NAMES (people, groups, artists, companies, products)
    - Specific TITLES (songs, albums, movies, books, articles)
    - RANKINGS and LISTS (top 10, #1, most popular, best-selling)
    - NUMBERS and STATISTICS (sales figures, chart positions, dates, percentages)
    - Direct QUOTES with attribution

    Format your response as a list of specific facts found in this source.
    Do NOT generalize or summarize - extract the actual names, titles, and numbers.
    If the source lists "top songs" or "most popular X", list EVERY item mentioned.

    Only include information relevant to "{main_topic}".
    """

    messages = [
        SystemMessage(content="You are a precise research assistant that extracts specific names, titles, rankings, and numbers from sources. Never generalize - always include the exact names and figures mentioned. If a source ranks items or lists 'top X', extract every item in that list."),
        HumanMessage(content=summary_prompt)
    ]

    # COST OPTIMIZATION: Use cheaper llm instead of llm_quality (25-30% savings)
    summary_response = await llm.ainvoke(messages)
    findings = summary_response.content if hasattr(summary_response, 'content') else str(summary_response)

    # Create a source identifier with credibility indicator
    if source_url:
        source_key = f"{source_title} ({source_url})"
    else:
        source_key = source_title

    return source_key, findings, credibility_score


# ============================================================================
# FEATURE 1: STREAMING PROGRESS NODES
# ============================================================================

async def stream_initial_progress(state: SubResearcherGraphState) -> dict:
    """Stream progress message after initial research completes"""
    subtopic = state["subtopic"]
    num_sources = len(state.get("research_results", {}))

    message = f"✓ Found {num_sources} initial sources for: {subtopic}"
    print(f"[Streaming] {message}")

    return {}


async def stream_quality_assessment(state: SubResearcherGraphState) -> dict:
    """Stream quality assessment result"""
    subtopic = state["subtopic"]
    quality = state.get("quality_assessment", "good")
    num_sources = len(state.get("research_results", {}))
    avg_credibility = sum(state.get("source_credibilities", {}).values()) / max(len(state.get("source_credibilities", {})), 1)

    quality_emoji = {
        "excellent": "🌟",
        "good": "✓",
        "poor": "⚠️"
    }.get(quality, "✓")

    message = f"{quality_emoji} Quality: {quality.upper()} | {num_sources} sources | Avg credibility: {avg_credibility:.2f}"
    print(f"[Streaming] {message}")

    return { }


async def stream_deep_dive_progress(state: SubResearcherGraphState) -> dict:
    """Stream progress during deep dive research"""
    subtopic = state["subtopic"]
    depth = state.get("research_depth", 1)
    num_sources = len(state.get("research_results", {}))

    message = f"🔍 Layer {depth} complete for: {subtopic} | Total: {num_sources} sources"
    print(f"[Streaming] {message}")

    return { }


# ============================================================================
# FEATURE 2: ADAPTIVE QUALITY ASSESSMENT & BRANCHING
# ============================================================================

async def assess_research_quality(state: SubResearcherGraphState) -> dict:
    """
    Evaluate research quality and determine next steps
    Returns quality_assessment: "excellent" | "good" | "poor"
    """
    results = state.get("research_results", {})
    credibilities = state.get("source_credibilities", {})
    subtopic = state["subtopic"]

    if not results:
        print(f"[Quality Check] No results found - marking as POOR")
        return {"quality_assessment": "poor"}

    # Calculate metrics
    num_sources = len(results)
    avg_credibility = sum(credibilities.values()) / len(credibilities) if credibilities else 0.5

    # Check source diversity
    domains = [extract_domain(src) for src in results.keys()]
    unique_domains = len(set(domains))
    domain_diversity = unique_domains / max(num_sources, 1)

    print(f"[Quality Check] {subtopic}: sources={num_sources}, avg_cred={avg_credibility:.2f}, diversity={domain_diversity:.2f}")

    # Quality assessment logic
    if avg_credibility > 0.75 and num_sources >= 4 and domain_diversity > 0.6:
        assessment = "excellent"
    elif avg_credibility < 0.5 or num_sources < 2 or domain_diversity < 0.3:
        assessment = "poor"
    else:
        assessment = "good"

    print(f"[Quality Check] Assessment: {assessment.upper()}")

    return {"quality_assessment": assessment}


def route_by_quality(state: SubResearcherGraphState) -> str:
    """
    Route based on quality assessment:
    - excellent: End early (cost savings!)
    - good: Continue normal flow
    - poor: Try alternative search strategy
    """
    quality = state.get("quality_assessment", "good")
    current_depth = state.get("research_depth", 1)
    max_depth = state.get("max_research_depth", 2)

    # If excellent quality at Layer 1, we can skip deep dive
    if quality == "excellent" and current_depth == 1:
        print(f"[Routing] EXCELLENT quality - ending early to save costs")
        return "end"

    # If poor quality, try alternative strategy
    if quality == "poor":
        print(f"[Routing] POOR quality - trying alternative search")
        return "alternative_strategy"

    # Good quality - continue normal analysis
    if current_depth < max_depth:
        print(f"[Routing] GOOD quality - continuing to analysis")
        return "analyze"

    print(f"[Routing] Max depth reached - ending")
    return "end"


async def alternative_search_strategy(state: SubResearcherGraphState) -> dict:
    """
    When initial research has poor quality, try alternative approach:
    - Rephrase query
    - Search with broader terms
    - Try different time ranges
    """
    subtopic = state["subtopic"]
    main_topic = state.get("main_topic", "")
    search_api = state.get("search_api", "tavily")
    max_search_results = state.get("max_search_results", 3)

    print(f"[Alternative Strategy] Rephrasing query for: {subtopic}")

    # Try broader query
    alternative_query = f"{main_topic} overview {subtopic.split()[-1]}"

    # Select search client
    if search_api == "serper" and serper_client:
        search_func = partial(serper_client.search, query=alternative_query, max_results=max_search_results)
    else:
        search_func = partial(tavily_client.search, query=alternative_query, search_depth="basic", max_results=max_search_results)

    search_response = await asyncio.to_thread(search_func)
    results = search_response.get("results", [])
    filtered_results = await filter_quality_sources(results, min_credibility=0.3)  # Lower threshold

    # Process new sources
    tasks = [process_source(result, subtopic, main_topic, credibility) for result, credibility in filtered_results[:3]]
    source_data = await asyncio.gather(*tasks)

    # Merge with existing
    existing_results = state.get("research_results", {})
    existing_credibilities = state.get("source_credibilities", {})

    for source_key, findings, credibility in source_data:
        if source_key not in existing_results:
            existing_results[source_key] = findings
            existing_credibilities[source_key] = credibility

    print(f"[Alternative Strategy] Added {len(source_data)} sources, total: {len(existing_results)}")

    return {
        "research_results": existing_results,
        "source_credibilities": existing_credibilities
    }


# ============================================================================
# FEATURE 3: PARALLEL SEARCH STRATEGIES
# ============================================================================

async def search_academic_sources(state: SubResearcherGraphState) -> dict:
    """
    Search academic/scholarly sources in parallel
    Targets: .edu, scholar.google, arxiv, researchgate, etc.
    """
    subtopic = state["subtopic"]
    main_topic = state.get("main_topic", "")
    search_api = state.get("search_api", "tavily")
    max_search_results = state.get("max_search_results", 3)

    # Academic-focused query
    query = f"{main_topic} {subtopic} research paper study analysis"

    print(f"[Academic Search] {query[:60]}...")

    # Tavily supports domain filtering
    if search_api == "tavily":
        search_func = partial(
            tavily_client.search,
            query=query,
            search_depth="advanced",
            max_results=max_search_results,
            include_domains=["edu", "scholar.google.com", "researchgate.net", "arxiv.org"]
        )
    else:
        # Serper doesn't support domain filtering, add to query
        query = f"site:edu OR site:scholar.google.com {query}"
        search_func = partial(serper_client.search, query=query, max_results=max_search_results) if serper_client else None

    if not search_func:
        return {"academic_results": {}}

    try:
        search_response = await asyncio.to_thread(search_func)
        results = search_response.get("results", [])
        filtered_results = await filter_quality_sources(results, min_credibility=0.5)

        # Process sources
        tasks = [process_source(result, subtopic, main_topic, credibility) for result, credibility in filtered_results[:2]]
        source_data = await asyncio.gather(*tasks)

        academic_results = {}
        for source_key, findings, credibility in source_data:
            academic_results[source_key] = findings

        print(f"[Academic Search] Found {len(academic_results)} sources")
        return {"academic_results": academic_results}

    except Exception as e:
        print(f"[Academic Search] Error: {e}")
        return {"academic_results": {}}


async def search_news_sources(state: SubResearcherGraphState) -> dict:
    """
    Search news sources in parallel
    Targets: bbc, reuters, nytimes, etc.
    """
    subtopic = state["subtopic"]
    main_topic = state.get("main_topic", "")
    search_api = state.get("search_api", "tavily")
    max_search_results = state.get("max_search_results", 3)

    # News-focused query with recency
    query = f"{main_topic} {subtopic} news latest 2024 2025"

    print(f"[News Search] {query[:60]}...")

    # Tavily supports domain filtering
    if search_api == "tavily":
        search_func = partial(
            tavily_client.search,
            query=query,
            search_depth="advanced",
            max_results=max_search_results,
            include_domains=["bbc.com", "reuters.com", "nytimes.com", "theguardian.com", "apnews.com"]
        )
    else:
        query = f"site:bbc.com OR site:reuters.com {query}"
        search_func = partial(serper_client.search, query=query, max_results=max_search_results) if serper_client else None

    if not search_func:
        return {"news_results": {}}

    try:
        search_response = await asyncio.to_thread(search_func)
        results = search_response.get("results", [])
        filtered_results = await filter_quality_sources(results, min_credibility=0.5)

        # Process sources
        tasks = [process_source(result, subtopic, main_topic, credibility) for result, credibility in filtered_results[:2]]
        source_data = await asyncio.gather(*tasks)

        news_results = {}
        for source_key, findings, credibility in source_data:
            news_results[source_key] = findings

        print(f"[News Search] Found {len(news_results)} sources")
        return {"news_results": news_results}

    except Exception as e:
        print(f"[News Search] Error: {e}")
        return {"news_results": {}}


async def search_social_web_sources(state: SubResearcherGraphState) -> dict:
    """
    Search social media and general web sources in parallel
    Targets: reddit, twitter, blogs, forums, etc.
    """
    subtopic = state["subtopic"]
    main_topic = state.get("main_topic", "")
    search_api = state.get("search_api", "tavily")
    max_search_results = state.get("max_search_results", 3)

    # Social/web-focused query
    query = f"{main_topic} {subtopic} discussion trending popular"

    print(f"[Social/Web Search] {query[:60]}...")

    # General web search (default behavior)
    if search_api == "serper" and serper_client:
        search_func = partial(serper_client.search, query=query, max_results=max_search_results)
    else:
        search_func = partial(tavily_client.search, query=query, search_depth="basic", max_results=max_search_results)

    try:
        search_response = await asyncio.to_thread(search_func)
        results = search_response.get("results", [])
        filtered_results = await filter_quality_sources(results, min_credibility=0.4)

        # Process sources
        tasks = [process_source(result, subtopic, main_topic, credibility) for result, credibility in filtered_results[:2]]
        source_data = await asyncio.gather(*tasks)

        social_results = {}
        for source_key, findings, credibility in source_data:
            social_results[source_key] = findings

        print(f"[Social/Web Search] Found {len(social_results)} sources")
        return {"social_results": social_results}

    except Exception as e:
        print(f"[Social/Web Search] Error: {e}")
        return {"social_results": {}}


async def merge_parallel_results(state: SubResearcherGraphState) -> dict:
    """
    Merge results from parallel search strategies
    Combines academic, news, and social/web results
    """
    academic = state.get("academic_results", {})
    news = state.get("news_results", {})
    social = state.get("social_results", {})

    # Merge all results
    merged_results = {}
    merged_credibilities = {}

    # Academic sources get slight credibility boost
    for source, findings in academic.items():
        merged_results[source] = findings
        merged_credibilities[source] = 0.85

    # News sources
    for source, findings in news.items():
        if source not in merged_results:
            merged_results[source] = findings
            merged_credibilities[source] = 0.80

    # Social/web sources
    for source, findings in social.items():
        if source not in merged_results:
            merged_results[source] = findings
            merged_credibilities[source] = 0.70

    total_sources = len(merged_results)
    print(f"[Merge] Combined {len(academic)} academic + {len(news)} news + {len(social)} social = {total_sources} total sources")

    # Check diversity
    source_types = classify_source_types(list(merged_results.keys()))
    print(f"[Merge] Source diversity: {source_types}")

    return {
        "research_results": merged_results,
        "source_credibilities": merged_credibilities,
        "parallel_search_complete": True,
        "research_depth": 1
    }


# ============================================================================
# ORIGINAL NODES (kept for deep dive functionality)
# ============================================================================

async def analyze_and_generate_follow_ups(state: SubResearcherGraphState) -> SubResearcherGraphState:
    """
    Analyze current research to identify gaps and generate follow-up queries
    for deeper research layers
    """
    subtopic = state["subtopic"]
    main_topic = state.get("main_topic", "")
    other_subtopics = state.get("other_subtopics", [])
    research_results = state.get("research_results", {})
    current_depth = state.get("research_depth", 1)
    max_research_depth = state.get("max_research_depth", 2)

    if current_depth >= max_research_depth:
        print(f"[Layer {current_depth}] Max depth reached ({max_research_depth}), no follow-ups needed")
        return {"follow_up_queries": []}

    # Build summary of current findings
    findings_summary = ""
    for i, (source, findings) in enumerate(list(research_results.items())[:5]):
        findings_summary += f"Source {i+1}: {findings[:200]}...\n\n"

    analysis_prompt = f"""
    You are analyzing research on the subtopic: {subtopic}
    Main research topic: {main_topic}

    Current research findings ({len(research_results)} sources):
    {findings_summary}

    This is research layer {current_depth} of {max_research_depth}.

    Analyze the findings and identify 1-2 specific follow-up queries that would:
    1. Fill critical gaps in coverage
    2. Verify important claims
    3. Find more recent information

    Return ONLY the queries, one per line, without numbering or bullets.
    If findings are comprehensive, respond with "COMPREHENSIVE".
    """

    messages = [
        SystemMessage(content="You are a research analyst that identifies gaps and generates follow-up research queries."),
        HumanMessage(content=analysis_prompt)
    ]

    response = await llm.ainvoke(messages)
    response_text = response.content if hasattr(response, 'content') else str(response)

    if "COMPREHENSIVE" in response_text.upper():
        print(f"[Layer {current_depth}] Research deemed comprehensive, no follow-ups needed")
        return {"follow_up_queries": []}

    # Parse follow-up queries
    follow_up_queries = [
        line.strip() for line in response_text.strip().split('\n')
        if line.strip() and not line.strip().startswith(('#', '-', '*'))
    ]

    print(f"[Layer {current_depth}] Generated {len(follow_up_queries)} follow-up queries")
    return {"follow_up_queries": follow_up_queries}


async def conduct_deep_dive_research(state: SubResearcherGraphState) -> SubResearcherGraphState:
    """
    Layers 2+: Conduct deeper research based on follow-up queries
    """
    follow_up_queries = state.get("follow_up_queries", [])
    subtopic = state["subtopic"]
    main_topic = state.get("main_topic", "")
    current_depth = state.get("research_depth", 1)
    max_search_results = state.get("max_search_results", 3)
    search_api = state.get("search_api", "tavily")

    if not follow_up_queries:
        print(f"[Layer {current_depth}] No follow-up queries, skipping deep dive")
        return {}

    new_depth = current_depth + 1
    print(f"[Layer {new_depth}] Conducting deep dive with {len(follow_up_queries)} queries")

    existing_results = state.get("research_results", {})
    existing_credibilities = state.get("source_credibilities", {})

    # Process each follow-up query
    for query in follow_up_queries[:2]:  # Limit to 2 queries per layer
        print(f"[Layer {new_depth}] Searching: {query}")

        # Select search client
        if search_api == "serper" and serper_client:
            search_func = partial(serper_client.search, query=query, max_results=max_search_results)
        else:
            search_func = partial(tavily_client.search, query=query, search_depth="advanced", max_results=max_search_results)

        search_response = await asyncio.to_thread(search_func)
        results = search_response.get("results", [])
        filtered_results = await filter_quality_sources(results, min_credibility=0.4)

        # Process sources
        tasks = [process_source(result, subtopic, main_topic, credibility) for result, credibility in filtered_results[:3]]
        source_data = await asyncio.gather(*tasks)

        # Integrate new findings
        for source_key, findings, credibility in source_data:
            if source_key not in existing_results:
                existing_results[source_key] = findings
                existing_credibilities[source_key] = credibility

    print(f"[Layer {new_depth}] Deep dive complete, total sources: {len(existing_results)}")

    return {
        "research_results": existing_results,
        "source_credibilities": existing_credibilities,
        "research_depth": new_depth,
        "follow_up_queries": []
    }


def should_continue_deep_dive(state: SubResearcherGraphState) -> str:
    """Determine if we should continue with deeper research"""
    follow_up_queries = state.get("follow_up_queries", [])
    current_depth = state.get("research_depth", 1)
    max_research_depth = state.get("max_research_depth", 2)

    if follow_up_queries and current_depth < max_research_depth:
        print(f"should_continue_deep_dive: YES (depth={current_depth}/{max_research_depth}, queries={len(follow_up_queries)})")
        return "continue"

    print(f"should_continue_deep_dive: NO (depth={current_depth}/{max_research_depth}, queries={len(follow_up_queries)})")
    return "end"


# ============================================================================
# ENHANCED GRAPH CONSTRUCTION
# ============================================================================

def create_subresearcher_graph():
    """
    Create enhanced subresearcher subgraph with:
    1. Streaming progress updates
    2. Adaptive quality branching
    3. Parallel search strategies

    Flow:
    1. Parallel searches (academic + news + social) → merge
    2. Stream progress
    3. Assess quality → route by quality
       - Excellent: End early (save costs)
       - Poor: Try alternative strategy → reassess
       - Good: Analyze for follow-ups
    4. If follow-ups exist: Deep dive → stream progress → analyze again
    5. End when max depth or excellent quality
    """
    workflow = StateGraph(SubResearcherGraphState)

    # === PARALLEL SEARCH NODES ===
    workflow.add_node("search_academic", search_academic_sources)
    workflow.add_node("search_news", search_news_sources)
    workflow.add_node("search_social", search_social_web_sources)
    workflow.add_node("merge_results", merge_parallel_results)

    # === STREAMING NODES ===
    workflow.add_node("stream_initial", stream_initial_progress)
    workflow.add_node("stream_quality", stream_quality_assessment)
    workflow.add_node("stream_deep_dive", stream_deep_dive_progress)

    # === QUALITY ASSESSMENT NODES ===
    workflow.add_node("assess_quality", assess_research_quality)
    workflow.add_node("alternative_strategy", alternative_search_strategy)

    # === ORIGINAL NODES ===
    workflow.add_node("analyze_research", analyze_and_generate_follow_ups)
    workflow.add_node("deep_dive", conduct_deep_dive_research)

    # === GRAPH STRUCTURE ===

    # Entry: Start with parallel searches
    workflow.set_entry_point("search_academic")
    workflow.set_entry_point("search_news")
    workflow.set_entry_point("search_social")

    # All parallel searches converge to merge
    workflow.add_edge("search_academic", "merge_results")
    workflow.add_edge("search_news", "merge_results")
    workflow.add_edge("search_social", "merge_results")

    # After merge: Stream progress
    workflow.add_edge("merge_results", "stream_initial")

    # After streaming: Assess quality
    workflow.add_edge("stream_initial", "assess_quality")

    # After assessment: Stream quality + route by quality
    workflow.add_edge("assess_quality", "stream_quality")
    workflow.add_conditional_edges(
        "stream_quality",
        route_by_quality,
        {
            "end": END,  # Excellent quality - stop early
            "alternative_strategy": "alternative_strategy",  # Poor quality - try alternative
            "analyze": "analyze_research"  # Good quality - continue
        }
    )

    # Alternative strategy loops back to quality check
    workflow.add_edge("alternative_strategy", "assess_quality")

    # Analysis leads to conditional deep dive
    workflow.add_conditional_edges(
        "analyze_research",
        should_continue_deep_dive,
        {
            "continue": "deep_dive",
            "end": END
        }
    )

    # After deep dive: Stream progress → analyze again
    workflow.add_edge("deep_dive", "stream_deep_dive")
    workflow.add_edge("stream_deep_dive", "analyze_research")

    return workflow.compile()


subresearcher_graph = create_subresearcher_graph()
