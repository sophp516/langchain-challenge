from typing import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from pydantic import create_model
from utils.model import llm
from utils.configuration import tavily_client, serper_client
from utils.verification import filter_quality_sources
from utils.mcp_fetch import fetch_full_content
import asyncio


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

    # Store initial results with relevance scores
    initial_results = {}
    relevance_scores = {}
    for idx, result in enumerate(search_results_list):
        source_url = result.get("url", "")
        source_title = result.get("title", "Untitled")
        source_content = result.get("content", "")
        search_score = result.get("score", 0.0)  # Search API relevance score

        if source_content:
            source_key = f"{source_title} ({source_url})"
            initial_results[source_key] = source_content

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
    Step 3: Execute targeted searches from research plan.

    Instead of extracting entities from results, we use the pre-planned
    targeted searches based on subtopics and key questions.
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

    print(f"[DEPTH 1→2: Targeted Searches] Executing {len(targeted_searches)} planned searches...")

    # Execute each targeted search
    all_targeted_results = {}
    relevance_scores = state.get("source_relevance_scores", {}).copy()

    for idx, search_spec in enumerate(targeted_searches):
        query = search_spec.get("query", "")
        priority = search_spec.get("priority", "medium")

        print(f"  [{priority.upper()}] {query[:80]}...")

        try:
            if search_api == "serper":
                loop = asyncio.get_event_loop()
                search_results = await loop.run_in_executor(
                    None,
                    lambda: serper_client.search(query=query, max_results=max_results)
                )
                results_list = search_results.get("results", [])[:max_results]
            else:  # tavily
                loop = asyncio.get_event_loop()
                search_results = await loop.run_in_executor(
                    None,
                    lambda: tavily_client.search(query=query, max_results=max_results)
                )
                results_list = search_results.get("results", [])[:max_results]

            # Store results
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

            print(f"    Found {len(results_list)} sources")

        except Exception as e:
            print(f"    Search failed: {e}")

    print(f"[DEPTH 1→2: Targeted Searches] Total: {len(all_targeted_results)} new sources")

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
        """Search academic sources for entity"""
        academic_query = f"{entity} {main_topic} site:edu OR site:scholar.google.com OR site:arxiv.org OR site:researchgate.net"
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
        news_query = f"{entity} {main_topic} site:bbc.com OR site:reuters.com OR site:nytimes.com OR site:theguardian.com OR site:cnn.com"
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

def create_subresearcher_graph():
    """
    Create the subresearcher graph with smart research planning + targeted searches.

    Flow:
    1. plan_research_strategy: Analyze subtopics/questions and create research plan
    2. initial_broad_search: Execute primary search from plan
    3. execute_targeted_searches: Execute targeted searches based on plan
    4. deep_research_entities: (Legacy) Research entities if extracted
    5. assess_quality: Filter and score sources
    6. fetch_full_content: Fetch full articles for top sources via Firecrawl
    """
    workflow = StateGraph(SubResearcherGraphState)

    # Add nodes
    workflow.add_node("plan_research", plan_research_strategy)
    workflow.add_node("initial_broad_search", initial_broad_search)
    workflow.add_node("targeted_searches", execute_targeted_searches)
    workflow.add_node("deep_research_entities", deep_research_entities)
    workflow.add_node("assess_quality", assess_quality)
    workflow.add_node("fetch_full_content", fetch_full_content_for_top_sources)

    # New flow with research planning
    workflow.set_entry_point("plan_research")
    workflow.add_edge("plan_research", "initial_broad_search")
    workflow.add_edge("initial_broad_search", "targeted_searches")
    workflow.add_edge("targeted_searches", "deep_research_entities")
    workflow.add_edge("deep_research_entities", "assess_quality")
    workflow.add_edge("assess_quality", "fetch_full_content")
    workflow.add_edge("fetch_full_content", END)

    return workflow.compile()


# Create the graph
subresearcher_graph = create_subresearcher_graph()