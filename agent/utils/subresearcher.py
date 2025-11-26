from typing import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from pydantic import create_model
from utils.model import llm
from utils.configuration import tavily_client, serper_client
from utils.verification import filter_quality_sources
import asyncio


class SubResearcherGraphState(TypedDict):
    """State for the subresearcher subgraph with iterative deepening"""
    subtopic_id: int
    subtopic: str  # The broad query
    main_topic: str
    other_subtopics: list[str]

    # Final outputs
    research_results: dict[str, str]  # All combined results
    source_credibilities: dict[str, float]
    research_depth: int

    # Config
    max_search_results: int
    max_research_depth: int
    search_api: str

    # Internal state for iterative deepening
    entities: list[str]  # Extracted entities to research


# ============================================================================
# ITERATIVE DEEPENING NODES
# ============================================================================

async def initial_broad_search(state: SubResearcherGraphState) -> dict:
    """
    Step 1: Do initial broad search on the query.
    This discovers what entities exist.
    """
    broad_query = state.get("subtopic", "")
    subtopic_id = state.get("subtopic_id", 0)
    search_api = state.get("search_api", "tavily")
    max_results = state.get("max_search_results", 3)

    print(f"[Subresearcher #{subtopic_id}] [DEPTH 1: Initial Search] Query: {broad_query[:60]}...")

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
        print(f"[Subresearcher] Initial search failed: {e}")
        search_results_list = []

    # Store initial results
    initial_results = {}
    for result in search_results_list:
        source_url = result.get("url", "")
        source_title = result.get("title", "Untitled")
        source_content = result.get("content", "")

        if source_content:
            source_key = f"{source_title} ({source_url})"
            initial_results[source_key] = source_content

    print(f"[Subresearcher #{subtopic_id}] [DEPTH 1: Initial Search] Found {len(initial_results)} sources")

    return {
        "research_results": initial_results,
        "research_depth": 1
    }


async def extract_entities(state: SubResearcherGraphState) -> dict:
    """
    Step 2: ADAPTIVE ENTITY/TOPIC EXTRACTION

    Analyzes initial research to:
    1. Identify key entities/topics that need deeper investigation
    2. Filter out minor/irrelevant items
    3. Prioritize what actually matters for comprehensive coverage

    Works for all topic types:
    - Type A: Extract important entities (games, songs, products, etc.)
    - Type B/C/D: Extract key themes/aspects that need more depth
    """
    research_results = state.get("research_results", {})
    main_topic = state.get("main_topic", "")
    subtopic = state.get("subtopic", "")
    max_depth = state.get("max_research_depth", 1)

    # If max_depth is 1, skip deeper research
    if max_depth <= 1:
        print(f"[Subresearcher] max_depth={max_depth}, skipping entity extraction")
        return {"entities": []}

    # Combine content from initial results
    combined_content = "\n\n".join([
        f"Source: {src}\n{content[:500]}"
        for src, content in list(research_results.items())[:5]
    ])

    if not combined_content.strip():
        print(f"[Subresearcher] No content to extract entities from")
        return {"entities": []}

    subtopic_id = state.get("subtopic_id", 0)
    print(f"[Subresearcher #{subtopic_id}] [DEPTH 1→2: Adaptive Analysis] Analyzing what needs deeper research...")

    # Use LLM to intelligently extract IMPORTANT topics/entities only
    AdaptiveExtractionOutput = create_model(
        'AdaptiveExtractionOutput',
        should_go_deeper=(bool, ...),
        reasoning=(str, ...),
        important_topics=(list[str], ...)
    )

    adaptive_llm = llm.with_structured_output(AdaptiveExtractionOutput)

    adaptive_prompt = f"""
    Analyze the research results and decide if deeper investigation is needed.

    Query: {subtopic}
    Main Topic: {main_topic}

    Initial Research Results:
    {combined_content[:2500]}

    **YOUR TASK:**
    1. Determine if current research is SUFFICIENT to comprehensively answer the query
    2. If NOT sufficient, identify 3-5 IMPORTANT topics/entities that need deeper research
    3. Filter out minor/irrelevant items - only select topics that significantly impact the answer

    **DECISION CRITERIA:**
    - Go deeper IF: Results mention multiple important entities/themes but lack detail on each
    - Go deeper IF: Critical aspects are mentioned but not explained
    - STOP IF: Current results already provide comprehensive coverage
    - STOP IF: Only generic/shallow information exists and going deeper won't help

    **EXTRACTION GUIDELINES:**
    - Type A (Entity lists like "best games 2024"): Extract specific entities (game names, product names, people)
    - Type B/C/D (Thematic/analytical): Extract key themes/aspects (e.g., "training methodology", "risk factors")
    - Focus on topics that matter for comprehensive coverage
    - Skip trivial/redundant topics

    Return:
    - should_go_deeper: true/false
    - reasoning: Why you made this decision (one sentence)
    - important_topics: List of 3-5 topics to research deeper (empty if should_go_deeper=false)
    """

    try:
        response = await adaptive_llm.ainvoke([
            SystemMessage(content="You are a research strategist who decides what needs deeper investigation."),
            HumanMessage(content=adaptive_prompt)
        ])

        should_go_deeper = response.should_go_deeper
        reasoning = response.reasoning
        important_topics = response.important_topics[:5]  # Max 5 topics

        print(f"[Subresearcher] Decision: {'GO DEEPER' if should_go_deeper else 'SUFFICIENT'}")
        print(f"[Subresearcher] Reasoning: {reasoning}")
        if should_go_deeper:
            print(f"[Subresearcher] Topics to research: {important_topics}")

        return {"entities": important_topics if should_go_deeper else []}

    except Exception as e:
        print(f"[Subresearcher] Adaptive extraction failed: {e}")
        # Fallback to simple entity extraction
        return {"entities": []}


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

    # Combine all results (initial + entity research)
    combined_results = {**initial_results}
    for entity_results in all_entity_results:
        combined_results.update(entity_results)

    print(f"[Subresearcher] Combined total: {len(combined_results)} sources from parallel searches")

    return {
        "research_results": combined_results,
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

    return {
        "research_results": filtered_results,
        "source_credibilities": credibilities
    }


# ============================================================================
# GRAPH CREATION
# ============================================================================

def create_subresearcher_graph():
    """
    Create the subresearcher graph with iterative deepening.

    Flow:
    1. initial_broad_search: Search the broad query
    2. extract_entities: Extract entities from results
    3. deep_research_entities: Research each entity
    4. assess_quality: Filter and score sources
    """
    workflow = StateGraph(SubResearcherGraphState)

    # Add nodes
    workflow.add_node("initial_broad_search", initial_broad_search)
    workflow.add_node("extract_entities", extract_entities)
    workflow.add_node("deep_research_entities", deep_research_entities)
    workflow.add_node("assess_quality", assess_quality)

    # Linear flow
    workflow.set_entry_point("initial_broad_search")
    workflow.add_edge("initial_broad_search", "extract_entities")
    workflow.add_edge("extract_entities", "deep_research_entities")
    workflow.add_edge("deep_research_entities", "assess_quality")
    workflow.add_edge("assess_quality", END)

    return workflow.compile()


# Create the graph
subresearcher_graph = create_subresearcher_graph()