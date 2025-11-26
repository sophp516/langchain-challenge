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
    search_api = state.get("search_api", "tavily")
    max_results = state.get("max_search_results", 3)

    print(f"[Subresearcher] Initial broad search: {broad_query[:60]}...")

    # Single search on the broad query
    try:
        if search_api == "serper":
            loop = asyncio.get_event_loop()
            search_results = await loop.run_in_executor(
                None,
                lambda: serper_client.search(query=broad_query, num_results=max_results)
            )
            results = search_results.get("organic", [])[:max_results]
            search_results_list = [
                {"url": r.get("link", ""), "title": r.get("title", ""), "content": r.get("snippet", "")}
                for r in results
            ]
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

    print(f"[Subresearcher] Found {len(initial_results)} sources from initial search")

    return {
        "research_results": initial_results,
        "research_depth": 1
    }


async def extract_entities(state: SubResearcherGraphState) -> dict:
    """
    Step 2: Extract entities (games, products, people, etc.) from initial results.
    These entities will be researched in depth.
    """
    research_results = state.get("research_results", {})
    main_topic = state.get("main_topic", "")
    max_depth = state.get("max_research_depth", 1)

    # If max_depth is 1, skip entity extraction (no deep dive)
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

    print(f"[Subresearcher] Extracting entities from initial results...")

    # Use LLM to extract entities
    EntitiesOutput = create_model(
        'EntitiesOutput',
        entities=(list[str], ...)
    )

    entity_llm = llm.with_structured_output(EntitiesOutput)

    entity_prompt = f"""
    Extract specific entities (names, titles, products) from the research results.

    Main Topic: {main_topic}

    Research Results:
    {combined_content[:2000]}

    Extract ALL entities mentioned:
    - For games: game titles (e.g., "Ashes of Creation", "Throne and Liberty")
    - For songs: song titles and artists (e.g., "APT by Rosé")
    - For products: product names (e.g., "iPhone 15", "Galaxy S24")
    - For people: person names (e.g., "Sam Altman")
    - For companies: company names (e.g., "OpenAI", "Google")

    Return ONLY the specific entity names, one per line.
    Extract up to 5 most relevant entities.
    """

    try:
        response = await entity_llm.ainvoke([
            SystemMessage(content="Extract specific entity names from research results. Return only names."),
            HumanMessage(content=entity_prompt)
        ])

        entities = response.entities[:5]  # Limit to 5 entities
        print(f"[Subresearcher] Extracted {len(entities)} entities: {entities}")

        return {"entities": entities}

    except Exception as e:
        print(f"[Subresearcher] Entity extraction failed: {e}")
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
                    lambda: serper_client.search(query=academic_query, num_results=max_results)
                )
                results = results.get("organic", [])[:max_results]
                return [
                    {"url": r.get("link", ""), "title": r.get("title", ""), "content": r.get("snippet", "")}
                    for r in results
                ]
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
                    lambda: serper_client.search(query=news_query, num_results=max_results)
                )
                results = results.get("organic", [])[:max_results]
                return [
                    {"url": r.get("link", ""), "title": r.get("title", ""), "content": r.get("snippet", "")}
                    for r in results
                ]
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
                    lambda: serper_client.search(query=general_query, num_results=max_results)
                )
                results = results.get("organic", [])[:max_results]
                return [
                    {"url": r.get("link", ""), "title": r.get("title", ""), "content": r.get("snippet", "")}
                    for r in results
                ]
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

    # Research all entities in parallel (each with their own parallel searches)
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