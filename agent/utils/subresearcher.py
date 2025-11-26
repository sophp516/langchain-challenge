from typing import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from utils.model import llm, llm_quality
from utils.configuration import tavily_client, serper_client
from utils.verification import filter_quality_sources
import asyncio
from functools import partial


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


async def process_source(
    result: dict,
    subtopic: str,
    main_topic: str,
    credibility_score: float
) -> tuple[str, str, float]:
    """
    Process a single source and return (source_key, findings, credibility) tuple
    Enhanced with credibility tracking and main topic context
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

    summary_response = await llm.ainvoke(messages)
    findings = summary_response.content if hasattr(summary_response, 'content') else str(summary_response)

    # Create a source identifier with credibility indicator
    if source_url:
        source_key = f"{source_title} ({source_url})"
    else:
        source_key = source_title

    return source_key, findings, credibility_score


async def conduct_initial_research(state: SubResearcherGraphState) -> SubResearcherGraphState:
    """
    Layer 1: Conduct initial broad research on the subtopic
    - Search for top sources
    - Filter by quality/credibility
    - Extract and summarize findings
    """
    subtopic = state["subtopic"]
    main_topic = state.get("main_topic", "")
    current_depth = state.get("research_depth", 1)
    max_search_results = state.get("max_search_results", 5)
    search_api = state.get("search_api", "tavily")

    print(f"[Layer {current_depth}] Starting research on: {subtopic} (api={search_api}, max_results={max_search_results})")

    # Enhance query if needed
    search_query = subtopic
    if main_topic and len(subtopic.split()) < 5:
        # Subtopic is short/generic, prepend main topic context
        search_query = f"{main_topic}: {subtopic}"

    # Select search client based on API
    if search_api == "serper" and serper_client:
        search_func = partial(
            serper_client.search,
            query=search_query,
            max_results=max_search_results
        )
    elif search_api == "tavily":
        search_func = partial(
            tavily_client.search,
            query=search_query,
            search_depth="advanced",
            max_results=max_search_results
        )
    else:
        print(f"[WARNING] Search API '{search_api}' not available, falling back to Tavily")
        search_func = partial(
            tavily_client.search,
            query=search_query,
            search_depth="advanced",
            max_results=max_search_results
        )

    search_response = await asyncio.to_thread(search_func)

    # Filter sources by quality
    results = search_response.get("results", [])
    filtered_results = await filter_quality_sources(results, min_credibility=0.4)

    print(f"[Layer {current_depth}] Found {len(results)} sources, {len(filtered_results)} passed quality filter")

    # Take top 5 by credibility
    top_sources = filtered_results[:5]

    # Process sources in parallel
    tasks = [
        process_source(result, subtopic, main_topic, credibility)
        for result, credibility in top_sources
    ]
    source_data = await asyncio.gather(*tasks)

    # Build research results and credibility mapping
    existing_results = state.get("research_results", {})
    existing_credibilities = state.get("source_credibilities", {})

    research_results = {**existing_results}
    source_credibilities = {**existing_credibilities}

    for source_key, findings, credibility in source_data:
        research_results[source_key] = findings
        source_credibilities[source_key] = credibility

    print(f"[Layer {current_depth}] Processed {len(source_data)} sources, total sources: {len(research_results)}")

    return {
        "subtopic_id": state.get("subtopic_id", 0),
        "subtopic": subtopic,
        "research_results": research_results,
        "source_credibilities": source_credibilities,
        "research_depth": current_depth
    }


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

    # IMPORTANT: Only skip follow-ups if we've COMPLETED at least one deep dive
    # This ensures we always do minimum 2 layers of research
    if current_depth >= max_research_depth:  # Max depth reached
        print(f"[Layer {current_depth}] Max depth reached ({max_research_depth}), no follow-ups needed")
        return {"follow_up_queries": []}

    # Build summary of current findings
    findings_summary = ""
    for i, (source, findings) in enumerate(list(research_results.items())[:5]):
        findings_summary += f"Source {i+1}: {findings[:200]}...\n\n"

    # Build context about other subtopics being researched
    other_subtopics_context = ""
    if other_subtopics:
        other_areas = [st for st in other_subtopics if st != subtopic]
        if other_areas:
            other_subtopics_context = f"""
    NOTE: Other researchers are covering these related areas:
    {chr(10).join(f'- {st}' for st in other_areas[:5])}

    Focus your follow-up queries on aspects unique to YOUR subtopic that won't overlap with the above.
    """

    # For Layer 1, be more aggressive about generating follow-ups to ensure deep research
    layer_1_instruction = ""
    if current_depth == 1:
        layer_1_instruction = """
    CRITICAL: This is Layer 1 (initial research). You MUST generate 2-3 follow-up queries to ensure thorough research.
    Do NOT mark as "COMPREHENSIVE" at Layer 1 - initial research always needs deeper investigation.
    Focus on specific aspects that need more detail, verification, or recent updates.
    """

    analysis_prompt = f"""
    You are analyzing research on the subtopic: {subtopic}
    Main research topic: {main_topic}

    Current research findings:
    {findings_summary}
    {other_subtopics_context}
    This is research layer {current_depth} of {max_research_depth}.
    {layer_1_instruction}

    Analyze the findings and identify 2-3 specific follow-up queries that would:
    1. Fill gaps in coverage THAT IS DIRECTLY RELEVANT TO THE MAIN TOPIC AND SUBTOPIC
    2. Explore promising areas in more depth (still needs to be directly relevant)
    3. Clarify ambiguous or interesting points
    4. Find more recent or specialized information
    5. Verify or expand on key claims from Layer 1 findings

    Each query should be specific and different from the original subtopic query.
    IMPORTANT: Do NOT generate queries that overlap with other subtopics being researched.

    Return ONLY the queries, one per line, without numbering or bullets.
    Only mark as "COMPREHENSIVE" if this is Layer {max_research_depth} or higher.
    """

    messages = [
        SystemMessage(content="You are a research analyst that identifies gaps and generates follow-up research queries. You avoid duplicating work being done by other researchers."),
        HumanMessage(content=analysis_prompt)
    ]

    response = await llm.ainvoke(messages)
    response_text = response.content if hasattr(response, 'content') else str(response)

    # Check if marked as comprehensive
    if "COMPREHENSIVE" in response_text.upper():
        # SAFEGUARD: Never end at Layer 1 - always do at least 2 layers
        if current_depth == 1:
            print(f"[Layer {current_depth}] LLM marked as comprehensive, but Layer 1 must continue. Generating default follow-ups.")
            # Generate generic but useful follow-up queries for Layer 1
            follow_up_queries = [
                f"{subtopic} - recent developments and updates",
                f"{subtopic} - detailed analysis and specific examples"
            ]
            print(f"[Layer {current_depth}] Generated {len(follow_up_queries)} default follow-up queries")
            return {"follow_up_queries": follow_up_queries}
        else:
            print(f"[Layer {current_depth}] Research deemed comprehensive, no follow-ups needed")
            return {"follow_up_queries": []}

    # Parse follow-up queries
    follow_up_queries = [
        line.strip() for line in response_text.strip().split('\n')
        if line.strip() and not line.strip().startswith(('#', '-', '*'))
    ]

    # SAFEGUARD: If Layer 1 and no queries generated, force at least one follow-up
    if current_depth == 1 and len(follow_up_queries) == 0:
        print(f"[Layer {current_depth}] No follow-ups generated at Layer 1. Adding default query.")
        follow_up_queries = [f"{subtopic} - detailed analysis and recent updates"]

    print(f"[Layer {current_depth}] Generated {len(follow_up_queries)} follow-up queries")

    return {"follow_up_queries": follow_up_queries}


async def conduct_deep_dive_research(state: SubResearcherGraphState) -> SubResearcherGraphState:
    """
    Layers 2-3: Conduct deeper research based on follow-up queries
    - Execute follow-up searches
    - Integrate with existing research
    - Build comprehensive knowledge base
    """
    follow_up_queries = state.get("follow_up_queries", [])
    subtopic = state["subtopic"]
    main_topic = state.get("main_topic", "")
    current_depth = state.get("research_depth", 1)
    max_search_results = state.get("max_search_results", 5)
    search_api = state.get("search_api", "tavily")

    if not follow_up_queries:
        print(f"[Layer {current_depth}] No follow-up queries, skipping deep dive")
        return {}

    new_depth = current_depth + 1
    print(f"[Layer {new_depth}] Conducting deep dive with {len(follow_up_queries)} queries (api={search_api}, max_results={max_search_results})")

    existing_results = state.get("research_results", {})
    existing_credibilities = state.get("source_credibilities", {})

    # Process each follow-up query
    for query in follow_up_queries[:2]:  # Limit to 2 queries per layer
        print(f"[Layer {new_depth}] Searching: {query}")

        # Select search client based on API
        if search_api == "serper" and serper_client:
            search_func = partial(
                serper_client.search,
                query=query,
                max_results=max_search_results
            )
        elif search_api == "tavily":
            search_func = partial(
                tavily_client.search,
                query=query,
                search_depth="advanced",
                max_results=max_search_results
            )
        else:
            print(f"[WARNING] Search API '{search_api}' not available, falling back to Tavily")
            search_func = partial(
                tavily_client.search,
                query=query,
                search_depth="advanced",
                max_results=max_search_results
            )

        search_response = await asyncio.to_thread(search_func)

        results = search_response.get("results", [])
        filtered_results = await filter_quality_sources(results, min_credibility=0.4)

        # Take top 3 per query
        top_sources = filtered_results[:3]

        # Process sources
        tasks = [
            process_source(result, subtopic, main_topic, credibility)
            for result, credibility in top_sources
        ]
        source_data = await asyncio.gather(*tasks)

        # Integrate new findings
        for source_key, findings, credibility in source_data:
            # Avoid duplicates
            if source_key not in existing_results:
                existing_results[source_key] = findings
                existing_credibilities[source_key] = credibility

    print(f"[Layer {new_depth}] Deep dive complete, total sources: {len(existing_results)}")

    return {
        "research_results": existing_results,
        "source_credibilities": existing_credibilities,
        "research_depth": new_depth,
        "follow_up_queries": []  # Clear for next iteration
    }


def should_continue_deep_dive(state: SubResearcherGraphState) -> str:
    """
    Conditional edge: Determine if we should continue with deeper research
    """
    follow_up_queries = state.get("follow_up_queries", [])
    current_depth = state.get("research_depth", 1)
    max_research_depth = state.get("max_research_depth", 2)

    # If we have follow-up queries and haven't reached max depth, continue
    if follow_up_queries and current_depth < max_research_depth:
        print(f"should_continue_deep_dive: YES (depth={current_depth}/{max_research_depth}, queries={len(follow_up_queries)})")
        return "continue"

    print(f"should_continue_deep_dive: NO (depth={current_depth}/{max_research_depth}, queries={len(follow_up_queries)})")
    return "end"


def create_subresearcher_graph():
    """
    Create and compile the enhanced subresearcher subgraph with multi-layer research

    Flow:
    1. Initial broad research (Layer 1)
    2. Analyze and generate follow-up queries
    3. Conditional: If queries exist and depth < 3, conduct deep dive
    4. Loop back to analyze again (for Layer 3) or end
    """
    workflow = StateGraph(SubResearcherGraphState)

    # Add nodes for multi-layer research
    workflow.add_node("initial_research", conduct_initial_research)
    workflow.add_node("analyze_research", analyze_and_generate_follow_ups)
    workflow.add_node("deep_dive", conduct_deep_dive_research)

    # Set entry point
    workflow.set_entry_point("initial_research")

    # Flow: initial research -> analysis
    workflow.add_edge("initial_research", "analyze_research")

    # Conditional: analysis -> deep dive (if queries exist) or END
    workflow.add_conditional_edges(
        "analyze_research",
        should_continue_deep_dive,
        {
            "continue": "deep_dive",
            "end": END
        }
    )

    # After deep dive, analyze again for potential Layer 3
    workflow.add_edge("deep_dive", "analyze_research")

    return workflow.compile()



subresearcher_graph = create_subresearcher_graph()
