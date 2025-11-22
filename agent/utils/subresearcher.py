from typing import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from utils.model import llm
from utils.configuration import tavily_client # Initialize tavily client separately for faster execution
from utils.verification import filter_quality_sources
import asyncio
from functools import partial


class SubResearcherGraphState(TypedDict):
    """State for the subresearcher subgraph"""
    subtopic_id: int
    subtopic: str
    main_topic: str  # The parent topic for context in searches
    research_results: dict[str, str]
    research_depth: int  # Current depth layer (1, 2, or 3)
    source_credibilities: dict[str, float]  # Track credibility scores
    follow_up_queries: list[str]  # Queries for deeper research


async def process_source(
    result: dict,
    subtopic: str,
    credibility_score: float
) -> tuple[str, str, float]:
    """
    Process a single source and return (source_key, findings, credibility) tuple
    Enhanced with credibility tracking
    """
    source_url = result.get("url", "")
    source_title = result.get("title", "Untitled Source")
    source_content = result.get("content", "")

    summary_prompt = f"""
    Based on the following content from a web source, extract and summarize the key findings
    relevant to the subtopic: {subtopic}

    Source Title: {source_title}
    Source URL: {source_url}
    Source Credibility: {credibility_score:.2f}/1.0

    Content:
    {source_content}

    Provide a concise summary of the key findings and insights from this source that are
    relevant to the subtopic. Focus on factual information, data, and insights.
    Include specific facts, statistics, or quotes when available.
    """

    messages = [
        SystemMessage(content="You are a research assistant that extracts and summarizes key findings from sources. Provide clear, factual summaries with specific details."),
        HumanMessage(content=summary_prompt)
    ]

    summary_response = await llm.ainvoke(messages)
    findings = summary_response.content if hasattr(summary_response, 'content') else str(summary_response)

    # Create a source identifier with credibility indicator
    if source_url:
        source_key = f"{source_title} ({source_url})"
    else:
        source_key = source_title

    return (source_key, findings, credibility_score)


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

    print(f"[Layer {current_depth}] Starting research on: {subtopic}")

    # Perform Tavily search (wrap in thread to avoid blocking)
    # Use subtopic directly since it should now include main topic constraints
    # But if main_topic is provided and subtopic seems generic, enhance the query
    search_query = subtopic
    if main_topic and len(subtopic.split()) < 5:
        # Subtopic is short/generic, prepend main topic context
        search_query = f"{main_topic}: {subtopic}"
    search_func = partial(
        tavily_client.search,
        query=search_query,
        search_depth="advanced",
        max_results=8  # Get more results for filtering
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
        process_source(result, subtopic, credibility)
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
    research_results = state.get("research_results", {})
    current_depth = state.get("research_depth", 1)

    if current_depth >= 3:  # Max depth reached
        print(f"[Layer {current_depth}] Max depth reached, no follow-ups needed")
        return {"follow_up_queries": []}

    # Build summary of current findings
    findings_summary = ""
    for i, (source, findings) in enumerate(list(research_results.items())[:5]):
        findings_summary += f"Source {i+1}: {findings[:200]}...\n\n"

    analysis_prompt = f"""
    You are analyzing research on the subtopic: {subtopic}

    Current research findings:
    {findings_summary}

    This is research layer {current_depth} of 3.

    Analyze the findings and identify 2-3 specific follow-up queries that would:
    1. Fill gaps in coverage
    2. Explore promising areas in more depth
    3. Clarify ambiguous or interesting points
    4. Find more recent or specialized information

    Each query should be specific and different from the original subtopic query.

    Return ONLY the queries, one per line, without numbering or bullets.
    If the research seems comprehensive, return "COMPREHENSIVE" instead.
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
    Layers 2-3: Conduct deeper research based on follow-up queries
    - Execute follow-up searches
    - Integrate with existing research
    - Build comprehensive knowledge base
    """
    follow_up_queries = state.get("follow_up_queries", [])
    subtopic = state["subtopic"]
    current_depth = state.get("research_depth", 1)

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

        search_func = partial(
            tavily_client.search,
            query=query,
            search_depth="advanced",
            max_results=5
        )
        search_response = await asyncio.to_thread(search_func)

        results = search_response.get("results", [])
        filtered_results = await filter_quality_sources(results, min_credibility=0.4)

        # Take top 3 per query
        top_sources = filtered_results[:3]

        # Process sources
        tasks = [
            process_source(result, subtopic, credibility)
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

    # If we have follow-up queries and haven't reached max depth, continue
    if follow_up_queries and current_depth < 3:
        print(f"should_continue_deep_dive: YES (depth={current_depth}, queries={len(follow_up_queries)})")
        return "continue"

    print(f"should_continue_deep_dive: NO (depth={current_depth}, queries={len(follow_up_queries)})")
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
