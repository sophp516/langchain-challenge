from typing import TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.graph import StateGraph, END
from utils.state import SubResearcherState
from utils.model import llm
from utils.tavily import tavily_client # initialize tavily client separately for faster execution
import asyncio


class SubResearcherGraphState(TypedDict):
    """State for the subresearcher subgraph"""
    subtopic_id: int
    subtopic: str
    research_results: dict[str, str]


async def process_source(result: dict, subtopic: str) -> tuple[str, str]:
    """Process a single source and return (source_key, findings) tuple"""
    source_url = result.get("url", "")
    source_title = result.get("title", "Untitled Source")
    source_content = result.get("content", "")
    
    summary_prompt = f"""
    Based on the following content from a web source, extract and summarize the key findings 
    relevant to the subtopic: {subtopic}
    
    Source Title: {source_title}
    Source URL: {source_url}
    
    Content:
    {source_content}
    
    Provide a concise summary of the key findings and insights from this source that are 
    relevant to the subtopic. Focus on factual information, data, and insights.
    """
    
    messages = [
        SystemMessage(content="You are a research assistant that extracts and summarizes key findings from sources. Provide clear, factual summaries."),
        HumanMessage(content=summary_prompt)
    ]
    
    summary_response = await llm.ainvoke(messages)
    findings = summary_response.content if hasattr(summary_response, 'content') else str(summary_response)
    
    # Create a source identifier
    if source_url:
        source_key = f"{source_title} ({source_url})"
    else:
        source_key = source_title
    
    return (source_key, findings)


async def conduct_research(state: SubResearcherGraphState) -> SubResearcherGraphState:
    """Conduct deep research on a specific subtopic with sources using Tavily web search"""
    subtopic = state["subtopic"]
    
    search_query = subtopic
    search_response = tavily_client.search(
        query=search_query,
        search_depth="advanced",  # Use advanced search for better results
        max_results=5  # Get top 5 results
    )
    
    # Process all sources in parallel for faster execution
    results = search_response.get("results", [])
    tasks = [process_source(result, subtopic) for result in results]
    source_findings = await asyncio.gather(*tasks)
    
    research_results = {source_key: findings for source_key, findings in source_findings}
    
    return {
        "subtopic_id": state.get("subtopic_id", 0),
        "subtopic": subtopic,
        "research_results": research_results
    }


def create_subresearcher_graph():
    """Create and compile the subresearcher subgraph"""
    workflow = StateGraph(SubResearcherGraphState)
    
    workflow.add_node("conduct_research", conduct_research)
    
    workflow.set_entry_point("conduct_research")
    
    workflow.add_edge("conduct_research", END)
    
    return workflow.compile()



subresearcher_graph = create_subresearcher_graph()
