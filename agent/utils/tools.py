"""
Tools for report retrieval and re.
These tools can be called by the agent to fetch reports from the database.
"""
from langchain_core.tools import tool
from utils.db import (
    get_report_version,
    get_recent_report_versions,
    mongodb_enabled
)


@tool
async def get_report(report_id: str, version_id: int = None) -> dict:
    """
    Retrieve a specific report from the database.

    Args:
        report_id: The unique identifier of the report (e.g., "report_abc123")
        version_id: Optional specific version number. If not provided, returns the latest version.

    Returns:
        The report document with content, version info, and metadata.
    """
    if not mongodb_enabled:
        return {"error": "Database not available", "report": None}

    if version_id:
        report = await get_report_version(report_id, version_id)
        if report:
            return {
                "found": True,
                "report_id": report_id,
                "version_id": version_id,
                "content": report.get("content", ""),
                "created_at": str(report.get("created_at", ""))
            }
        else:
            return {"found": False, "error": f"Report {report_id} version {version_id} not found"}
    else:
        # Get latest version
        versions = await get_recent_report_versions(report_id, limit=1)
        if versions:
            report = versions[0]
            return {
                "found": True,
                "report_id": report_id,
                "version_id": report.get("version_id"),
                "content": report.get("content", ""),
                "created_at": str(report.get("created_at", ""))
            }
        else:
            return {"found": False, "error": f"Report {report_id} not found"}


@tool
async def list_report_versions(report_id: str, limit: int = 10) -> dict:
    """
    List all versions of a specific report.

    Args:
        report_id: The unique identifier of the report
        limit: Maximum number of versions to return (default 10)

    Returns:
        List of version summaries with version numbers and timestamps.
    """
    if not mongodb_enabled:
        return {"error": "Database not available", "versions": []}

    versions = await get_recent_report_versions(report_id, limit=limit)

    if not versions:
        return {"found": False, "error": f"No versions found for report {report_id}", "versions": []}

    version_list = [
        {
            "version_id": v.get("version_id"),
            "created_at": str(v.get("created_at", "")),
            "content_preview": v.get("content", "")[:200] + "..." if v.get("content") else ""
        }
        for v in versions
    ]

    return {
        "found": True,
        "report_id": report_id,
        "total_versions": len(version_list),
        "versions": version_list
    }


@tool
async def list_all_reports(limit: int = 20) -> dict:
    """
    List all available reports in the database.

    Args:
        limit: Maximum number of reports to return (default 20)

    Returns:
        List of report summaries with IDs and metadata.
    """
    if not mongodb_enabled:
        return {"error": "Database not available", "reports": []}

    from utils.db import reports_collection

    if reports_collection is None:
        return {"error": "Reports collection not available", "reports": []}

    try:
        # Get unique report IDs with their latest versions
        pipeline = [
            {"$sort": {"version_id": -1}},
            {"$group": {
                "_id": "$report_id",
                "latest_version": {"$first": "$version_id"},
                "created_at": {"$first": "$created_at"},
                "content_preview": {"$first": {"$substr": ["$content", 0, 200]}}
            }},
            {"$limit": limit}
        ]

        # AsyncMongoClient: await aggregate() to get cursor, then iterate
        cursor = await reports_collection.aggregate(pipeline)
        reports = []
        async for doc in cursor:
            reports.append(doc)
            if len(reports) >= limit:
                break

        report_list = [
            {
                "report_id": r.get("_id"),
                "latest_version": r.get("latest_version"),
                "created_at": str(r.get("created_at", "")),
                "content_preview": r.get("content_preview", "") + "..."
            }
            for r in reports
        ]

        return {
            "total_reports": len(report_list),
            "reports": report_list
        }
    except Exception as e:
        return {"error": str(e), "reports": []}


@tool
async def revise_report(report_id: str, feedback: str, version_id: int = None) -> dict:
    """
    Revise a report based on user feedback using original research context from database.

    Args:
        report_id: The unique identifier of the report to revise
        feedback: User's feedback describing what to improve/change
        version_id: Optional specific version to revise. If not provided, uses latest version.

    Returns:
        The revised report with new version_id and metadata.
    """
    if not mongodb_enabled:
        return {"error": "Database not available - cannot revise report"}

    from utils.model import llm
    from langchain_core.messages import SystemMessage, HumanMessage
    from utils.db import save_report

    # Fetch the report version to revise
    if version_id:
        report = await get_report_version(report_id, version_id)
    else:
        # Get latest version
        versions = await get_recent_report_versions(report_id, limit=1)
        report = versions[0] if versions else None
        version_id = report.get("version_id") if report else None

    if not report:
        return {"error": f"Report {report_id} not found"}

    # Extract data from report
    current_content = report.get("content", "")
    search_results = report.get("search_results", [])

    # Build research context from search results
    research_context = ""
    if search_results:
        research_context = "\n\n## ORIGINAL RESEARCH DATA (use this to improve the report):\n\n"
        for idx, researcher in enumerate(search_results[:5]):  # Top 5 sections
            subtopic = researcher.get("subtopic", f"Section {idx+1}")
            results = researcher.get("research_results", {})
            credibilities = researcher.get("source_credibilities", {})

            research_context += f"### Research for '{subtopic}':\n"

            # Add top sources from this section
            for source, content in list(results.items())[:5]:  # Top 5 sources per section
                cred = credibilities.get(source, 0.5)
                research_context += f"\n**Source** (credibility: {cred:.2f}): {source}\n{content[:500]}...\n"


    revision_prompt = f"""
    You are revising a research report based on user feedback.

    **CURRENT REPORT:**
    {current_content[:8000]}

    **USER FEEDBACK:**
    {feedback}

    {research_context}

    **INSTRUCTIONS:**
    - Address the user's feedback while maintaining report quality and coherence
    - Use the ORIGINAL RESEARCH DATA above to add new information if needed
    - Keep all existing citations intact and add new ones where appropriate
    - Maintain the existing structure and section organization
    - Ensure the revised report is well-written, clear, and comprehensive

    **CRITICAL - STANDALONE REPORT REQUIREMENTS:**
    - Only use information from the CURRENT REPORT or ORIGINAL RESEARCH DATA
    - Do NOT hallucinate or add information not present in the sources
    - Every new claim must be supported by citations
    - The revised report MUST be COMPLETE and STANDALONE - no meta-commentary allowed
    - Do NOT include phrases like "citation list to be maintained", "details forthcoming", "to be continued", etc.
    - Do NOT reference the original report (e.g., "from the original text", "as mentioned in the original")
    - Include ALL sections, ALL content, and a COMPLETE References section with properly formatted citations
    - The report should read as if it was written fresh, not as a revision

    **FORBIDDEN PHRASES (never use these):**
    - "Citation list to be maintained"
    - "References from the original report"
    - "Details forthcoming"
    - "To be continued"
    - "As per the original"
    - "From the original text"
    - Any meta-commentary about the revision process

    Provide the COMPLETE revised report with all sections and a full references list.
    """

    messages = [
        SystemMessage(content="You are an expert research report writer who revises reports based on feedback while maintaining accuracy and quality."),
        HumanMessage(content=revision_prompt)
    ]

    response = await llm.ainvoke(messages)
    revised_content = response.content if hasattr(response, 'content') else str(response)

    new_version_id = version_id + 1

    try:
        await save_report(report_id, new_version_id, revised_content, search_results)
        print(f"revise_report: saved revised report {report_id} version {new_version_id}")

        return {
            "success": True,
            "report_id": report_id,
            "version_id": new_version_id,
            "previous_version": version_id,
            "content": revised_content,
            "message": f"Report revised successfully. New version: {new_version_id}"
        }
    except Exception as e:
        return {"error": f"Failed to save revised report: {str(e)}"}


report_tools = [get_report, list_report_versions, list_all_reports, revise_report]
