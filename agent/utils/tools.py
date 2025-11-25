"""
Tools for report retrieval and management.
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


report_tools = [get_report, list_report_versions, list_all_reports]
