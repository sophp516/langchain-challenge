from pymongo import AsyncMongoClient
import os, datetime

mongodb_uri = os.getenv("MONGODB_URI")
mongodb_enabled = bool(mongodb_uri)

if mongodb_enabled:
    try:
        client = AsyncMongoClient(
            mongodb_uri,
            tls=True,
            serverSelectionTimeoutMS=5000,
            connectTimeoutMS=10000,
        )
        db = client["langchain_challenge"]
        reports_collection = db["reports"]
        test_data_collection = db["test-data"]
        print("MongoDB connection initialized successfully")
    except Exception as e:
        print(f"[Error] MongoDB connection failed during initialization: {e}")
        mongodb_enabled = False
        client = None
        db = None
        reports_collection = None
        test_data_collection = None
else:
    print("MONGODB_URI not set - MongoDB saving disabled")
    client = None
    db = None
    reports_collection = None
    test_data_collection = None



async def save_report(
        report_id: str,
        version_id: int,
        full_report: str,
        report_title: str = None,
        search_results: str = None,
        report_sections: list = None
):
    """
    Save report with corresponding report id and version_id.

    Args:
        report_id: Unique identifier for the report
        version_id: Version number
        full_report: Complete markdown report content
        report_title: Generated report title (optional)
        search_results: Dictionary of search results by subtopic (optional)
        report_sections: List of section objects with metadata (optional)
    """
    if not mongodb_enabled or reports_collection is None:
        print(f"MongoDB disabled - skipping save for report {report_id} version {version_id}")
        return None

    report_doc = {
        "report_id": report_id,
        "version_id": version_id,
        "content": full_report,
        "created_at": datetime.datetime.utcnow(),
    }

    # Add optional fields if provided
    if report_title:
        report_doc["report_title"] = report_title

    if search_results:
        # Store search results for revision context
        report_doc["search_results"] = search_results

    if report_sections:
        # Store section metadata (titles, subtopics, etc.)
        report_doc["report_sections"] = report_sections

    result = await reports_collection.insert_one(report_doc)
    print(f"Saved report {report_id} v{version_id} with title='{report_title}' and {len(search_results or {})} search result sets")
    return result.inserted_id


async def get_recent_report_versions(report_id: str, limit: int = 10) -> list[dict]:
    """Get the N most recent versions of a report"""
    if not mongodb_enabled or reports_collection is None:
        print(f"MongoDB disabled - cannot retrieve report versions for {report_id}")
        return []

    cursor = reports_collection.find(
      {"report_id": report_id}
    ).sort("version_id", -1).limit(limit)

    reports = await cursor.to_list(length=limit)
    return reports


async def get_report_version(report_id: str, version_id: int):
    """Get a specific report version"""
    if not mongodb_enabled or reports_collection is None:
        print(f"MongoDB disabled - cannot retrieve report {report_id} version {version_id}")
        return None

    report = await reports_collection.find_one({"report_id": report_id, "version_id": version_id})

    return report # None if not found


