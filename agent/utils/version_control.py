from pymongo import AsyncMongoClient
import os, datetime

mongodb_uri = os.getenv("MONGODB_URI")

if not mongodb_uri:
    raise ValueError("MONGODB_URI environment variable is not set")

client = AsyncMongoClient(mongodb_uri)
db = client["langchain_challenge"]
reports_collection = db["reports"]


async def save_report(
        report_id: str,
        version_id: int,
        full_report: str,
):
    """Save report with corresponding report id and version_id"""
    report_doc = {
        "report_id": report_id,
        "version_id": version_id,
        "content": full_report,
        "created_at": datetime.datetime.utcnow(),
    }

    result = await reports_collection.insert_one(report_doc)
    return result.inserted_id


async def get_recent_report_versions(report_id: str, limit: int = 10) -> list[dict]:
    """Get the N most recent versions of a report"""
    cursor = reports_collection.find(
      {"report_id": report_id}
    ).sort("version_id", -1).limit(limit)

    reports = await cursor.to_list(length=limit)
    return reports


async def get_report_version(report_id: str, version_id: int):
    """Get a specific report version"""
    report = reports_collection.find_one({"report_id": report_id, "version_id": version_id})

    return report # None if not found
