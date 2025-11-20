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
        print("MongoDB connection initialized successfully")
    except Exception as e:
        print(f"[Error] MongoDB connection failed during initialization: {e}")
        mongodb_enabled = False
        client = None
        db = None
        reports_collection = None
else:
    print("MONGODB_URI not set - MongoDB saving disabled")
    client = None
    db = None
    reports_collection = None


async def save_report(
        report_id: str,
        version_id: int,
        full_report: str,
):
    """Save report with corresponding report id and version_id"""
    if not mongodb_enabled or reports_collection is None:
        print(f"MongoDB disabled - skipping save for report {report_id} version {version_id}")
        return None

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
