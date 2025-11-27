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
        search_results: list[dict] = None
):
    """
    Save report with corresponding report id, version_id, and research data.

    Args:
        report_id: Unique identifier for the report
        version_id: Version number of this report
        full_report: The complete report content
        search_results: List of subresearcher results with research data (optional)
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

    # Add search results if provided
    if search_results:
        report_doc["search_results"] = search_results

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



async def save_test_data(data: dict) -> str:
    """Save a single test data entry to the test-data collection"""
    if not mongodb_enabled or test_data_collection is None:
        print("MongoDB disabled - skipping test data save")
        return None

    data["created_at"] = datetime.datetime.utcnow()
    result = await test_data_collection.insert_one(data)
    return str(result.inserted_id)


async def save_test_data_batch(data_list: list[dict]) -> list[str]:
    """Save multiple test data entries to the test-data collection"""
    if not mongodb_enabled or test_data_collection is None:
        print("MongoDB disabled - skipping test data batch save")
        return []

    for data in data_list:
        data["created_at"] = datetime.datetime.utcnow()

    result = await test_data_collection.insert_many(data_list)
    return [str(id) for id in result.inserted_ids]


async def get_all_test_data(limit: int = 100) -> list[dict]:
    """Get all test data entries"""
    if not mongodb_enabled or test_data_collection is None:
        print("MongoDB disabled - cannot retrieve test data")
        return []

    cursor = test_data_collection.find({}).limit(limit)
    return await cursor.to_list(length=limit)


async def get_test_data_by_id(data_id: str) -> dict:
    """Get a specific test data entry by ID"""
    if not mongodb_enabled or test_data_collection is None:
        print("MongoDB disabled - cannot retrieve test data")
        return None

    from bson import ObjectId
    return await test_data_collection.find_one({"_id": ObjectId(data_id)})


async def get_test_data_by_model(model_name: str, limit: int = 100) -> list[dict]:
    """Get test data entries for a specific model"""
    if not mongodb_enabled or test_data_collection is None:
        print("MongoDB disabled - cannot retrieve test data")
        return []

    cursor = test_data_collection.find({"model": model_name}).limit(limit)
    return await cursor.to_list(length=limit)


async def clear_test_data_collection() -> int:
    """Clear all test data - use with caution"""
    if not mongodb_enabled or test_data_collection is None:
        print("MongoDB disabled - cannot clear test data")
        return 0

    result = await test_data_collection.delete_many({})
    return result.deleted_count
