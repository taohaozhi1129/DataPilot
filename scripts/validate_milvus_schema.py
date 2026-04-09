import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from app.infrastructure.milvus.definitions import DEFAULT_REQUIRED_COLLECTIONS  # noqa: E402
from app.services.milvus_service import MilvusService  # noqa: E402


def main() -> None:
    service = MilvusService(validate_collections=False)
    results = service.validate_required_collections(DEFAULT_REQUIRED_COLLECTIONS)
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print("\nMilvus schema validation passed.")


if __name__ == "__main__":
    main()
