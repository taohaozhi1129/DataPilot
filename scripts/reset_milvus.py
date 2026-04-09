import os
import sys

from pymilvus import MilvusClient

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from app.infrastructure.milvus.definitions import iter_collection_definitions  # noqa: E402
from config import settings  # noqa: E402


def reset_milvus() -> None:
    print(f"Connecting to Milvus at {settings.MILVUS_URI}...")
    try:
        client = MilvusClient(uri=settings.MILVUS_URI, token=settings.MILVUS_TOKEN)
    except Exception as exc:
        print(f"Failed to connect: {exc}")
        return

    for definition in iter_collection_definitions():
        try:
            client.describe_alias(definition.alias)
            print(f"Dropping alias: {definition.alias}")
            client.drop_alias(definition.alias)
        except Exception:
            print(f"Alias {definition.alias} does not exist.")

        if client.has_collection(definition.name):
            print(f"Dropping collection: {definition.name}")
            client.drop_collection(definition.name)
            print(f"Successfully dropped: {definition.name}")
        else:
            print(f"Collection {definition.name} does not exist.")

    print("\nDone. DataPilot Milvus collections and aliases dropped.")


if __name__ == "__main__":
    reset_milvus()
