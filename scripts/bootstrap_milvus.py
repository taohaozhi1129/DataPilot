import argparse
import os
import sys

from pymilvus import MilvusClient

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from app.infrastructure.milvus.definitions import (  # noqa: E402
    create_collection_indexes,
    create_collection_schema,
    iter_collection_definitions,
    resolve_alias_target,
)
from config import settings  # noqa: E402


def ensure_alias(client: MilvusClient, collection_name: str, alias: str) -> str:
    try:
        alias_info = client.describe_alias(alias)
    except Exception:
        client.create_alias(collection_name=collection_name, alias=alias)
        return "created"

    current_target = resolve_alias_target(alias_info)
    if current_target == collection_name:
        return "unchanged"

    client.alter_alias(collection_name=collection_name, alias=alias)
    return f"updated ({current_target} -> {collection_name})"


def drop_alias_if_exists(client: MilvusClient, alias: str) -> bool:
    try:
        client.describe_alias(alias)
    except Exception:
        return False
    client.drop_alias(alias)
    return True


def bootstrap(drop_existing: bool = False) -> None:
    client = MilvusClient(uri=settings.MILVUS_URI, token=settings.MILVUS_TOKEN)

    for definition in iter_collection_definitions():
        if drop_existing:
            if drop_alias_if_exists(client, definition.alias):
                print(f"Dropped alias: {definition.alias}")
            if client.has_collection(definition.name):
                print(f"Dropping collection: {definition.name}")
                client.drop_collection(definition.name)

        if not client.has_collection(definition.name):
            print(f"Creating collection: {definition.name}")
            schema = create_collection_schema(client, definition)
            indexes = create_collection_indexes(definition)
            client.create_collection(
                collection_name=definition.name,
                schema=schema,
                index_params=indexes,
            )
        else:
            print(f"Collection already exists: {definition.name}")

        client.load_collection(definition.name)
        alias_status = ensure_alias(client, definition.name, definition.alias)
        print(f"Alias {definition.alias}: {alias_status}")

    print("\nMilvus bootstrap completed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create DataPilot Milvus collections and aliases.")
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop existing collections and aliases before bootstrap.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    bootstrap(drop_existing=args.drop_existing)
