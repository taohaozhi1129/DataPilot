import json
import os
import sys
import time
from typing import Any, Dict, List

from pymilvus import MilvusClient
from sentence_transformers import SentenceTransformer

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from app.infrastructure.milvus.definitions import (  # noqa: E402
    COLUMN_CATALOG,
    TABLE_CATALOG,
    TASK_TEMPLATE_CATALOG,
)
from config import settings  # noqa: E402

DATA_DIR = os.path.join(PROJECT_ROOT, "tests", "data")
TABLES_FILE = os.path.join(DATA_DIR, "tables.json")
COLUMNS_FILE = os.path.join(DATA_DIR, "columns.json")
TEMPLATES_FILE = os.path.join(DATA_DIR, "task_templates.json")

client = MilvusClient(uri=settings.MILVUS_URI, token=settings.MILVUS_TOKEN)
model = SentenceTransformer(settings.EMBEDDING_MODEL_PATH)


def embed_text(text: str) -> List[float]:
    return model.encode(text, normalize_embeddings=True).tolist()


def load_json_file(file_path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(file_path):
        return []
    with open(file_path, "r", encoding="utf-8") as file:
        return json.load(file)


def table_doc_id(table: Dict[str, Any]) -> str:
    schema_name = table.get("schema_name") or table.get("database_name") or "default"
    return f"table::{schema_name}::{table['table_name']}"


def column_doc_id(column: Dict[str, Any], schema_name: str) -> str:
    return f"column::{schema_name}::{column['table_name']}::{column['column_name']}"


def build_table_metadata_index() -> Dict[str, Dict[str, str]]:
    metadata: Dict[str, Dict[str, str]] = {}
    for table in load_json_file(TABLES_FILE):
        metadata[table["table_name"]] = {
            "schema_name": table.get("schema_name") or table.get("database_name") or "default",
            "datasource_id": table.get("data_source", "default"),
        }
    return metadata


def detect_semantic_type(column: Dict[str, Any]) -> str:
    column_name = column.get("column_name", "").lower()
    data_type = column.get("data_type", "").lower()
    if column.get("is_primary_key"):
        return "ID"
    if "time" in data_type or "date" in data_type or column_name.endswith("_at"):
        return "TIME"
    if any(token in data_type for token in ("int", "decimal", "double", "float", "bigint")):
        return "METRIC"
    return "DIMENSION"


def detect_metric_role(column: Dict[str, Any]) -> str:
    semantic_type = detect_semantic_type(column)
    if semantic_type != "METRIC":
        return "NON_AGG"
    return "SUMMABLE"


def build_table_rows() -> List[Dict[str, Any]]:
    rows = []
    now = int(time.time())
    for table in load_json_file(TABLES_FILE):
        schema_name = table.get("schema_name") or "default"
        full_table_name = f"{schema_name}.{table['table_name']}"
        search_text = "\n".join(
            [
                table.get("text", ""),
                table.get("table_name", ""),
                table.get("table_comment", ""),
                table.get("business_domain", ""),
                table.get("data_source", ""),
            ]
        ).strip()
        rows.append(
            {
                "doc_id": table_doc_id(table),
                "tenant_id": "default",
                "env": "dev",
                "status": "ACTIVE",
                "schema_version": 1,
                "datasource_id": table.get("data_source", "default"),
                "catalog_name": "",
                "database_name": schema_name,
                "schema_name": schema_name,
                "table_name": table["table_name"],
                "full_table_name": full_table_name,
                "table_type": "FACT" if table.get("is_fact_table") else "DIM",
                "dialect": "ANSI",
                "business_name": table.get("table_name", ""),
                "aliases": [table.get("table_name", "")],
                "business_domain": table.get("business_domain", ""),
                "grain_desc": table.get("table_comment", ""),
                "table_desc": table.get("table_comment", ""),
                "primary_keys": [],
                "partition_keys": [],
                "time_columns": [],
                "join_hints": [],
                "row_count_level": "",
                "freshness_sla": "",
                "owner": "",
                "security_level": "INTERNAL",
                "is_active": True,
                "search_text": search_text,
                "dense_vector": embed_text(search_text),
                "payload": table,
                "updated_at": now,
            }
        )
    return rows


def build_column_rows() -> List[Dict[str, Any]]:
    rows = []
    now = int(time.time())
    table_metadata = build_table_metadata_index()
    for column in load_json_file(COLUMNS_FILE):
        source_table = table_metadata.get(column["table_name"], {})
        schema_name = (
            column.get("schema_name")
            or column.get("database_name")
            or source_table.get("schema_name")
            or "default"
        )
        table_id = f"table::{schema_name}::{column['table_name']}"
        full_column_name = f"{schema_name}.{column['table_name']}.{column['column_name']}"
        search_text = "\n".join(
            [
                column.get("text", ""),
                column.get("column_name", ""),
                column.get("column_comment", ""),
                column.get("business_meaning", ""),
                column.get("metric_type", ""),
                column.get("enum_values", ""),
            ]
        ).strip()
        rows.append(
            {
                "doc_id": column_doc_id(column, schema_name),
                "table_id": table_id,
                "tenant_id": "default",
                "env": "dev",
                "status": "ACTIVE",
                "schema_version": 1,
                "datasource_id": source_table.get("datasource_id", "default"),
                "database_name": schema_name,
                "schema_name": schema_name,
                "table_name": column["table_name"],
                "column_name": column["column_name"],
                "full_column_name": full_column_name,
                "business_name": column.get("column_comment", ""),
                "aliases": [column.get("column_name", "")],
                "data_type": column.get("data_type", ""),
                "semantic_type": detect_semantic_type(column),
                "metric_role": detect_metric_role(column),
                "is_primary_key": bool(column.get("is_primary_key")),
                "is_foreign_key": bool(column.get("is_foreign_key")),
                "is_nullable": bool(column.get("is_nullable")),
                "is_partition_key": False,
                "is_time_column": detect_semantic_type(column) == "TIME",
                "time_granularity": "",
                "enum_values": _safe_json_value(column.get("enum_values")),
                "sample_values": [],
                "unit": "",
                "aggregation_hints": [],
                "join_hints": [],
                "column_desc": column.get("business_meaning", column.get("column_comment", "")),
                "security_level": "INTERNAL",
                "is_queryable": True,
                "is_active": True,
                "search_text": search_text,
                "dense_vector": embed_text(search_text),
                "payload": column,
                "updated_at": now,
            }
        )
    return rows


def build_template_rows() -> List[Dict[str, Any]]:
    rows = []
    now = int(time.time())
    for template in load_json_file(TEMPLATES_FILE):
        search_text = "\n".join(
            [
                template.get("template_name", ""),
                template.get("template_type", ""),
                template.get("template_desc", ""),
                json.dumps(template.get("required_slots", []), ensure_ascii=False),
                json.dumps(template.get("source_types", []), ensure_ascii=False),
                json.dumps(template.get("target_types", []), ensure_ascii=False),
            ]
        ).strip()
        rows.append(
            {
                "doc_id": template["doc_id"],
                "tenant_id": template.get("tenant_id", "default"),
                "env": template.get("env", "dev"),
                "status": template.get("status", "ACTIVE"),
                "schema_version": template.get("schema_version", 1),
                "template_code": template.get("template_code", ""),
                "template_name": template.get("template_name", ""),
                "template_type": template.get("template_type", ""),
                "business_domain": template.get("business_domain", ""),
                "source_types": template.get("source_types", []),
                "target_types": template.get("target_types", []),
                "schedule_modes": template.get("schedule_modes", []),
                "required_slots": template.get("required_slots", []),
                "optional_slots": template.get("optional_slots", []),
                "slot_schema": template.get("slot_schema", {}),
                "compatibility_rules": template.get("compatibility_rules", {}),
                "default_payload": template.get("default_payload", {}),
                "payload_schema": template.get("payload_schema", {}),
                "render_rules": template.get("render_rules", {}),
                "example_inputs": template.get("example_inputs", []),
                "example_payloads": template.get("example_payloads", []),
                "template_desc": template.get("template_desc", ""),
                "risk_level": template.get("risk_level", "MEDIUM"),
                "version_name": template.get("version_name", "v1"),
                "is_active": template.get("is_active", True),
                "search_text": search_text,
                "dense_vector": embed_text(search_text),
                "payload": template,
                "updated_at": now,
            }
        )
    return rows


def upsert_rows(collection_name: str, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print(f"Skip {collection_name}: no rows to upsert.")
        return
    client.upsert(collection_name=collection_name, data=rows)
    print(f"Upserted {len(rows)} rows into {collection_name}.")


def _safe_json_value(value: Any) -> Any:
    if isinstance(value, str) and value:
        try:
            return json.loads(value)
        except Exception:
            return value
    return value


def main() -> None:
    upsert_rows(TABLE_CATALOG.alias, build_table_rows())
    upsert_rows(COLUMN_CATALOG.alias, build_column_rows())
    upsert_rows(TASK_TEMPLATE_CATALOG.alias, build_template_rows())


if __name__ == "__main__":
    main()
