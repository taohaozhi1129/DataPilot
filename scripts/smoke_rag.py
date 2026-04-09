import argparse
import json
import os
import sys
from typing import Any, Dict, List

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from app.services.rag_service import RagService  # noqa: E402


def summarize_hits(hits: List[Dict[str, Any]], fields: List[str]) -> List[Dict[str, Any]]:
    summarized = []
    for item in hits:
        row = {field: item.get(field) for field in fields if item.get(field) is not None}
        row["score"] = round(float(item.get("score", 0.0)), 6)
        summarized.append(row)
    return summarized


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a local RAG smoke check against Milvus.")
    parser.add_argument(
        "--sql-query",
        default="查询订单金额和用户信息",
        help="Query used for table and column retrieval.",
    )
    parser.add_argument(
        "--task-query",
        default="从 mysql 同步到 doris 的离线任务",
        help="Query used for task template retrieval.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 1 when any retrieval branch returns no results.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = RagService()

    sql_context = service.search_sql_context(args.sql_query)
    templates = service.search_templates(args.task_query, limit=3)

    payload = {
        "sql_query": args.sql_query,
        "tables": summarize_hits(
            sql_context.get("tables", []),
            ["doc_id", "full_table_name", "business_domain", "table_type"],
        ),
        "columns": summarize_hits(
            sql_context.get("columns", []),
            ["doc_id", "table_id", "full_column_name", "data_type", "semantic_type"],
        ),
        "task_query": args.task_query,
        "templates": summarize_hits(
            templates,
            ["doc_id", "template_code", "template_name", "template_type"],
        ),
    }

    print(json.dumps(payload, ensure_ascii=False, indent=2))

    if args.strict:
        has_tables = bool(sql_context.get("tables"))
        has_columns = bool(sql_context.get("columns"))
        has_templates = bool(templates)
        if not (has_tables and has_columns and has_templates):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
