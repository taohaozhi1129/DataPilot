from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from sentence_transformers import SentenceTransformer

from app.infrastructure.milvus.repositories.column_catalog_repository import ColumnCatalogRepository
from app.infrastructure.milvus.repositories.table_catalog_repository import TableCatalogRepository
from app.infrastructure.milvus.repositories.task_template_repository import TaskTemplateRepository
from app.services.milvus_service import MilvusService
from config import settings

logger = logging.getLogger(__name__)

_GLOBAL_EMBEDDING_MODEL: Optional[SentenceTransformer] = None


class RagService:
    """
    Retrieval service for SQL and task generation.

    SQL retrieval uses:
    - table-level catalog
    - column-level catalog

    Task retrieval uses:
    - task template catalog
    """

    def __init__(
        self,
        *,
        milvus_service: Optional[MilvusService] = None,
        table_repository: Optional[TableCatalogRepository] = None,
        column_repository: Optional[ColumnCatalogRepository] = None,
        task_template_repository: Optional[TaskTemplateRepository] = None,
    ) -> None:
        self._milvus_service = milvus_service
        self._table_repository = table_repository
        self._column_repository = column_repository
        self._task_template_repository = task_template_repository

    def _get_milvus_service(self) -> MilvusService:
        if self._milvus_service is None:
            self._milvus_service = MilvusService()
        return self._milvus_service

    def _get_table_repository(self) -> TableCatalogRepository:
        if self._table_repository is None:
            self._table_repository = TableCatalogRepository(self._get_milvus_service())
        return self._table_repository

    def _get_column_repository(self) -> ColumnCatalogRepository:
        if self._column_repository is None:
            self._column_repository = ColumnCatalogRepository(self._get_milvus_service())
        return self._column_repository

    def _get_task_template_repository(self) -> TaskTemplateRepository:
        if self._task_template_repository is None:
            self._task_template_repository = TaskTemplateRepository(self._get_milvus_service())
        return self._task_template_repository

    def _get_embedding_model(self) -> SentenceTransformer:
        global _GLOBAL_EMBEDDING_MODEL
        if _GLOBAL_EMBEDDING_MODEL is None:
            logger.info("Loading embedding model from: %s", settings.EMBEDDING_MODEL_PATH)
            _GLOBAL_EMBEDDING_MODEL = SentenceTransformer(settings.EMBEDDING_MODEL_PATH)
            logger.info("Embedding model loaded successfully.")
        return _GLOBAL_EMBEDDING_MODEL

    def _get_embedding(self, text: str, instruction: str = "") -> List[float]:
        input_text = f"{instruction}{text}" if instruction else text
        embeddings = self._get_embedding_model().encode(
            [input_text],
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return embeddings[0].tolist()

    def search_tables(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        instruction = "为这个句子生成表示以用于检索相关文章："
        query_vector = self._get_embedding(query, instruction=instruction)
        results = self._get_table_repository().search_tables(query, query_vector, limit=limit)
        return [self._normalize_json_fields(item) for item in results]

    def search_columns(
        self,
        query: str,
        *,
        table_ids: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        instruction = "为这个句子生成表示以用于检索相关文章："
        query_vector = self._get_embedding(query, instruction=instruction)
        results = self._get_column_repository().search_columns(
            query,
            query_vector,
            table_ids=table_ids,
            limit=limit,
        )
        return [self._normalize_json_fields(item) for item in results]

    def search_templates(self, query: str, limit: int = 3) -> List[Dict[str, Any]]:
        instruction = "为这个句子生成表示以用于检索相关文章："
        query_vector = self._get_embedding(query, instruction=instruction)
        results = self._get_task_template_repository().search_templates(
            query,
            query_vector,
            limit=limit,
        )
        return [self._normalize_json_fields(item) for item in results]

    def search_sql_context(
        self,
        query: str,
        *,
        table_limit: int = 4,
        column_limit: int = 12,
    ) -> Dict[str, List[Dict[str, Any]]]:
        tables = self.search_tables(query, limit=table_limit)
        table_ids = [table.get("doc_id") for table in tables if table.get("doc_id")]
        columns = self.search_columns(
            query,
            table_ids=table_ids or None,
            limit=column_limit,
        )
        if table_ids and not columns:
            logger.info("No columns found for retrieved tables. Retrying column search without table filter.")
            columns = self.search_columns(query, limit=column_limit)
        return {"tables": tables, "columns": columns}

    def search_schemas(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        sql_context = self.search_sql_context(
            query,
            table_limit=min(max(limit, 1), 4),
            column_limit=max(limit * 2, 6),
        )
        schemas: List[Dict[str, Any]] = []
        columns_by_table: Dict[str, List[Dict[str, Any]]] = {}
        for column in sql_context["columns"]:
            columns_by_table.setdefault(column.get("table_id", ""), []).append(column)

        for table in sql_context["tables"]:
            related_columns = columns_by_table.get(table.get("doc_id", ""), [])
            schemas.append(
                {
                    "content": table.get("full_table_name") or table.get("table_name", ""),
                    "metadata": {
                        "asset_type": "table",
                        **table,
                        "columns": related_columns,
                    },
                    "score": table.get("score", 0.0),
                }
            )

        return schemas[:limit]

    def _normalize_json_fields(self, value: Dict[str, Any]) -> Dict[str, Any]:
        normalized = {}
        for key, item in value.items():
            normalized[key] = self._safe_json(item)
        return normalized

    def _safe_json(self, value: Any) -> Any:
        if isinstance(value, str) and value:
            try:
                return json.loads(value)
            except Exception:
                return value
        return value
