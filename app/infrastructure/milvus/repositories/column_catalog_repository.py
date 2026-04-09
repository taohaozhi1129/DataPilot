from __future__ import annotations

from typing import Dict, Iterable, List, Optional

from app.infrastructure.milvus.definitions import COLUMN_CATALOG
from app.infrastructure.milvus.repositories.base import BaseMilvusRepository


class ColumnCatalogRepository(BaseMilvusRepository):
    def __init__(self, milvus_service=None) -> None:
        super().__init__(COLUMN_CATALOG, milvus_service=milvus_service)

    def search_columns(
        self,
        query_text: str,
        query_vector: List[float],
        *,
        table_ids: Optional[Iterable[str]] = None,
        limit: int = 10,
        tenant_id: Optional[str] = None,
        env: Optional[str] = None,
    ) -> List[Dict]:
        filters = {
            "status": "ACTIVE",
            "is_active": True,
            "is_queryable": True,
            "tenant_id": tenant_id,
            "env": env,
            "table_id": list(table_ids) if table_ids else None,
        }
        return self._search(query_text, query_vector, limit=limit, filters=filters)
