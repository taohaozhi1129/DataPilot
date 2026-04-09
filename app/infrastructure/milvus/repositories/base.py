from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from app.infrastructure.milvus.definitions import CollectionDefinition, build_filter_expr
from app.services.milvus_service import MilvusService


class BaseMilvusRepository:
    def __init__(
        self,
        collection: CollectionDefinition,
        milvus_service: Optional[MilvusService] = None,
    ) -> None:
        self.collection = collection
        self.milvus_service = milvus_service or MilvusService()

    def _search(
        self,
        query_text: str,
        query_vector: List[float],
        *,
        limit: int,
        filters: Optional[Dict[str, Any]] = None,
        output_fields: Optional[Iterable[str]] = None,
    ) -> List[Dict[str, Any]]:
        filter_expr = build_filter_expr(filters)
        return self.milvus_service.hybrid_search(
            collection_name=self.collection.alias,
            query_text=query_text,
            query_vector=query_vector,
            limit=limit,
            output_fields=list(output_fields or self.collection.output_fields),
            filter_expr=filter_expr,
        )
