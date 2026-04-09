from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

from pymilvus import AnnSearchRequest, MilvusClient, WeightedRanker

from app.infrastructure.milvus.definitions import (
    DEFAULT_REQUIRED_COLLECTIONS,
    CollectionDefinition,
    build_dense_search_params,
    build_sparse_search_params,
    get_collection_definition,
    resolve_alias_target,
)
from config import settings

logger = logging.getLogger(__name__)


class MilvusValidationError(RuntimeError):
    pass


class MilvusService:
    """
    Thin Milvus access layer for online retrieval.

    Responsibilities:
    - connect to Milvus
    - validate required collections / aliases
    - run dense or hybrid search

    Non-responsibilities:
    - create collections
    - mutate schemas
    - insert data
    """

    def __init__(
        self,
        required_collections: Optional[Sequence[str]] = None,
        *,
        validate_collections: bool = True,
    ) -> None:
        try:
            self.client = MilvusClient(uri=settings.MILVUS_URI, token=settings.MILVUS_TOKEN)
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize Milvus client: {exc}") from exc

        self._bm25_supported: Dict[str, bool] = {}
        if validate_collections:
            self.validate_required_collections(required_collections or DEFAULT_REQUIRED_COLLECTIONS)

    def describe_alias_target(self, alias: str) -> Optional[str]:
        try:
            alias_info = self.client.describe_alias(alias)
        except Exception:
            return None
        return resolve_alias_target(alias_info)

    def resolve_collection_name(self, identifier: str) -> str:
        return self.describe_alias_target(identifier) or identifier

    def validate_required_collections(self, collection_names: Sequence[str]) -> List[Dict[str, Any]]:
        results = []
        for identifier in collection_names:
            definition = get_collection_definition(identifier)
            results.append(self._validate_collection(definition))
        return results

    def collection_supports_bm25(self, identifier: str) -> bool:
        definition = get_collection_definition(identifier)
        cache_key = definition.alias
        if cache_key not in self._bm25_supported:
            self._bm25_supported[cache_key] = self._schema_supports_bm25(definition)
        return self._bm25_supported[cache_key]

    def hybrid_search(
        self,
        collection_name: str,
        query_text: str,
        query_vector: List[float],
        *,
        limit: int = 5,
        output_fields: Optional[List[str]] = None,
        filter_expr: str = "",
    ) -> List[Dict[str, Any]]:
        definition = get_collection_definition(collection_name)
        resolved_collection = self.resolve_collection_name(definition.alias)
        fields = list(output_fields or definition.output_fields)
        dense_search_params = build_dense_search_params(definition)
        sparse_search_params = build_sparse_search_params(definition)

        if not self.collection_supports_bm25(definition.alias):
            return self.search_similar(
                collection_name=definition.alias,
                query_vector=query_vector,
                limit=limit,
                output_fields=fields,
                filter_expr=filter_expr,
            )

        dense_req = AnnSearchRequest(
            data=[query_vector],
            anns_field=definition.vector_field,
            param=dense_search_params,
            limit=max(limit * settings.HYBRID_CANDIDATE_MULT, limit),
            expr=filter_expr or None,
        )
        sparse_req = AnnSearchRequest(
            data=[query_text],
            anns_field=definition.sparse_field,
            param=sparse_search_params,
            limit=max(limit * settings.HYBRID_CANDIDATE_MULT, limit),
            expr=filter_expr or None,
        )
        ranker = WeightedRanker(
            settings.HYBRID_DENSE_WEIGHT,
            settings.HYBRID_SPARSE_WEIGHT,
            norm_score=True,
        )

        try:
            res = self.client.hybrid_search(
                collection_name=resolved_collection,
                reqs=[dense_req, sparse_req],
                ranker=ranker,
                limit=limit,
                output_fields=fields,
            )
            hits = res[0] if res else []
            return [self._normalize_hit(hit, fields) for hit in hits]
        except Exception as exc:
            logger.warning(
                "Hybrid search failed for %s, falling back to dense search: %s",
                resolved_collection,
                exc,
            )
            return self.search_similar(
                collection_name=definition.alias,
                query_vector=query_vector,
                limit=limit,
                output_fields=fields,
                filter_expr=filter_expr,
            )

    def search_similar(
        self,
        collection_name: str,
        query_vector: List[float],
        *,
        limit: int = 5,
        output_fields: Optional[List[str]] = None,
        filter_expr: str = "",
    ) -> List[Dict[str, Any]]:
        definition = get_collection_definition(collection_name)
        resolved_collection = self.resolve_collection_name(definition.alias)
        fields = list(output_fields or definition.output_fields)
        dense_search_params = build_dense_search_params(definition)

        res = self.client.search(
            collection_name=resolved_collection,
            data=[query_vector],
            filter=filter_expr,
            limit=limit,
            output_fields=fields,
            anns_field=definition.vector_field,
            search_params=dense_search_params,
        )
        hits = res[0] if res else []
        return [self._normalize_hit(hit, fields) for hit in hits]

    def query(
        self,
        collection_name: str,
        *,
        filter_expr: str = "",
        output_fields: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        definition = get_collection_definition(collection_name)
        resolved_collection = self.resolve_collection_name(definition.alias)
        fields = list(output_fields or definition.output_fields)
        return self.client.query(
            collection_name=resolved_collection,
            filter=filter_expr,
            output_fields=fields,
            ids=ids,
        )

    def _validate_collection(self, definition: CollectionDefinition) -> Dict[str, Any]:
        alias_target = self.describe_alias_target(definition.alias)
        resolved_collection = alias_target or definition.alias
        if not self.client.has_collection(resolved_collection):
            alias_msg = (
                f"alias {definition.alias} -> {resolved_collection}"
                if alias_target
                else f"alias {definition.alias} is missing"
            )
            raise MilvusValidationError(
                f"Required collection {definition.name} is unavailable: {alias_msg}"
            )

        schema_info = self.client.describe_collection(resolved_collection)
        fields = {field.get("name"): field for field in schema_info.get("fields", [])}
        missing_fields = [name for name in definition.required_fields if name not in fields]
        if missing_fields:
            raise MilvusValidationError(
                f"Collection {resolved_collection} is missing required fields: {missing_fields}"
            )

        vector_dim = self._extract_vector_dim(fields.get(definition.vector_field, {}))
        if vector_dim is not None and vector_dim != settings.EMBEDDING_DIMENSION:
            raise MilvusValidationError(
                f"Collection {resolved_collection} has dim={vector_dim}, expected {settings.EMBEDDING_DIMENSION}"
            )

        try:
            self.client.load_collection(resolved_collection)
        except Exception as exc:
            raise MilvusValidationError(
                f"Failed to load Milvus collection {resolved_collection}: {exc}"
            ) from exc

        self._bm25_supported[definition.alias] = self._schema_supports_bm25(definition)
        return {
            "collection": resolved_collection,
            "alias": definition.alias,
            "bm25_supported": self._bm25_supported[definition.alias],
            "vector_dim": vector_dim,
        }

    def _schema_supports_bm25(self, definition: CollectionDefinition) -> bool:
        resolved_collection = self.resolve_collection_name(definition.alias)
        schema_info = self.client.describe_collection(resolved_collection)
        fields = {field.get("name") for field in schema_info.get("fields", [])}
        return definition.search_text_field in fields and definition.sparse_field in fields

    @staticmethod
    def _extract_vector_dim(field_info: Dict[str, Any]) -> Optional[int]:
        params = field_info.get("params") or {}
        for key in ("dim", "dimension"):
            value = params.get(key)
            if value is None:
                continue
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
        return None

    @staticmethod
    def _normalize_hit(hit: Dict[str, Any], output_fields: List[str]) -> Dict[str, Any]:
        entity = hit.get("entity", {})
        if not entity:
            entity = hit
        normalized = {field: entity.get(field) for field in output_fields if field in entity}
        normalized["score"] = hit.get("score", hit.get("distance", 0.0))
        return normalized
