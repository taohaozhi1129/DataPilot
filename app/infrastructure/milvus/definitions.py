from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Tuple

from pymilvus import DataType, Function, FunctionType, MilvusClient
from pymilvus.milvus_client.index import IndexParams

from config import settings

SEARCH_TEXT_MAX_LENGTH = 65535


@dataclass(frozen=True)
class CollectionDefinition:
    name: str
    alias: str
    required_fields: Tuple[str, ...]
    output_fields: Tuple[str, ...]
    vector_field: str = "dense_vector"
    sparse_field: str = "sparse_vector"
    search_text_field: str = "search_text"
    dense_index_type: str = "HNSW"
    dense_metric_type: str = "COSINE"
    dense_index_params: Tuple[Tuple[str, Any], ...] = ()
    dense_search_params: Tuple[Tuple[str, Any], ...] = ()
    sparse_index_type: str = "SPARSE_INVERTED_INDEX"
    sparse_metric_type: str = "BM25"
    sparse_index_params: Tuple[Tuple[str, Any], ...] = ()
    sparse_search_params: Tuple[Tuple[str, Any], ...] = ()


TABLE_CATALOG = CollectionDefinition(
    name="dp_table_catalog_v1",
    alias="dp_table_catalog_current",
    required_fields=(
        "doc_id",
        "tenant_id",
        "env",
        "status",
        "schema_version",
        "datasource_id",
        "catalog_name",
        "database_name",
        "schema_name",
        "table_name",
        "full_table_name",
        "table_type",
        "dialect",
        "business_name",
        "aliases",
        "business_domain",
        "grain_desc",
        "table_desc",
        "primary_keys",
        "partition_keys",
        "time_columns",
        "join_hints",
        "row_count_level",
        "freshness_sla",
        "owner",
        "security_level",
        "is_active",
        "search_text",
        "dense_vector",
        "sparse_vector",
        "payload",
        "updated_at",
    ),
    output_fields=(
        "doc_id",
        "datasource_id",
        "database_name",
        "schema_name",
        "table_name",
        "full_table_name",
        "table_type",
        "dialect",
        "business_name",
        "aliases",
        "business_domain",
        "grain_desc",
        "table_desc",
        "primary_keys",
        "partition_keys",
        "time_columns",
        "join_hints",
        "security_level",
        "payload",
    ),
    dense_index_params=(
        ("M", settings.HNSW_M),
        ("efConstruction", settings.HNSW_EF_CONSTRUCTION),
    ),
    dense_search_params=(("ef", settings.HNSW_EF_SEARCH),),
    sparse_index_params=(
        ("inverted_index_algo", settings.SPARSE_INVERTED_INDEX_ALGO),
        ("bm25_k1", settings.BM25_K1),
        ("bm25_b", settings.BM25_B),
    ),
)

COLUMN_CATALOG = CollectionDefinition(
    name="dp_column_catalog_v1",
    alias="dp_column_catalog_current",
    required_fields=(
        "doc_id",
        "table_id",
        "tenant_id",
        "env",
        "status",
        "schema_version",
        "datasource_id",
        "database_name",
        "schema_name",
        "table_name",
        "column_name",
        "full_column_name",
        "business_name",
        "aliases",
        "data_type",
        "semantic_type",
        "metric_role",
        "is_primary_key",
        "is_foreign_key",
        "is_nullable",
        "is_partition_key",
        "is_time_column",
        "time_granularity",
        "enum_values",
        "sample_values",
        "unit",
        "aggregation_hints",
        "join_hints",
        "column_desc",
        "security_level",
        "is_queryable",
        "is_active",
        "search_text",
        "dense_vector",
        "sparse_vector",
        "payload",
        "updated_at",
    ),
    output_fields=(
        "doc_id",
        "table_id",
        "database_name",
        "schema_name",
        "table_name",
        "column_name",
        "full_column_name",
        "business_name",
        "aliases",
        "data_type",
        "semantic_type",
        "metric_role",
        "is_primary_key",
        "is_foreign_key",
        "is_partition_key",
        "is_time_column",
        "time_granularity",
        "enum_values",
        "sample_values",
        "aggregation_hints",
        "join_hints",
        "column_desc",
        "security_level",
        "payload",
    ),
    dense_index_params=(
        ("M", settings.HNSW_M),
        ("efConstruction", settings.HNSW_EF_CONSTRUCTION),
    ),
    dense_search_params=(("ef", settings.HNSW_EF_SEARCH),),
    sparse_index_params=(
        ("inverted_index_algo", settings.SPARSE_INVERTED_INDEX_ALGO),
        ("bm25_k1", settings.BM25_K1),
        ("bm25_b", settings.BM25_B),
    ),
)

TASK_TEMPLATE_CATALOG = CollectionDefinition(
    name="dp_task_template_v1",
    alias="dp_task_template_current",
    required_fields=(
        "doc_id",
        "tenant_id",
        "env",
        "status",
        "schema_version",
        "template_code",
        "template_name",
        "template_type",
        "business_domain",
        "source_types",
        "target_types",
        "schedule_modes",
        "required_slots",
        "optional_slots",
        "slot_schema",
        "compatibility_rules",
        "default_payload",
        "payload_schema",
        "render_rules",
        "example_inputs",
        "example_payloads",
        "template_desc",
        "risk_level",
        "version_name",
        "is_active",
        "search_text",
        "dense_vector",
        "sparse_vector",
        "payload",
        "updated_at",
    ),
    output_fields=(
        "doc_id",
        "template_code",
        "template_name",
        "template_type",
        "business_domain",
        "source_types",
        "target_types",
        "schedule_modes",
        "required_slots",
        "optional_slots",
        "slot_schema",
        "compatibility_rules",
        "default_payload",
        "payload_schema",
        "render_rules",
        "example_inputs",
        "example_payloads",
        "template_desc",
        "risk_level",
        "version_name",
        "payload",
    ),
    dense_index_type="FLAT",
    sparse_index_params=(
        ("inverted_index_algo", settings.SPARSE_INVERTED_INDEX_ALGO),
        ("bm25_k1", settings.BM25_K1),
        ("bm25_b", settings.BM25_B),
    ),
)

COLLECTION_DEFINITIONS: Dict[str, CollectionDefinition] = {
    TABLE_CATALOG.name: TABLE_CATALOG,
    TABLE_CATALOG.alias: TABLE_CATALOG,
    COLUMN_CATALOG.name: COLUMN_CATALOG,
    COLUMN_CATALOG.alias: COLUMN_CATALOG,
    TASK_TEMPLATE_CATALOG.name: TASK_TEMPLATE_CATALOG,
    TASK_TEMPLATE_CATALOG.alias: TASK_TEMPLATE_CATALOG,
}

DEFAULT_REQUIRED_COLLECTIONS = (
    TABLE_CATALOG.alias,
    COLUMN_CATALOG.alias,
    TASK_TEMPLATE_CATALOG.alias,
)


def get_collection_definition(identifier: str) -> CollectionDefinition:
    try:
        return COLLECTION_DEFINITIONS[identifier]
    except KeyError as exc:
        raise KeyError(f"Unknown Milvus collection identifier: {identifier}") from exc


def iter_collection_definitions() -> Iterable[CollectionDefinition]:
    yielded = set()
    for definition in (TABLE_CATALOG, COLUMN_CATALOG, TASK_TEMPLATE_CATALOG):
        if definition.name not in yielded:
            yielded.add(definition.name)
            yield definition


def build_filter_expr(filters: Dict[str, Any] | None) -> str:
    if not filters:
        return ""

    expressions = []
    for key, value in filters.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            values = [item for item in value if item is not None]
            if not values:
                continue
            formatted = ", ".join(_format_filter_value(item) for item in values)
            expressions.append(f"{key} in [{formatted}]")
            continue
        expressions.append(f"{key} == {_format_filter_value(value)}")
    return " and ".join(expressions)


def _format_filter_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    escaped = str(value).replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def resolve_alias_target(alias_info: Dict[str, Any]) -> str | None:
    for key in ("collection", "collection_name", "collectionName"):
        value = alias_info.get(key)
        if value:
            return value
    return None


def _pairs_to_dict(pairs: Tuple[Tuple[str, Any], ...]) -> Dict[str, Any]:
    return {key: value for key, value in pairs}


def create_collection_schema(client: MilvusClient, definition: CollectionDefinition):
    if definition is TABLE_CATALOG:
        return _build_table_catalog_schema(client)
    if definition is COLUMN_CATALOG:
        return _build_column_catalog_schema(client)
    if definition is TASK_TEMPLATE_CATALOG:
        return _build_task_template_schema(client)
    raise ValueError(f"Unsupported collection definition: {definition.name}")


def create_collection_indexes(definition: CollectionDefinition) -> IndexParams:
    index_params = IndexParams()
    dense_kwargs: Dict[str, Any] = {"metric_type": definition.dense_metric_type}
    dense_params = _pairs_to_dict(definition.dense_index_params)
    if dense_params:
        dense_kwargs["params"] = dense_params
    index_params.add_index(
        definition.vector_field,
        index_type=definition.dense_index_type,
        **dense_kwargs,
    )

    sparse_kwargs: Dict[str, Any] = {"metric_type": definition.sparse_metric_type}
    sparse_params = _pairs_to_dict(definition.sparse_index_params)
    if sparse_params:
        sparse_kwargs["params"] = sparse_params
    index_params.add_index(
        definition.sparse_field,
        index_type=definition.sparse_index_type,
        **sparse_kwargs,
    )
    return index_params


def build_dense_search_params(definition: CollectionDefinition) -> Dict[str, Any]:
    params: Dict[str, Any] = {"metric_type": definition.dense_metric_type}
    dense_search_params = _pairs_to_dict(definition.dense_search_params)
    if dense_search_params:
        params["params"] = dense_search_params
    return params


def build_sparse_search_params(definition: CollectionDefinition) -> Dict[str, Any]:
    params: Dict[str, Any] = {"metric_type": definition.sparse_metric_type}
    sparse_search_params = _pairs_to_dict(definition.sparse_search_params)
    if sparse_search_params:
        params["params"] = sparse_search_params
    return params


def _build_base_schema(client: MilvusClient):
    schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
    schema.add_field("doc_id", DataType.VARCHAR, max_length=64, is_primary=True)
    schema.add_field("tenant_id", DataType.VARCHAR, max_length=32)
    schema.add_field("env", DataType.VARCHAR, max_length=16)
    schema.add_field("status", DataType.VARCHAR, max_length=16)
    schema.add_field("schema_version", DataType.INT64)
    return schema


def _attach_search_fields(schema) -> None:
    schema.add_field(
        "search_text",
        DataType.VARCHAR,
        max_length=SEARCH_TEXT_MAX_LENGTH,
        enable_analyzer=True,
    )
    schema.add_field("dense_vector", DataType.FLOAT_VECTOR, dim=settings.EMBEDDING_DIMENSION)
    schema.add_field("sparse_vector", DataType.SPARSE_FLOAT_VECTOR)
    schema.add_field("payload", DataType.JSON)
    schema.add_field("updated_at", DataType.INT64)
    schema.add_function(
        Function(
            name="bm25_fn",
            function_type=FunctionType.BM25,
            input_field_names=["search_text"],
            output_field_names=["sparse_vector"],
        )
    )


def _build_table_catalog_schema(client: MilvusClient):
    schema = _build_base_schema(client)
    schema.add_field("datasource_id", DataType.VARCHAR, max_length=64)
    schema.add_field("catalog_name", DataType.VARCHAR, max_length=128)
    schema.add_field("database_name", DataType.VARCHAR, max_length=128)
    schema.add_field("schema_name", DataType.VARCHAR, max_length=128)
    schema.add_field("table_name", DataType.VARCHAR, max_length=256)
    schema.add_field("full_table_name", DataType.VARCHAR, max_length=512)
    schema.add_field("table_type", DataType.VARCHAR, max_length=32)
    schema.add_field("dialect", DataType.VARCHAR, max_length=32)
    schema.add_field("business_name", DataType.VARCHAR, max_length=256)
    schema.add_field("aliases", DataType.JSON)
    schema.add_field("business_domain", DataType.VARCHAR, max_length=128)
    schema.add_field("grain_desc", DataType.VARCHAR, max_length=512)
    schema.add_field("table_desc", DataType.VARCHAR, max_length=4096)
    schema.add_field("primary_keys", DataType.JSON)
    schema.add_field("partition_keys", DataType.JSON)
    schema.add_field("time_columns", DataType.JSON)
    schema.add_field("join_hints", DataType.JSON)
    schema.add_field("row_count_level", DataType.VARCHAR, max_length=32)
    schema.add_field("freshness_sla", DataType.VARCHAR, max_length=64)
    schema.add_field("owner", DataType.VARCHAR, max_length=128)
    schema.add_field("security_level", DataType.VARCHAR, max_length=32)
    schema.add_field("is_active", DataType.BOOL)
    _attach_search_fields(schema)
    return schema


def _build_column_catalog_schema(client: MilvusClient):
    schema = _build_base_schema(client)
    schema.add_field("table_id", DataType.VARCHAR, max_length=64)
    schema.add_field("datasource_id", DataType.VARCHAR, max_length=64)
    schema.add_field("database_name", DataType.VARCHAR, max_length=128)
    schema.add_field("schema_name", DataType.VARCHAR, max_length=128)
    schema.add_field("table_name", DataType.VARCHAR, max_length=256)
    schema.add_field("column_name", DataType.VARCHAR, max_length=256)
    schema.add_field("full_column_name", DataType.VARCHAR, max_length=768)
    schema.add_field("business_name", DataType.VARCHAR, max_length=256)
    schema.add_field("aliases", DataType.JSON)
    schema.add_field("data_type", DataType.VARCHAR, max_length=64)
    schema.add_field("semantic_type", DataType.VARCHAR, max_length=64)
    schema.add_field("metric_role", DataType.VARCHAR, max_length=64)
    schema.add_field("is_primary_key", DataType.BOOL)
    schema.add_field("is_foreign_key", DataType.BOOL)
    schema.add_field("is_nullable", DataType.BOOL)
    schema.add_field("is_partition_key", DataType.BOOL)
    schema.add_field("is_time_column", DataType.BOOL)
    schema.add_field("time_granularity", DataType.VARCHAR, max_length=32)
    schema.add_field("enum_values", DataType.JSON)
    schema.add_field("sample_values", DataType.JSON)
    schema.add_field("unit", DataType.VARCHAR, max_length=32)
    schema.add_field("aggregation_hints", DataType.JSON)
    schema.add_field("join_hints", DataType.JSON)
    schema.add_field("column_desc", DataType.VARCHAR, max_length=4096)
    schema.add_field("security_level", DataType.VARCHAR, max_length=32)
    schema.add_field("is_queryable", DataType.BOOL)
    schema.add_field("is_active", DataType.BOOL)
    _attach_search_fields(schema)
    return schema


def _build_task_template_schema(client: MilvusClient):
    schema = _build_base_schema(client)
    schema.add_field("template_code", DataType.VARCHAR, max_length=128)
    schema.add_field("template_name", DataType.VARCHAR, max_length=256)
    schema.add_field("template_type", DataType.VARCHAR, max_length=64)
    schema.add_field("business_domain", DataType.VARCHAR, max_length=128)
    schema.add_field("source_types", DataType.JSON)
    schema.add_field("target_types", DataType.JSON)
    schema.add_field("schedule_modes", DataType.JSON)
    schema.add_field("required_slots", DataType.JSON)
    schema.add_field("optional_slots", DataType.JSON)
    schema.add_field("slot_schema", DataType.JSON)
    schema.add_field("compatibility_rules", DataType.JSON)
    schema.add_field("default_payload", DataType.JSON)
    schema.add_field("payload_schema", DataType.JSON)
    schema.add_field("render_rules", DataType.JSON)
    schema.add_field("example_inputs", DataType.JSON)
    schema.add_field("example_payloads", DataType.JSON)
    schema.add_field("template_desc", DataType.VARCHAR, max_length=4096)
    schema.add_field("risk_level", DataType.VARCHAR, max_length=32)
    schema.add_field("version_name", DataType.VARCHAR, max_length=64)
    schema.add_field("is_active", DataType.BOOL)
    _attach_search_fields(schema)
    return schema
