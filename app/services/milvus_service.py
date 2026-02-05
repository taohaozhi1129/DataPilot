import json
import logging
from typing import List, Dict, Any

from pymilvus import (
    MilvusClient,
    DataType,
    Function,
    FunctionType,
    AnnSearchRequest,
    WeightedRanker,
)
from pymilvus.milvus_client.index import IndexParams

from config import settings

# 初始化日志记录器
logger = logging.getLogger(__name__)


class MilvusService:
    """
    Milvus 向量数据库服务类。
    负责管理集合、插入数据以及执行混合检索 (Hybrid Search)。
    """
    def __init__(self):
        try:
            self.client = MilvusClient(uri=settings.MILVUS_URI, token=settings.MILVUS_TOKEN)
            self._init_collections()
            # 检查集合是否支持 BM25 (用于兼容旧版本集合)
            self._bm25_supported = {
                "metadata_collection": self._collection_supports_bm25("metadata_collection"),
                "template_collection": self._collection_supports_bm25("template_collection"),
            }
        except Exception as exc:
            raise RuntimeError(f"Failed to initialize Milvus client: {exc}") from exc

    def _init_collections(self):
        """初始化 metadata 和 template 集合（如果不存在）。"""
        if not self.client.has_collection("metadata_collection"):
            self._create_metadata_collection()
        if not self.client.has_collection("template_collection"):
            self._create_template_collection()

    def _create_metadata_collection(self):
        """
        创建元数据集合。
        Schema 包含:
        - id: 主键
        - content: 用于显示的文本内容
        - metadata: JSON 格式的元数据
        - text: 用于 BM25 检索的全文文本 (content + metadata)
        - vector: 稠密向量
        - bm25_vector: 稀疏向量 (由 Milvus 内部函数生成)
        """
        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("content", DataType.VARCHAR, max_length=settings.MAX_TEXT_LEN)
        schema.add_field("metadata", DataType.JSON)
        # text 字段用于 BM25 检索，包含 content + metadata，因此长度需要更大 (65535)
        schema.add_field("text", DataType.VARCHAR, max_length=settings.MAX_TEXT_LEN)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=settings.EMBEDDING_DIMENSION)
        schema.add_field("bm25_vector", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_function(
            Function(
                name="bm25_fn",
                function_type=FunctionType.BM25,
                input_field_names=["text"],
                output_field_names=["bm25_vector"],
                params={
                    "analyzer_name": settings.BM25_ANALYZER,
                    "bm25_k1": settings.BM25_K1,
                    "bm25_b": settings.BM25_B,
                },
            )
        )
        index_params = IndexParams()
        index_params.add_index("vector", index_type="AUTOINDEX", metric_type="COSINE")
        index_params.add_index("bm25_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
        self.client.create_collection(
            collection_name="metadata_collection",
            schema=schema,
            index_params=index_params,
        )
        self.client.load_collection("metadata_collection")

    def _create_template_collection(self):
        """创建任务模板集合，结构类似于 metadata_collection。"""
        schema = self.client.create_schema(auto_id=True, enable_dynamic_field=True)
        schema.add_field("id", DataType.INT64, is_primary=True, auto_id=True)
        schema.add_field("content", DataType.VARCHAR, max_length=settings.MAX_TEXT_LEN)
        schema.add_field("payload", DataType.JSON)
        # text 字段用于 BM25 检索，包含 content + payload，因此长度需要更大 (65535)
        schema.add_field("text", DataType.VARCHAR, max_length=65535)
        schema.add_field("vector", DataType.FLOAT_VECTOR, dim=settings.EMBEDDING_DIMENSION)
        schema.add_field("bm25_vector", DataType.SPARSE_FLOAT_VECTOR)
        schema.add_function(
            Function(
                name="bm25_fn",
                function_type=FunctionType.BM25,
                input_field_names=["text"],
                output_field_names=["bm25_vector"],
                params={
                    "analyzer_name": settings.BM25_ANALYZER,
                    "bm25_k1": settings.BM25_K1,
                    "bm25_b": settings.BM25_B,
                },
            )
        )
        index_params = IndexParams()
        index_params.add_index("vector", index_type="AUTOINDEX", metric_type="COSINE")
        index_params.add_index("bm25_vector", index_type="SPARSE_INVERTED_INDEX", metric_type="BM25")
        self.client.create_collection(
            collection_name="template_collection",
            schema=schema,
            index_params=index_params,
        )
        self.client.load_collection("template_collection")

    def _collection_supports_bm25(self, collection_name: str) -> bool:
        """检查集合 schema 是否包含支持 BM25 所需的字段。"""
        try:
            schema_info = self.client.describe_collection(collection_name)
        except Exception:
            return False
        fields = {f.get("name") for f in schema_info.get("fields", [])}
        return "text" in fields and "bm25_vector" in fields

    def insert_metadata(self, content: str, metadata: dict, vector: list[float]):
        """插入数据到 metadata_collection。自动拼接 text 字段以支持 BM25。"""
        row = {
            "vector": vector,
            "content": content,
            "metadata": metadata,
        }
        if self._bm25_supported.get("metadata_collection"):
            row["text"] = f"{content}\n{json.dumps(metadata, ensure_ascii=False)}"
        data = [row]
        return self.client.insert(collection_name="metadata_collection", data=data)

    def insert_template(self, content: str, payload: dict, vector: list[float]):
        """插入数据到 template_collection。自动拼接 text 字段以支持 BM25。"""
        row = {
            "vector": vector,
            "content": content,
            "payload": payload,
        }
        if self._bm25_supported.get("template_collection"):
            row["text"] = f"{content}\n{json.dumps(payload, ensure_ascii=False)}"
        data = [row]
        return self.client.insert(collection_name="template_collection", data=data)

    def search_similar(self, collection_name: str, query_vector: list[float], limit: int = 5):
        """
        仅使用稠密向量进行搜索 (Fallback 方案)。
        """
        output_fields = ["content"]
        if collection_name == "metadata_collection":
            output_fields.append("metadata")
        elif collection_name == "template_collection":
            output_fields.append("payload")

        res = self.client.search(
            collection_name=collection_name,
            data=[query_vector],
            limit=limit,
            output_fields=output_fields,
        )
        hits = res[0] if res else []
        return [self._normalize_hit(hit, collection_name) for hit in hits]

    def _normalize_hit(self, hit: Dict[str, Any], collection_name: str) -> Dict[str, Any]:
        """
        标准化搜索结果。
        """
        # 注意: search 和 hybrid_search 返回的结构略有不同，需要统一处理
        # 这里简化处理，假设直接返回 entity 字典
        entity = hit.get("entity", {})
        if not entity:
             # 兼容 pymilvus 不同版本的返回格式
             entity = hit

        if collection_name == "metadata_collection":
             return {
                 "content": entity.get("content"),
                 "metadata": entity.get("metadata"),
                 "score": hit.get("score", 0.0)
             }
        else:
             return {
                 "content": entity.get("content"),
                 "payload": entity.get("payload"),
                 "score": hit.get("score", 0.0)
             }

    def hybrid_search(
        self,
        collection_name: str,
        query_text: str,
        query_vector: list[float],
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        执行混合检索 (Dense Vector + BM25 Sparse Vector)。
        使用 WeightedRanker 进行结果融合。
        """
        if not self._bm25_supported.get(collection_name):
            return self.search_similar(collection_name, query_vector, limit=limit)

        output_fields = ["content"]
        if collection_name == "metadata_collection":
            output_fields.append("metadata")
        elif collection_name == "template_collection":
            output_fields.append("payload")

        dense_req = AnnSearchRequest(
            data=[query_vector],
            anns_field="vector",
            param={"metric_type": "COSINE"},
            limit=max(limit * settings.HYBRID_CANDIDATE_MULT, limit),
        )
        sparse_req = AnnSearchRequest(
            data=[query_text],
            anns_field="bm25_vector",
            param={
                "metric_type": "BM25",
                "params": {
                    "analyzer_name": settings.BM25_ANALYZER,
                    "bm25_k1": settings.BM25_K1,
                    "bm25_b": settings.BM25_B,
                },
            },
            limit=max(limit * settings.HYBRID_CANDIDATE_MULT, limit),
        )
        ranker = WeightedRanker(
            settings.HYBRID_DENSE_WEIGHT,
            settings.HYBRID_SPARSE_WEIGHT,
            norm_score=True,
        )

        try:
            res = self.client.hybrid_search(
                collection_name=collection_name,
                reqs=[dense_req, sparse_req],
                ranker=ranker,
                limit=limit,
                output_fields=output_fields,
            )
            # hybrid_search 返回的是 hits 列表
            hits = res[0] if res else []
            return [self._normalize_hit(hit, collection_name) for hit in hits]
        except Exception as e:
            logger.warning(f"Hybrid search failed for collection {collection_name}, falling back to dense search. Error: {e}")
            return self.search_similar(collection_name, query_vector, limit=limit)
