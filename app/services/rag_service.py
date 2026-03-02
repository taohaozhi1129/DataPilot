from typing import List, Dict, Any, Optional
import json
import logging

from sentence_transformers import SentenceTransformer

from app.services.milvus_service import MilvusService
from config import settings

# 配置日志记录器
logger = logging.getLogger(__name__)

# 全局单例模型实例，避免重复加载
_GLOBAL_EMBEDDING_MODEL: Optional[SentenceTransformer] = None

class RagService:
    """
    RAG (Retrieval-Augmented Generation) 服务类。
    负责协调 Embedding 模型和 Milvus 向量数据库，执行语义检索任务。
    """
    def __init__(self):
        self._milvus_service: Optional[MilvusService] = None
        # 注意：这里不再保存 _embedding_model 实例变量，而是使用全局变量

    def _get_milvus(self) -> MilvusService:
        """延迟加载 MilvusService 实例"""
        if self._milvus_service is None:
            self._milvus_service = MilvusService()
        return self._milvus_service

    def _get_embedding_model(self) -> SentenceTransformer:
        """
        延迟加载 Embedding 模型 (全局单例模式)。
        确保整个应用生命周期内只加载一次模型，避免性能损耗。
        """
        global _GLOBAL_EMBEDDING_MODEL
        if _GLOBAL_EMBEDDING_MODEL is None:
            logger.info(f"Loading embedding model (Singleton) from: {settings.EMBEDDING_MODEL_PATH}")
            # 这里可能会耗时几秒钟，但只会执行一次
            _GLOBAL_EMBEDDING_MODEL = SentenceTransformer(settings.EMBEDDING_MODEL_PATH)
            logger.info("Embedding model loaded successfully.")
        return _GLOBAL_EMBEDDING_MODEL

    def _get_embedding(self, text: str, instruction: str = "") -> List[float]:
        """
        为单个文本生成向量 embedding。
        
        Args:
            text: 输入文本
            instruction: (Optional) 针对检索任务的查询指令前缀 (适用于 BGE v1.5+)
            
        Returns:
            List[float]: 文本的向量表示
        """
        # 针对 BGE v1.5 等模型，查询端通常需要添加特定指令以提升效果
        input_text = f"{instruction}{text}" if instruction else text
        
        # show_progress_bar=False 避免在生产日志中打印冗余的 tqdm 进度条
        embeddings = self._get_embedding_model().encode([input_text], show_progress_bar=False)
        return embeddings[0].tolist()

    def search_schemas(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """
        在 metadata_collection 中搜索相关的表结构或元数据。
        使用混合检索 (Hybrid Search) 结合关键词 (BM25) 和语义向量。
        
        Args:
            query: 用户查询文本
            limit: 返回结果数量限制
            
        Returns:
            List[Dict[str, Any]]: 检索到的元数据列表
        """
        # BGE v1.5 推荐的中文查询指令
        instruction = "为这个句子生成表示以用于检索相关文章："
        query_vector = self._get_embedding(query, instruction=instruction)
        # 调用 Milvus 混合检索
        results = self._get_milvus().hybrid_search(
            collection_name="metadata_collection",
            query_text=query,
            query_vector=query_vector,
            limit=limit,
        )
        return [self._normalize_schema(hit) for hit in results]

    def search_templates(self, query: str, limit: int = 1) -> List[Dict[str, Any]]:
        """
        在 template_collection 中搜索相关的任务模板。
        使用混合检索 (Hybrid Search) 结合关键词 (BM25) 和语义向量。
        
        Args:
            query: 用户查询文本
            limit: 返回结果数量限制
            
        Returns:
            List[Dict[str, Any]]: 检索到的模板列表
        """
        # BGE v1.5 推荐的中文查询指令
        instruction = "为这个句子生成表示以用于检索相关文章："
        query_vector = self._get_embedding(query, instruction=instruction)
        results = self._get_milvus().hybrid_search(
            collection_name="template_collection",
            query_text=query,
            query_vector=query_vector,
            limit=limit,
        )
        return [self._normalize_template(hit) for hit in results]

    def _normalize_schema(self, hit: Dict[str, Any]) -> Dict[str, Any]:
        """格式化元数据检索结果"""
        metadata = self._safe_json(hit.get("metadata", ""))
        return {
            "content": hit.get("content", ""),
            "metadata": metadata,
            "score": hit.get("score", 0.0)
        }

    def _normalize_template(self, hit: Dict[str, Any]) -> Dict[str, Any]:
        """格式化模板检索结果"""
        payload = self._safe_json(hit.get("payload", ""))
        return {
            "content": hit.get("content", ""),
            "payload": payload,
            "score": hit.get("score", 0.0)
        }

    def _safe_json(self, value: Any) -> Any:
        """安全地解析 JSON 字符串，如果解析失败或不是字符串则原样返回"""
        if isinstance(value, str) and value:
            try:
                return json.loads(value)
            except Exception as e:
                logger.warning(f"JSON parse error: {e}")
                return value
        return value
