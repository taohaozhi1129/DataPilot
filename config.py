from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional
# import os
# os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
class Settings(BaseSettings):
    """
    应用全局配置
    
    已将所有配置集中于此文件。
    您可以直接修改下方的默认值。
    """
    
    # -------------------------------------------------------------------------
    # LLM (OpenAI / Compatible) Configuration
    # -------------------------------------------------------------------------
    OPENAI_API_KEY: Optional[str] = ""
    OPENAI_BASE_URL: str = "https://api.deepseek.com"
    OPENAI_MODEL_NAME: str = "deepseek-chat"
    
    # -------------------------------------------------------------------------
    # Vector Database (Milvus) Configuration
    # -------------------------------------------------------------------------
    MILVUS_URI: str = "http://localhost:19530"
    MILVUS_TOKEN: Optional[str] = ""
    
    # -------------------------------------------------------------------------
    # Embedding Model Configuration
    # -------------------------------------------------------------------------
    EMBEDDING_MODEL_PATH: str = r"D:\Embedding\bge-large-zh-v1.5"
    EMBEDDING_DIMENSION: int = 1024
    MAX_TEXT_LEN: int = 8192

    # -------------------------------------------------------------------------
    # Hybrid Search (BM25 + Dense Vector)
    # -------------------------------------------------------------------------
    BM25_ANALYZER: str = "jieba"
    BM25_K1: float = 1.2
    BM25_B: float = 0.75
    SPARSE_INVERTED_INDEX_ALGO: str = "DAAT_WAND"
    HYBRID_DENSE_WEIGHT: float = 0.6
    HYBRID_SPARSE_WEIGHT: float = 0.4
    HYBRID_CANDIDATE_MULT: int = 5
    HNSW_M: int = 32
    HNSW_EF_CONSTRUCTION: int = 200
    HNSW_EF_SEARCH: int = 96

    # -------------------------------------------------------------------------
    # Conversation Memory
    # -------------------------------------------------------------------------
    HISTORY_MAX_MESSAGES: int = 12
    SUMMARY_TRIGGER_MESSAGES: int = 24
    SUMMARY_WINDOW_MESSAGES: int = 24
    
    # -------------------------------------------------------------------------
    # Logging Configuration
    # -------------------------------------------------------------------------
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    # -------------------------------------------------------------------------
    # Storage & Cache (Redis) Configuration
    # -------------------------------------------------------------------------
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 1
    REDIS_PASSWORD: Optional[str] = None
    ENABLE_REDIS_CHECKPOINTER: bool = True
    REDIS_CONNECT_TIMEOUT_SECONDS: float = 1.5

    @property
    def REDIS_URL(self) -> str:
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # 支持从 .env 和系统环境变量读取配置（系统环境变量优先）
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True, 
        extra="ignore"
    )

settings = Settings()
