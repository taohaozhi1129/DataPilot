from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    """
    应用全局配置
    
    已将所有配置集中于此文件。
    您可以直接修改下方的默认值。
    """
    
    # -------------------------------------------------------------------------
    # LLM (OpenAI / Compatible) Configuration
    # -------------------------------------------------------------------------
    OPENAI_API_KEY: str = "sk-b073a26a7c7d4eba85d6ebe077edd761"
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
    EMBEDDING_MODEL_PATH: str = "BAAI/bge-base-zh"
    EMBEDDING_DIMENSION: int = 768
    MAX_TEXT_LEN: int = 8192

    # -------------------------------------------------------------------------
    # Hybrid Search (BM25 + Dense Vector)
    # -------------------------------------------------------------------------
    BM25_ANALYZER: str = "jieba"
    BM25_K1: float = 1.2
    BM25_B: float = 0.75
    HYBRID_DENSE_WEIGHT: float = 0.6
    HYBRID_SPARSE_WEIGHT: float = 0.4
    HYBRID_CANDIDATE_MULT: int = 5
    
    # -------------------------------------------------------------------------
    # Storage & Cache (Redis) Configuration
    # -------------------------------------------------------------------------
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 1
    REDIS_PASSWORD: Optional[str] = None

    @property
    def REDIS_URL(self) -> str:
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"

    # 忽略环境变量文件，直接使用上述值（也可以保留环境变量覆盖的能力）
    model_config = SettingsConfigDict(
        env_ignore_empty=True, 
        extra="ignore"
    )

settings = Settings()
