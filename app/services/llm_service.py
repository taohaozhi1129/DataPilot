from langchain_openai import ChatOpenAI
import logging
from config import settings

logger = logging.getLogger(__name__)

class LLMService:
    """
    LLM 服务封装类。
    负责初始化和提供 LangChain ChatOpenAI 实例。
    """
    def __init__(self):
        """
        初始化 LLM 服务。
        检查 API Key 并配置 ChatOpenAI 客户端。
        """
        if not settings.OPENAI_API_KEY:
            logger.critical("OPENAI_API_KEY is missing!")
            raise ValueError("OPENAI_API_KEY is required. Please set it via environment variables or config.")
        
        logger.info(f"Initializing LLM Service with model: {settings.OPENAI_MODEL_NAME} at {settings.OPENAI_BASE_URL}")
        try:
            self.llm = ChatOpenAI(
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_BASE_URL,
                model=settings.OPENAI_MODEL_NAME,
                temperature=0
            )
            logger.debug("LLM initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}", exc_info=True)
            raise

    def get_llm(self):
        """
        获取 LLM 实例。
        
        Returns:
            ChatOpenAI: LangChain 兼容的 LLM 对象
        """
        return self.llm
