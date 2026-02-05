from langchain_openai import ChatOpenAI
from config import settings

class LLMService:
    def __init__(self):
        if not settings.OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is required. Please set it via environment variables or config.")
        self.llm = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            base_url=settings.OPENAI_BASE_URL,
            model=settings.OPENAI_MODEL_NAME,
            temperature=0
        )

    def get_llm(self):
        return self.llm
