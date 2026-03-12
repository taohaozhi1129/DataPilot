from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage
import logging

from app.services.llm_service import LLMService
from app.agents.state import AgentState
from app.agents.utils.history import split_history_and_input
from config import settings

logger = logging.getLogger(__name__)

class ChatNode:
    """
    通用对话节点 (Chat Node)。
    处理非特定任务的闲聊、问答或兜底回复。
    """
    def __init__(self):
        self._llm_service = None
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            self._llm_service = self._llm_service or LLMService()
            self._llm = self._llm_service.get_llm()
        return self._llm

    def __call__(self, state: AgentState):
        """
        执行对话生成逻辑。
        
        Args:
            state (AgentState): 当前 Agent 的状态
            
        Returns:
            dict: 更新后的状态，包含 messages 和 final_output
        """
        messages = state['messages']
        logger.debug(f"ChatNode processing state with {len(messages)} messages.")
        
        history, user_input = split_history_and_input(
            messages,
            max_history=settings.HISTORY_MAX_MESSAGES,
        )
        summary = state.get("conversation_summary") or ""
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个友好的AI助手。"),
            ("system", "对话摘要（如有）: {summary}"),
            MessagesPlaceholder("history", optional=True),
            ("user", "{input}")
        ])
        
        chain = prompt | self._get_llm()
        
        logger.info(f"Generating chat response for input: {user_input[:50]}...")
        try:
            response = chain.invoke({
                "history": history,
                "input": user_input,
                "summary": summary,
            })
            logger.info("Chat response generated successfully.")
        except Exception as exc:
            logger.error(f"Chat service failed: {exc}", exc_info=True)
            msg = f"抱歉，当前对话服务不可用：{exc}"
            return {
                "messages": [AIMessage(content=msg)],
                "final_output": msg,
                "retrieved_context": {}
            }
        
        return {
            "messages": [AIMessage(content=response.content)],
            "final_output": response.content,
            "retrieved_context": {}
        }

chat_node = ChatNode()
