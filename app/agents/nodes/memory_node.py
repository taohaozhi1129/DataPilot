from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import logging

from app.services.llm_service import LLMService
from app.agents.state import AgentState
from app.core.prompts import MEMORY_SUMMARY_SYSTEM_PROMPT
from config import settings

logger = logging.getLogger(__name__)

class MemoryNode:
    """
    记忆节点 (Memory Node)。
    负责在对话轮次达到一定阈值时，对历史消息进行摘要压缩，
    以减少 Context Window 占用并保持长期记忆。
    """
    def __init__(self):
        self.llm = LLMService().get_llm()

    def __call__(self, state: AgentState):
        """
        执行记忆摘要逻辑。
        
        Args:
            state (AgentState): 当前 Agent 的状态
            
        Returns:
            dict: 更新后的状态，包含 conversation_summary
        """
        messages = state.get("messages", [])
        summary = state.get("conversation_summary") or ""
        
        current_length = len(messages)
        logger.debug(f"MemoryNode checking conversation length: {current_length} (Trigger: {settings.SUMMARY_TRIGGER_MESSAGES})")

        if current_length < settings.SUMMARY_TRIGGER_MESSAGES:
            return {"conversation_summary": summary}

        logger.info(f"Triggering conversation summary update (Length: {current_length})")
        
        # 提取最近的窗口消息用于生成新摘要
        history = messages[-settings.SUMMARY_WINDOW_MESSAGES :]

        prompt_messages = [
            ("system", MEMORY_SUMMARY_SYSTEM_PROMPT),
        ]
        if summary:
            prompt_messages.append(("system", f"当前摘要：\n{summary}"))

        prompt_messages.append(MessagesPlaceholder("history"))

        prompt = ChatPromptTemplate.from_messages(prompt_messages)
        chain = prompt | self.llm

        try:
            result = chain.invoke({"history": history})
            new_summary = result.content if hasattr(result, "content") else str(result)
            logger.info("Conversation summary updated successfully.")
            logger.debug(f"New Summary: {new_summary[:100]}...")
        except Exception as e:
            logger.error(f"Failed to update conversation summary: {e}", exc_info=True)
            new_summary = summary

        return {"conversation_summary": new_summary}


memory_node = MemoryNode()
