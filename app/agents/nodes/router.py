from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
import logging

from app.services.llm_service import LLMService
from app.agents.state import AgentState
from app.core.prompts import ROUTER_SYSTEM_PROMPT
from app.core.intents import get_intent_options, get_few_shot_examples
from app.agents.utils.history import split_history_and_input
from config import settings

logger = logging.getLogger(__name__)

# 定义结构化输出，包含推理过程 (Chain of Thought)
class RouterOutput(BaseModel):
    """
    路由器的输出结构模型。
    包含思维链 (CoT) 和最终分类结果。
    """
    thought: str = Field(
        description="思考过程。简要分析用户的输入，判断其核心意图，并解释分类理由。"
    )
    category: str = Field(
        description="用户意图的分类。必须是已注册的意图名称之一 (如 SQL_QUERY, TASK_CREATE, CHAT)。"
    )

class RouterNode:
    """
    路由节点 (Router Node)。
    负责根据用户输入和对话历史，判断用户的意图 (Intent)。
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
        执行路由逻辑。
        
        Args:
            state (AgentState): 当前 Agent 的状态
            
        Returns:
            dict: 更新后的状态，包含 intent 字段
        """
        messages = state['messages']
        logger.debug(f"Router processing state with {len(messages)} messages.")
        
        # 分离历史消息和当前输入，避免上下文过长
        history, user_input = split_history_and_input(
            messages,
            max_history=settings.HISTORY_MAX_MESSAGES,
        )
        summary = state.get("conversation_summary") or ""

        # 1. 动态获取意图选项和示例
        intent_options = get_intent_options()
        few_shot_examples = get_few_shot_examples()

        # 2. 注入到 Prompt 模板中
        system_prompt = ROUTER_SYSTEM_PROMPT.format(
            intent_options=intent_options,
            few_shot_examples=few_shot_examples
        )
        
        # 3. 使用 PydanticOutputParser 确保兼容性
        parser = PydanticOutputParser(pydantic_object=RouterOutput)
        
        # 将格式说明附加到系统提示词
        # 使用 placeholder 避免直接拼接导致的 prompt template 解析错误（将 JSON schema 中的花括号误认为是变量）
        system_prompt += "\n\n请严格按照以下 JSON 格式输出结果，不要包含 markdown 代码块标记：\n{format_instructions}"

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("system", "对话摘要（如有）: {summary}"),
            MessagesPlaceholder("history", optional=True),
            ("user", "{input}"),
        ])
        
        # 使用 partial 注入 format_instructions，这样其中的花括号不会被再次解析
        prompt = prompt.partial(format_instructions=parser.get_format_instructions())
        
        chain = prompt | self._get_llm() | parser
        
        logger.info(f"Routing user input: {user_input[:50]}...")
        try:
            result = chain.invoke({
                "history": history,
                "input": user_input,
                "summary": summary,
            })
            logger.info(f"Intent detected: {result.category} | Thought: {result.thought}")
        except Exception as e:
            logger.error(f"Router failed to determine intent: {e}", exc_info=True)
            # 兜底策略：如果解析失败，默认为 CHAT
            return {"intent": "CHAT"}
        
        # 4. 返回结果
        return {"intent": result.category}

router_node = RouterNode()
