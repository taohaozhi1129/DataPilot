from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
from app.services.llm_service import LLMService
from app.agents.state import AgentState
from app.core.prompts import ROUTER_SYSTEM_PROMPT
from app.core.intents import get_intent_options, get_few_shot_examples

# 定义结构化输出，包含推理过程 (Chain of Thought)
class RouterOutput(BaseModel):
    thought: str = Field(
        description="思考过程。简要分析用户的输入，判断其核心意图，并解释分类理由。"
    )
    category: str = Field(
        description="用户意图的分类。必须是已注册的意图名称之一 (如 SQL_QUERY, TASK_CREATE, CHAT)。"
    )

class RouterNode:
    def __init__(self):
        self.llm_service = LLMService()
        self.llm = self.llm_service.get_llm()

    def __call__(self, state: AgentState):
        messages = state['messages']
        last_message = messages[-1]
        user_input = last_message.content if hasattr(last_message, 'content') else str(last_message)

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
            ("user", "{input}")
        ])
        
        # 使用 partial 注入 format_instructions，这样其中的花括号不会被再次解析
        prompt = prompt.partial(format_instructions=parser.get_format_instructions())
        
        chain = prompt | self.llm | parser
        try:
            result = chain.invoke({"input": user_input})
        except Exception:
            return {"intent": "CHAT"}
        
        # 4. 返回结果
        return {"intent": result.category}

router_node = RouterNode()
