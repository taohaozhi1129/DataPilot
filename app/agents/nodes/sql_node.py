from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field
from app.services.rag_service import RagService
from app.services.llm_service import LLMService
from app.agents.state import AgentState
from app.core.prompts import SQL_GEN_SYSTEM_PROMPT
from app.utils.sql_safety import is_safe_sql
from typing import Optional
import json

# 定义结构化输出模型
class SQLOutput(BaseModel):
    sql: str = Field(description="生成的 SQL 查询语句")
    explanation: str = Field(description="对 SQL 逻辑的简要解释")
    is_safe: bool = Field(description="SQL 是否安全（无删除/修改操作）")

class SQLNode:
    def __init__(self, rag_service: Optional[RagService] = None, llm_service: Optional[LLMService] = None):
        self.rag_service = rag_service or RagService()
        self.llm = (llm_service or LLMService()).get_llm()

    def __call__(self, state: AgentState):
        messages = state['messages']
        user_query = messages[-1].content
        
        # 1. RAG 检索
        try:
            schemas = self.rag_service.search_schemas(user_query)
        except Exception as exc:
            msg = f"抱歉，检索服务暂时不可用：{exc}"
            return {
                "messages": [AIMessage(content=msg)],
                "final_output": msg
            }
        
        if not schemas:
            msg = "抱歉，我在知识库中未找到与您查询相关的数据表资产。请确认数据是否已录入元数据中心。"
            return {
                "messages": [AIMessage(content=msg)],
                "final_output": msg
            }
            
        # 格式化 schema 用于 prompt
        formatted_schemas = "\n".join([
            f"Table: {s.get('content', '')}\nMetadata: {json.dumps(s.get('metadata', {}), ensure_ascii=False)}"
            for s in schemas
        ])
        
        # 2. 构建 Prompt (使用集中管理的 Template)
        prompt = ChatPromptTemplate.from_messages([
            ("system", SQL_GEN_SYSTEM_PROMPT),
            ("user", "{user_query}")
        ])
        
        # 3. 使用结构化输出
        structured_llm = self.llm.with_structured_output(SQLOutput)
        chain = prompt | structured_llm
        
        try:
            result = chain.invoke({
                "retrieved_schemas": formatted_schemas,
                "user_query": user_query
            })
        except Exception as exc:
            msg = f"抱歉，SQL 生成暂时不可用：{exc}"
            return {
                "messages": [AIMessage(content=msg)],
                "final_output": msg
            }
        
        # 4. 后处理与简单的校验逻辑 (Self-Correction 雏形)
        if not result.is_safe or not is_safe_sql(result.sql):
            final_msg = f"检测到生成的 SQL 包含非安全操作，已被拦截。\n解释: {result.explanation}"
        else:
            final_msg = f"```sql\n{result.sql}\n```\n\n**解释**: {result.explanation}"
        
        return {
            "messages": [AIMessage(content=final_msg)],
            "retrieved_context": {"schemas": schemas},
            "final_output": final_msg
        }

sql_node = SQLNode()
