from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field
from typing import Optional
import json
import logging

from app.services.rag_service import RagService
from app.services.llm_service import LLMService
from app.agents.state import AgentState
from app.core.prompts import SQL_GEN_SYSTEM_PROMPT
from app.utils.sql_safety import is_safe_sql
from app.agents.utils.history import split_history_and_input
from config import settings

logger = logging.getLogger(__name__)

# 定义结构化输出模型
class SQLOutput(BaseModel):
    """
    Text-to-SQL 生成结果模型。
    """
    sql: str = Field(description="生成的 SQL 查询语句")
    explanation: str = Field(description="对 SQL 逻辑的简要解释")
    is_safe: bool = Field(description="SQL 是否安全（无删除/修改操作）")

class SQLNode:
    """
    SQL 生成节点 (SQL Node)。
    负责根据用户查询和检索到的元数据生成安全的 SQL 语句。
    """
    def __init__(self, rag_service: Optional[RagService] = None, llm_service: Optional[LLMService] = None):
        self.rag_service = rag_service or RagService()
        self._llm_service = llm_service
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            service = self._llm_service or LLMService()
            self._llm = service.get_llm()
            self._llm_service = service
        return self._llm

    def __call__(self, state: AgentState):
        """
        执行 SQL 生成逻辑。
        
        Args:
            state (AgentState): 当前 Agent 的状态
            
        Returns:
            dict: 更新后的状态，包含 messages, retrieved_context 和 final_output
        """
        messages = state['messages']
        logger.debug(f"SQLNode processing state with {len(messages)} messages.")
        
        history, user_query = split_history_and_input(
            messages,
            max_history=settings.HISTORY_MAX_MESSAGES,
        )
        summary = state.get("conversation_summary") or ""
        
        logger.info(f"Processing SQL query for input: {user_query[:50]}...")

        # 1. RAG 检索
        try:
            schemas = self.rag_service.search_schemas(user_query)
            logger.info(f"Retrieved {len(schemas)} schemas from RAG.")
        except Exception as exc:
            logger.error(f"RAG search failed: {exc}", exc_info=True)
            msg = f"抱歉，检索服务暂时不可用：{exc}"
            return {
                "messages": [AIMessage(content=msg)],
                "final_output": msg
            }
        
        if not schemas:
            logger.warning("No schemas found for user query.")
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
            ("system", "对话摘要（如有）: {summary}"),
            MessagesPlaceholder("history", optional=True),
            ("user", "{input}")
        ])
        
        # 3. 使用结构化输出
        structured_llm = self._get_llm().with_structured_output(SQLOutput)
        chain = prompt | structured_llm
        
        try:
            logger.info("Invoking LLM for SQL generation...")
            result = chain.invoke({
                "retrieved_schemas": formatted_schemas,
                "history": history,
                "input": user_query,
                "summary": summary,
            })
            logger.info(f"SQL generated (Safe: {result.is_safe}): {result.sql[:50]}...")
        except Exception as exc:
            logger.error(f"SQL generation failed: {exc}", exc_info=True)
            msg = f"抱歉，SQL 生成暂时不可用：{exc}"
            return {
                "messages": [AIMessage(content=msg)],
                "final_output": msg
            }
        
        # 4. 后处理与简单的校验逻辑 (Self-Correction 雏形)
        if not result.is_safe or not is_safe_sql(result.sql):
            logger.warning(f"Unsafe SQL detected and blocked: {result.sql}")
            final_msg = f"检测到生成的 SQL 包含非安全操作，已被拦截。\n解释: {result.explanation}"
        else:
            final_msg = f"```sql\n{result.sql}\n```\n\n**解释**: {result.explanation}"
        
        return {
            "messages": [AIMessage(content=final_msg)],
            "retrieved_context": {"schemas": schemas},
            "final_output": final_msg
        }

sql_node = SQLNode()
