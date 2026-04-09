from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Dict, Optional

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

from app.agents.state import AgentState
from app.agents.utils.history import split_history_and_input
from app.core.prompts import SQL_GEN_SYSTEM_PROMPT
from app.services.llm_service import LLMService
from app.services.rag_service import RagService
from app.utils.sql_safety import is_safe_sql
from config import settings

logger = logging.getLogger(__name__)


class SQLOutput(BaseModel):
    sql: str = Field(description="生成的 SQL 查询语句")
    explanation: str = Field(description="SQL 逻辑说明")
    is_safe: bool = Field(description="模型判断 SQL 是否安全")


class SQLNode:
    def __init__(
        self,
        rag_service: Optional[RagService] = None,
        llm_service: Optional[LLMService] = None,
    ) -> None:
        self.rag_service = rag_service or RagService()
        self._llm_service = llm_service
        self._llm = None

    def _get_llm(self):
        if self._llm is None:
            service = self._llm_service or LLMService()
            self._llm = service.get_llm()
            self._llm_service = service
        return self._llm

    @staticmethod
    def _format_sql_context(sql_context: Dict) -> str:
        tables = sql_context.get("tables", [])
        columns = sql_context.get("columns", [])
        columns_by_table = defaultdict(list)
        for column in columns:
            columns_by_table[column.get("table_id", "")].append(column)

        rendered_tables = []
        for table in tables:
            table_lines = [
                f"Table: {table.get('full_table_name') or table.get('table_name', '')}",
                f"Type: {table.get('table_type', '')}",
                f"Dialect: {table.get('dialect', '')}",
                f"Business Name: {table.get('business_name', '')}",
                f"Business Domain: {table.get('business_domain', '')}",
                f"Description: {table.get('table_desc', '')}",
                f"Grain: {table.get('grain_desc', '')}",
                f"Primary Keys: {json.dumps(table.get('primary_keys', []), ensure_ascii=False)}",
                f"Partition Keys: {json.dumps(table.get('partition_keys', []), ensure_ascii=False)}",
                f"Time Columns: {json.dumps(table.get('time_columns', []), ensure_ascii=False)}",
                f"Join Hints: {json.dumps(table.get('join_hints', []), ensure_ascii=False)}",
            ]
            related_columns = columns_by_table.get(table.get("doc_id", ""), [])
            if related_columns:
                table_lines.append("Columns:")
                for column in related_columns[:8]:
                    table_lines.append(
                        "  - {name} | type={dtype} | semantic={semantic} | metric={metric} | desc={desc}".format(
                            name=column.get("full_column_name") or column.get("column_name", ""),
                            dtype=column.get("data_type", ""),
                            semantic=column.get("semantic_type", ""),
                            metric=column.get("metric_role", ""),
                            desc=column.get("column_desc", ""),
                        )
                    )
            rendered_tables.append("\n".join(table_lines))

        return "\n\n".join(rendered_tables)

    def __call__(self, state: AgentState):
        messages = state["messages"]
        logger.debug("SQLNode processing state with %s messages.", len(messages))

        history, user_query = split_history_and_input(
            messages,
            max_history=settings.HISTORY_MAX_MESSAGES,
        )
        summary = state.get("conversation_summary") or ""

        logger.info("Processing SQL query for input: %s...", user_query[:50])

        try:
            sql_context = self.rag_service.search_sql_context(user_query)
            tables = sql_context.get("tables", [])
            columns = sql_context.get("columns", [])
            logger.info("Retrieved SQL context. tables=%s columns=%s", len(tables), len(columns))
        except Exception as exc:
            logger.error("RAG search failed: %s", exc, exc_info=True)
            msg = f"抱歉，检索服务暂时不可用：{exc}"
            return {
                "messages": [AIMessage(content=msg)],
                "final_output": msg,
            }

        if not sql_context.get("tables"):
            logger.warning("No tables found for user query.")
            msg = "抱歉，我在知识库中未找到与您查询相关的数据表资产。请确认元数据是否已准备完成。"
            return {
                "messages": [AIMessage(content=msg)],
                "final_output": msg,
            }

        formatted_schemas = self._format_sql_context(sql_context)

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", SQL_GEN_SYSTEM_PROMPT),
                ("system", "对话摘要（如有）: {summary}"),
                MessagesPlaceholder("history", optional=True),
                ("user", "{input}"),
            ]
        )

        structured_llm = self._get_llm().with_structured_output(
            SQLOutput,
            method="function_calling",
        )

        try:
            logger.info("Invoking LLM for SQL generation...")
            prompt_value = prompt.invoke(
                {
                    "retrieved_schemas": formatted_schemas,
                    "history": history,
                    "input": user_query,
                    "summary": summary,
                }
            )
            result = structured_llm.invoke(prompt_value)
            logger.info("SQL generated (safe=%s): %s...", result.is_safe, result.sql[:50])
        except Exception as exc:
            logger.error("SQL generation failed: %s", exc, exc_info=True)
            msg = f"抱歉，SQL 生成暂时不可用：{exc}"
            return {
                "messages": [AIMessage(content=msg)],
                "final_output": msg,
            }

        if not result.is_safe or not is_safe_sql(result.sql):
            logger.warning("Unsafe SQL detected and blocked: %s", result.sql)
            final_msg = f"检测到生成的 SQL 包含非安全操作，已被拦截。\n解释: {result.explanation}"
        else:
            final_msg = f"```sql\n{result.sql}\n```\n\n**解释**: {result.explanation}"

        return {
            "messages": [AIMessage(content=final_msg)],
            "retrieved_context": sql_context,
            "final_output": final_msg,
        }


sql_node = SQLNode()
