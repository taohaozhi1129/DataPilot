from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from pydantic import BaseModel, Field

from app.agents.state import AgentState
from app.agents.utils.history import split_history_and_input
from app.core.prompts import TASK_GEN_SYSTEM_PROMPT
from app.services.llm_service import LLMService
from app.services.rag_service import RagService
from config import settings

logger = logging.getLogger(__name__)


class TaskConfiguration(BaseModel):
    task_json: Optional[Dict[str, Any]] = Field(description="生成的 JSON 配置对象，如信息缺失则为 null")
    extracted_params: Optional[Dict[str, Any]] = Field(description="从输入中抽取出的参数")
    missing_params: List[str] = Field(description="仍然缺失的必要参数")
    explanation: str = Field(description="对配置逻辑的说明或追问")


class TaskNode:
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
    def _format_template_candidates(templates: List[Dict[str, Any]]) -> str:
        chunks = []
        for idx, template in enumerate(templates, start=1):
            chunks.append(
                "\n".join(
                    [
                        f"[Template {idx}]",
                        f"Code: {template.get('template_code', '')}",
                        f"Name: {template.get('template_name', '')}",
                        f"Type: {template.get('template_type', '')}",
                        f"Domain: {template.get('business_domain', '')}",
                        f"Description: {template.get('template_desc', template.get('content', ''))}",
                        f"Required Slots: {json.dumps(template.get('required_slots', []), ensure_ascii=False)}",
                        f"Optional Slots: {json.dumps(template.get('optional_slots', []), ensure_ascii=False)}",
                        f"Slot Schema: {json.dumps(template.get('slot_schema', {}), ensure_ascii=False)}",
                        f"Compatibility: {json.dumps(template.get('compatibility_rules', {}), ensure_ascii=False)}",
                        f"Default Payload: {json.dumps(template.get('default_payload', template.get('payload', {})), ensure_ascii=False)}",
                    ]
                )
            )
        return "\n\n".join(chunks)

    def __call__(self, state: AgentState):
        messages = state["messages"]
        logger.debug("TaskNode processing state with %s messages.", len(messages))

        history, user_query = split_history_and_input(
            messages,
            max_history=settings.HISTORY_MAX_MESSAGES,
        )
        summary = state.get("conversation_summary") or ""
        active_task = state.get("active_task")

        logger.info("Processing task request for input: %s...", user_query[:50])

        if active_task and active_task["status"] == "collecting":
            retrieved_template = active_task["template"]
            template_prompt_text = self._format_template_candidates([retrieved_template])
            logger.info("Resuming task with template: %s", retrieved_template.get("template_code"))
        else:
            try:
                templates = self.rag_service.search_templates(user_query, limit=3)
                logger.info("Retrieved %s templates from RAG.", len(templates))
            except Exception as exc:
                logger.error("RAG search failed: %s", exc, exc_info=True)
                msg = f"抱歉，模板检索服务暂时不可用：{exc}"
                return {
                    "messages": [AIMessage(content=msg)],
                    "final_output": msg,
                }

            if not templates:
                logger.warning("No templates found for user query.")
                msg = "抱歉，我没有找到相关的任务模板。请尝试描述得更具体一些，例如“从 MySQL 同步到 Doris”。"
                return {
                    "messages": [AIMessage(content=msg)],
                    "final_output": msg,
                }

            retrieved_template = templates[0]
            template_prompt_text = self._format_template_candidates(templates)
            logger.info("Started new task with template: %s", retrieved_template.get("template_code"))

        existing_params_str = ""
        if active_task:
            existing_params_str = (
                "\n[Context] 已知信息: "
                + json.dumps(active_task.get("collected_params", {}), ensure_ascii=False)
            )

        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", TASK_GEN_SYSTEM_PROMPT + existing_params_str),
                ("system", "对话摘要（如有）: {summary}"),
                MessagesPlaceholder("history", optional=True),
                ("user", "{input}"),
            ]
        )

        structured_llm = self._get_llm().with_structured_output(
            TaskConfiguration,
            method="function_calling",
        )

        try:
            logger.info("Invoking LLM for task configuration extraction...")
            prompt_value = prompt.invoke(
                {
                    "retrieved_template": template_prompt_text,
                    "history": history,
                    "input": user_query,
                    "summary": summary,
                }
            )
            result = structured_llm.invoke(prompt_value)
            logger.info("Task extraction result. missing_params=%s", result.missing_params)
        except Exception as exc:
            logger.error("Task generation failed: %s", exc, exc_info=True)
            msg = f"抱歉，任务生成暂时不可用：{exc}"
            return {
                "messages": [AIMessage(content=msg)],
                "final_output": msg,
            }

        collected_params: Dict[str, Any] = {}
        if active_task:
            collected_params.update(active_task.get("collected_params", {}))
        if result.extracted_params:
            collected_params.update(result.extracted_params)

        new_active_task = {
            "template": retrieved_template,
            "collected_params": collected_params,
            "missing_params": result.missing_params,
            "status": "collecting" if result.missing_params else "ready",
        }

        if result.missing_params:
            final_msg = (
                f"为了为您构建任务，我还需要以下信息：{', '.join(result.missing_params)}。\n\n"
                f"{result.explanation}"
            )
        else:
            final_msg = (
                "任务配置已生成：\n```json\n"
                f"{json.dumps(result.task_json, ensure_ascii=False, indent=2)}\n"
                "```\n\n"
                f"说明: {result.explanation}"
            )

        return {
            "messages": [AIMessage(content=final_msg)],
            "retrieved_context": {"template": retrieved_template},
            "active_task": new_active_task,
            "final_output": final_msg,
        }


task_node = TaskNode()
