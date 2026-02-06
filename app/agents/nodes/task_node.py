from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import json
import logging

from app.services.rag_service import RagService
from app.services.llm_service import LLMService
from app.agents.state import AgentState, TaskContext
from app.core.prompts import TASK_GEN_SYSTEM_PROMPT
from app.agents.utils.history import split_history_and_input
from config import settings

logger = logging.getLogger(__name__)

# 任务配置的结构化输出
class TaskConfiguration(BaseModel):
    """
    任务配置生成的结构化输出。
    """
    task_json: Optional[Dict[str, Any]] = Field(description="生成的 JSON 配置对象。如果信息缺失则为 None")
    extracted_params: Optional[Dict[str, Any]] = Field(description="从用户输入中抽取到的参数键值对")
    missing_params: List[str] = Field(description="缺失的必要参数列表，例如 ['目标数据库', '同步周期']")
    explanation: str = Field(description="配置说明或对用户的反问")

class TaskNode:
    """
    任务生成节点 (Task Node)。
    负责根据用户需求，检索任务模板，并进行多轮对话以补全任务参数。
    """
    def __init__(self, rag_service: Optional[RagService] = None, llm_service: Optional[LLMService] = None):
        self.rag_service = rag_service or RagService()
        self.llm = (llm_service or LLMService()).get_llm()

    def __call__(self, state: AgentState):
        """
        执行任务配置生成逻辑。
        
        Args:
            state (AgentState): 当前 Agent 的状态
            
        Returns:
            dict: 更新后的状态，包含 messages, retrieved_context, active_task 和 final_output
        """
        messages = state['messages']
        logger.debug(f"TaskNode processing state with {len(messages)} messages.")
        
        history, user_query = split_history_and_input(
            messages,
            max_history=settings.HISTORY_MAX_MESSAGES,
        )
        summary = state.get("conversation_summary") or ""
        active_task = state.get('active_task')
        
        logger.info(f"Processing task request for input: {user_query[:50]}...")
        
        # 1. 确定上下文 (RAG or Existing Context)
        if active_task and active_task['status'] == 'collecting':
            # 处于多轮对话中，复用之前的模板
            retrieved_template = active_task['template']
            logger.info(f"Resuming task with template: {retrieved_template.get('content')}")
        else:
            # 新任务，执行检索
            try:
                templates = self.rag_service.search_templates(user_query)
                logger.info(f"Retrieved {len(templates)} templates from RAG.")
            except Exception as exc:
                logger.error(f"RAG search failed: {exc}", exc_info=True)
                msg = f"抱歉，模板检索服务暂时不可用：{exc}"
                return {
                    "messages": [AIMessage(content=msg)],
                    "final_output": msg
                }
            if not templates:
                logger.warning("No templates found for user query.")
                msg = "抱歉，我没有找到相关的任务模板。请尝试描述得更具体一些，例如'MySQL同步到Doris'。"
                return {
                    "messages": [AIMessage(content=msg)],
                    "final_output": msg
                }
            retrieved_template = templates[0] # Top 1
            logger.info(f"Started new task with template: {retrieved_template.get('content')}")

        # 2. 构建 Prompt
        existing_params_str = ""
        if active_task:
            existing_params_str = f"\n[Context] 已知信息: {json.dumps(active_task.get('collected_params', {}), ensure_ascii=False)}"

        prompt = ChatPromptTemplate.from_messages([
            ("system", TASK_GEN_SYSTEM_PROMPT + existing_params_str),
            ("system", "对话摘要（如有）: {summary}"),
            MessagesPlaceholder("history", optional=True),
            ("user", "{input}")
        ])
        
        # 3. 结构化抽取
        structured_llm = self.llm.with_structured_output(TaskConfiguration)
        chain = prompt | structured_llm
        
        # 这里 user_query 可能是 "我要同步数据" (第一轮) 也可能是 "目标是 Doris" (第二轮)
        try:
            logger.info("Invoking LLM for task configuration extraction...")
            result = chain.invoke({
                "retrieved_template": f"描述: {retrieved_template['content']}\nPayload: {retrieved_template['payload']}",
                "history": history,
                "input": user_query,
                "summary": summary,
            })
            logger.info(f"Task extraction result (Missing params: {result.missing_params})")
        except Exception as exc:
            logger.error(f"Task generation failed: {exc}", exc_info=True)
            msg = f"抱歉，任务生成暂时不可用：{exc}"
            return {
                "messages": [AIMessage(content=msg)],
                "final_output": msg
            }
        
        # 4. 更新状态与逻辑处理
        collected_params: Dict[str, Any] = {}
        if active_task:
            collected_params.update(active_task.get("collected_params", {}))
        if result.extracted_params:
            collected_params.update(result.extracted_params)

        new_active_task = {
            "template": retrieved_template,
            "collected_params": collected_params,
            "missing_params": result.missing_params,
            "status": "collecting" if result.missing_params else "ready"
        }
        
        if result.missing_params:
            final_msg = f"为了为您构建任务，我还需要以下信息：{', '.join(result.missing_params)}。\n\n{result.explanation}"
        else:
            final_msg = f"任务配置已生成：\n```json\n{json.dumps(result.task_json, ensure_ascii=False, indent=2)}\n```\n\n说明: {result.explanation}"
            
        return {
            "messages": [AIMessage(content=final_msg)],
            "retrieved_context": {"template": retrieved_template},
            "active_task": new_active_task,
            "final_output": final_msg
        }

task_node = TaskNode()
