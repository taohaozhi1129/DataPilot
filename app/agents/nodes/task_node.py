from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field
from app.services.rag_service import RagService
from app.services.llm_service import LLMService
from app.agents.state import AgentState, TaskContext
from app.core.prompts import TASK_GEN_SYSTEM_PROMPT
from typing import Optional, List, Dict, Any
import json

# 任务配置的结构化输出
class TaskConfiguration(BaseModel):
    task_json: Optional[Dict[str, Any]] = Field(description="生成的 JSON 配置对象。如果信息缺失则为 None")
    extracted_params: Optional[Dict[str, Any]] = Field(description="从用户输入中抽取到的参数键值对")
    missing_params: List[str] = Field(description="缺失的必要参数列表，例如 ['目标数据库', '同步周期']")
    explanation: str = Field(description="配置说明或对用户的反问")

class TaskNode:
    def __init__(self, rag_service: Optional[RagService] = None, llm_service: Optional[LLMService] = None):
        self.rag_service = rag_service or RagService()
        self.llm = (llm_service or LLMService()).get_llm()

    def __call__(self, state: AgentState):
        messages = state['messages']
        user_query = messages[-1].content
        active_task = state.get('active_task')
        
        # 1. 确定上下文 (RAG or Existing Context)
        if active_task and active_task['status'] == 'collecting':
            # 处于多轮对话中，复用之前的模板
            retrieved_template = active_task['template']
            print(f"[TaskNode] Resuming task with template: {retrieved_template.get('content')}")
        else:
            # 新任务，执行检索
            try:
                templates = self.rag_service.search_templates(user_query)
            except Exception as exc:
                msg = f"抱歉，模板检索服务暂时不可用：{exc}"
                return {
                    "messages": [AIMessage(content=msg)],
                    "final_output": msg
                }
            if not templates:
                msg = "抱歉，我没有找到相关的任务模板。请尝试描述得更具体一些，例如'MySQL同步到Doris'。"
                return {
                    "messages": [AIMessage(content=msg)],
                    "final_output": msg
                }
            retrieved_template = templates[0] # Top 1
            print(f"[TaskNode] Started new task with template: {retrieved_template.get('content')}")

        # 2. 构建 Prompt
        # 注意：这里我们简单地将用户当前输入作为 user_query。
        # 在多轮对话中，理想情况下应该把收集到的参数历史也告诉 LLM，或者让 LLM 看整个 messages 历史。
        # 由于 LangGraph 传递了 messages 列表，Prompt Template 最好能包含历史，但这里为了保持 Prompt 简单，
        # 我们依赖 LLM 自身的能力去理解 "user_query" (如果是追加信息)。
        # 为了增强效果，我们可以把 "已收集的信息" 注入到 Prompt 中。
        
        existing_params_str = ""
        if active_task:
            existing_params_str = f"\n[Context] 已知信息: {json.dumps(active_task.get('collected_params', {}), ensure_ascii=False)}"

        prompt = ChatPromptTemplate.from_messages([
            ("system", TASK_GEN_SYSTEM_PROMPT + existing_params_str),
            ("user", "{user_query}")
        ])
        
        # 3. 结构化抽取
        structured_llm = self.llm.with_structured_output(TaskConfiguration)
        chain = prompt | structured_llm
        
        # 这里 user_query 可能是 "我要同步数据" (第一轮) 也可能是 "目标是 Doris" (第二轮)
        try:
            result = chain.invoke({
                "retrieved_template": f"描述: {retrieved_template['content']}\nPayload: {retrieved_template['payload']}",
                "user_query": user_query
            })
        except Exception as exc:
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
