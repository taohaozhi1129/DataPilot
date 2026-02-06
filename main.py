from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import List, Optional
from langchain_core.messages import HumanMessage
from app.agents.graph import graph
from config import settings
from app.core.logger import setup_logging
import uuid
import logging

# 初始化日志
logger = setup_logging()

app = FastAPI(
    title="DataPilot AI Platform",
    description="""
    **赋能数据中台的下一代 AI 智能体基石**
    
    核心能力 (Core Capabilities):
    - 🧠 **智能意图识别 (Intent Reasoning)**: 基于思维链 (CoT) 的精准意图分发，不仅听懂指令，更能理解业务。
    - 📊 **Text-to-SQL (RAG)**: 融合向量检索技术，自动关联元数据，生成精准、安全的数据查询语句。
    - ⚙️ **Text-to-Task (Automation)**: 智能解析自然语言，自动构建数据同步与 ETL 任务配置，实现从“说话”到“落地”的自动化。
    - 🔄 **多轮交互 (Context Aware)**: 支持长对话记忆与槽位填充 (Slot Filling)，主动追问缺失信息，确保任务完整执行。
    """,
    version="1.0.0"
)

class ChatRequest(BaseModel):
    query: str = Field(..., description="用户的自然语言指令，例如：'帮我查一下上个月的销售额'")
    thread_id: Optional[str] = Field(default=None, description="会话 ID，用于多轮对话上下文保持")

class ChatResponse(BaseModel):
    response: str = Field(..., description="Agent 的回复内容")
    thread_id: str = Field(..., description="当前会话的 ID")
    intent: Optional[str] = Field(None, description="识别出的用户意图 (SQL_QUERY / TASK_CREATE / CHAT)")
    context: Optional[dict] = Field(None, description="当前的上下文状态数据")
    missing_params: Optional[List[str]] = Field(None, description="当前任务缺失的参数列表")

@app.post("/chat", response_model=ChatResponse, summary="与 AI Data Agent 进行对话")
async def chat_endpoint(request: ChatRequest):
    """
    核心对话接口。
    
    发送自然语言指令，Agent 会自动：
    1. 识别意图
    2. 检索必要的元数据或模板
    3. 生成 SQL 或 任务配置
    4. 如果信息缺失，会主动反问
    """
    # 如果没有 thread_id，生成一个新的
    thread_id = request.thread_id or str(uuid.uuid4())
    
    # 构造 LangGraph 的输入状态
    initial_state = {
        "messages": [HumanMessage(content=request.query)]
    }
    
    # 配置线程上下文（用于 MemorySaver / RedisSaver）
    config = {"configurable": {"thread_id": thread_id}}
    
    # 执行图 (Graph)
    # 这里的 graph 已经配置了 AsyncRedisSaver，所以状态会自动持久化
    logger.info(f"Invoking agent graph for thread_id: {thread_id} with query: {request.query}")
    try:
        final_state = await graph.ainvoke(initial_state, config=config)
    except Exception as e:
        logger.error(f"Error executing agent graph: {e}", exc_info=True)
        return ChatResponse(
            response="抱歉，系统内部发生错误，请稍后再试。",
            thread_id=thread_id,
            intent="ERROR"
        )
    
    # 从最终状态中提取回复
    messages = final_state.get('messages', [])
    ai_response = messages[-1].content if messages else "抱歉，我没有理解您的指令。"
    
    # 提取其他元数据
    intent = final_state.get('intent')
    logger.info(f"Agent execution completed. Intent: {intent}")
    active_task = final_state.get('active_task', {})
    
    # 检查是否有缺失参数
    missing = []
    if active_task and active_task.get('missing_params'):
        missing = active_task['missing_params']

    return ChatResponse(
        response=ai_response,
        thread_id=thread_id,
        intent=intent,
        context=active_task, # 返回当前任务上下文供前端展示
        missing_params=missing if missing else None
    )

@app.get("/health", summary="健康检查")
async def health_check():
    return {"status": "ok", "version": "1.0.0", "backend": "DataCopilot AI"}
