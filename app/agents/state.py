from typing import TypedDict, Annotated, List, Literal, Dict, Any, Optional
import operator
from langchain_core.messages import BaseMessage

class TaskContext(TypedDict):
    template: Dict[str, Any]
    collected_params: Dict[str, Any]
    missing_params: List[str]
    status: Literal['collecting', 'ready']

class AgentState(TypedDict):
    # 使用 BaseMessage 确保兼容性，add 操作符用于追加消息历史
    messages: Annotated[List[BaseMessage], operator.add]
    
    intent: Literal['SQL_QUERY', 'TASK_CREATE', 'CHAT', 'UNKNOWN']
    
    # 通用的检索上下文（单轮使用）
    retrieved_context: Dict[str, Any] 
    
    # 专门用于多轮任务填槽的上下文 (Persistence)
    # 使用覆盖更新策略 (默认行为)，即新值覆盖旧值
    active_task: Optional[TaskContext]
    
    final_output: str
