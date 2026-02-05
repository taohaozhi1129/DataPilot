from langgraph.graph import StateGraph, END
from app.agents.state import AgentState
from app.agents.nodes.router import router_node
from app.agents.nodes.sql_node import sql_node
from app.agents.nodes.task_node import task_node
from app.agents.nodes.chat_node import chat_node
from app.core.memory.redis_saver import AsyncRedisSaver
from config import settings

def route_decision(state: AgentState):
    intent = state['intent']
    if intent == 'SQL_QUERY':
        return 'sql_node'
    elif intent == 'TASK_CREATE':
        return 'task_node'
    else:
        return 'chat_node'

# 构建图
workflow = StateGraph(AgentState)

# 添加节点
workflow.add_node("router", router_node)
workflow.add_node("sql_node", sql_node)
workflow.add_node("task_node", task_node)
workflow.add_node("chat_node", chat_node)

# 添加边
workflow.set_entry_point("router")

workflow.add_conditional_edges(
    "router",
    route_decision,
    {
        "sql_node": "sql_node",
        "task_node": "task_node",
        "chat_node": "chat_node"
    }
)

workflow.add_edge("sql_node", END)
workflow.add_edge("task_node", END)
workflow.add_edge("chat_node", END)

# 编译并添加持久化 Checkpointer
# 强制使用 Redis，移除 SQLite 支持
checkpointer = AsyncRedisSaver.from_url(settings.REDIS_URL)
print(f"Using Redis Checkpointer at {settings.REDIS_URL}")

graph = workflow.compile(checkpointer=checkpointer)
