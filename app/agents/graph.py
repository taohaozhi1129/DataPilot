from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
import logging
import socket

from app.agents.state import AgentState
from app.agents.nodes.router import router_node
from app.agents.nodes.sql_node import sql_node
from app.agents.nodes.task_node import task_node
from app.agents.nodes.chat_node import chat_node
from app.agents.nodes.memory_node import memory_node
from app.core.memory.redis_saver import AsyncRedisSaver
from config import settings

logger = logging.getLogger(__name__)


def _can_connect_redis() -> bool:
    """Fast pre-flight check to avoid hard failing startup when Redis is unavailable."""
    try:
        with socket.create_connection(
            (settings.REDIS_HOST, settings.REDIS_PORT),
            timeout=settings.REDIS_CONNECT_TIMEOUT_SECONDS,
        ):
            return True
    except OSError:
        return False


def _build_checkpointer():
    """Prefer Redis in production, but gracefully degrade to in-memory saver."""
    if not settings.ENABLE_REDIS_CHECKPOINTER:
        logger.warning("Redis checkpointer disabled by configuration. Using in-memory checkpointer.")
        return MemorySaver()

    if not _can_connect_redis():
        logger.warning(
            "Redis is unreachable at %s. Using in-memory checkpointer.",
            settings.REDIS_URL,
        )
        return MemorySaver()

    logger.info(f"Initializing Redis Checkpointer at {settings.REDIS_URL}")
    try:
        return AsyncRedisSaver.from_url(settings.REDIS_URL)
    except Exception as e:
        logger.warning("Failed to initialize Redis checkpointer, fallback to memory: %s", e)
        return MemorySaver()

def route_decision(state: AgentState):
    """
    根据 Router 节点的输出决定下一步跳转的节点。
    """
    intent = state['intent']
    logger.info(f"Routing to next node based on intent: {intent}")
    if intent == 'SQL_QUERY':
        return 'sql_node'
    elif intent == 'TASK_CREATE':
        return 'task_node'
    else:
        return 'chat_node'

# 构建图 (StateGraph)
# -------------------------------------------------------------------------
workflow = StateGraph(AgentState)

# 添加节点
# -------------------------------------------------------------------------
logger.debug("Adding nodes to the graph...")
workflow.add_node("router", router_node)
workflow.add_node("sql_node", sql_node)
workflow.add_node("task_node", task_node)
workflow.add_node("chat_node", chat_node)
workflow.add_node("memory_node", memory_node)

# 添加边 (Edges)
# -------------------------------------------------------------------------
logger.debug("Configuring edges...")
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

workflow.add_edge("sql_node", "memory_node")
workflow.add_edge("task_node", "memory_node")
workflow.add_edge("chat_node", "memory_node")
workflow.add_edge("memory_node", END)

# 编译并添加持久化 Checkpointer
# -------------------------------------------------------------------------
checkpointer = _build_checkpointer()

graph = workflow.compile(checkpointer=checkpointer)
logger.info("Agent Graph compiled successfully.")
