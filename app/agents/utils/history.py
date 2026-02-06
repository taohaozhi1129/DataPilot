from typing import List, Tuple
from langchain_core.messages import BaseMessage, HumanMessage

def split_history_and_input(
    messages: List[BaseMessage],
    max_history: int,
) -> Tuple[List[BaseMessage], str]:
    """
    分离对话历史和当前用户输入，并根据最大历史条数限制上下文长度。
    
    Args:
        messages (List[BaseMessage]): 完整的消息列表
        max_history (int): 保留的最大历史消息数
        
    Returns:
        Tuple[List[BaseMessage], str]: (截断后的历史消息列表, 当前用户输入内容)
    """
    if not messages:
        return [], ""

    last = messages[-1]
    # 通常最后一条消息是 HumanMessage，作为本次的 input
    if isinstance(last, HumanMessage):
        current_input = last.content
        history = messages[:-1]
    else:
        # 兼容性处理，以防最后一条不是 HumanMessage
        current_input = last.content if hasattr(last, "content") else str(last)
        history = messages[:-1]

    # 截断历史，保留最近的 N 条
    if max_history and max_history > 0 and len(history) > max_history:
        history = history[-max_history:]

    return history, current_input
