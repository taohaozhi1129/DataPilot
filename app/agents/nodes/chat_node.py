from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import AIMessage
from app.services.llm_service import LLMService
from app.agents.state import AgentState

class ChatNode:
    def __init__(self):
        self.llm = LLMService().get_llm()

    def __call__(self, state: AgentState):
        messages = state['messages']
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "你是一个友好的AI助手。"),
            ("user", "{input}")
        ])
        
        chain = prompt | self.llm
        # 假设 messages 是 BaseMessage 或类似的列表，我们获取最后一条的内容
        # 或者如果使用 LangChain 的历史感知机制，则传递整个历史
        # 为了简单起见，仅使用最后一条消息内容
        last_message = messages[-1].content
        
        try:
            response = chain.invoke({"input": last_message})
        except Exception as exc:
            msg = f"抱歉，当前对话服务不可用：{exc}"
            return {
                "messages": [AIMessage(content=msg)],
                "final_output": msg,
                "retrieved_context": {}
            }
        
        return {
            "messages": [AIMessage(content=response.content)],
            "final_output": response.content,
            "retrieved_context": {}
        }

chat_node = ChatNode()
