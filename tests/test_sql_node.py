from types import SimpleNamespace
from app.agents.nodes.sql_node import SQLNode, SQLOutput


class DummyLLM:
    def __init__(self, result):
        self._result = result

    def with_structured_output(self, _model):
        return self

    def invoke(self, _inputs):
        return self._result


class DummyLLMService:
    def __init__(self, result):
        self._result = result

    def get_llm(self):
        return DummyLLM(self._result)


class DummyRagService:
    def __init__(self, schemas):
        self._schemas = schemas

    def search_schemas(self, _query, limit=5):
        return self._schemas[:limit]


def test_sql_node_blocks_unsafe_sql():
    rag = DummyRagService([{"content": "users(id, name)", "metadata": "{}"}])
    result = SQLOutput(sql="DELETE FROM users", explanation="bad", is_safe=True)
    node = SQLNode(rag_service=rag, llm_service=DummyLLMService(result))

    state = {"messages": [SimpleNamespace(content="delete users")]}
    output = node(state)

    assert "拦截" in output["final_output"]


def test_sql_node_returns_sql_when_safe():
    rag = DummyRagService([{"content": "users(id, name)", "metadata": "{}"}])
    result = SQLOutput(sql="SELECT * FROM users", explanation="ok", is_safe=True)
    node = SQLNode(rag_service=rag, llm_service=DummyLLMService(result))

    state = {"messages": [SimpleNamespace(content="查询用户")]}
    output = node(state)

    assert "```sql" in output["final_output"]

