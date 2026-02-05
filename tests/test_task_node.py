from types import SimpleNamespace
from app.agents.nodes.task_node import TaskNode, TaskConfiguration


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
    def __init__(self, templates):
        self._templates = templates

    def search_templates(self, _query, limit=1):
        return self._templates[:limit]


def test_task_node_merges_params_and_ready():
    rag = DummyRagService([{"content": "mysql->doris", "payload": {"src": "mysql", "dst": "doris"}}])
    result = TaskConfiguration(
        task_json={"src": "mysql", "dst": "doris", "table": "t1"},
        extracted_params={"table": "t1"},
        missing_params=[],
        explanation="ok",
    )
    node = TaskNode(rag_service=rag, llm_service=DummyLLMService(result))

    state = {"messages": [SimpleNamespace(content="同步表 t1 到 doris")]}
    output = node(state)

    assert output["active_task"]["status"] == "ready"
    assert output["active_task"]["collected_params"]["table"] == "t1"
    assert "```json" in output["final_output"]


def test_task_node_collecting_when_missing_params():
    rag = DummyRagService([{"content": "mysql->doris", "payload": {"src": "mysql", "dst": "doris"}}])
    result = TaskConfiguration(
        task_json=None,
        extracted_params={},
        missing_params=["目标数据库"],
        explanation="need target",
    )
    node = TaskNode(rag_service=rag, llm_service=DummyLLMService(result))

    state = {"messages": [SimpleNamespace(content="同步数据")]}
    output = node(state)

    assert output["active_task"]["status"] == "collecting"
    assert "目标数据库" in output["final_output"]
