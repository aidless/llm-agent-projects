"""执行引擎测试"""

import pytest
from engine.executor import WorkflowExecutor, ExecutionStatus
from engine.dag import DAG
from engine.context import ExecutionContext


class TestExecutionContext:
    """执行上下文测试"""

    def test_set_and_get_output(self):
        ctx = ExecutionContext()
        ctx.set_output("node1", "text", "hello")
        assert ctx.get_output("node1", "text") == "hello"

    def test_get_all_outputs(self):
        ctx = ExecutionContext()
        ctx.set_outputs("node1", {"key1": "val1", "key2": "val2"})
        result = ctx.get_output("node1")
        assert result == {"key1": "val1", "key2": "val2"}

    def test_get_missing_node_raises(self):
        from engine.errors import VariableResolutionError
        ctx = ExecutionContext()
        with pytest.raises(VariableResolutionError):
            ctx.get_output("nonexistent")

    def test_get_missing_key_raises(self):
        from engine.errors import VariableResolutionError
        ctx = ExecutionContext()
        ctx.set_output("node1", "text", "hello")
        with pytest.raises(VariableResolutionError):
            ctx.get_output("node1", "missing_key")

    def test_resolve_template_single_var(self):
        ctx = ExecutionContext()
        ctx.set_output("node1", "text", "world")
        result = ctx.resolve("{{node1.output.text}}")
        assert result == "world"

    def test_resolve_template_mixed_text(self):
        ctx = ExecutionContext()
        ctx.set_output("node1", "name", "Alice")
        result = ctx.resolve("Hello, {{node1.output.name}}!")
        assert result == "Hello, Alice!"

    def test_resolve_template_nested_key(self):
        ctx = ExecutionContext()
        ctx.set_output("node1", "data", {"score": 42})
        result = ctx.resolve("{{node1.output.data.score}}")
        assert result == 42

    def test_resolve_env_var(self):
        ctx = ExecutionContext(env_vars={"API_KEY": "secret123"})
        result = ctx.resolve("{{env.API_KEY}}")
        assert result == "secret123"

    def test_resolve_dict(self):
        ctx = ExecutionContext()
        ctx.set_output("node1", "name", "Alice")
        ctx.set_output("node1", "age", 30)
        result = ctx.resolve_dict({
            "greeting": "Hi {{node1.output.name}}",
            "static": "no_vars",
        })
        assert result["greeting"] == "Hi Alice"
        assert result["static"] == "no_vars"

    def test_no_template_returns_original(self):
        ctx = ExecutionContext()
        assert ctx.resolve("plain text") == "plain text"
        assert ctx.resolve("") == ""
        assert ctx.resolve("123") == "123"

    def test_input_snapshot(self):
        ctx = ExecutionContext()
        ctx.set_input("node1", {"key": "value"})
        inp = ctx.get_input("node1")
        assert inp == {"key": "value"}

    def test_clear(self):
        ctx = ExecutionContext()
        ctx.set_output("node1", "key", "val")
        ctx.clear()
        from engine.errors import VariableResolutionError
        with pytest.raises(VariableResolutionError):
            ctx.get_output("node1")

    def test_snapshot(self):
        ctx = ExecutionContext()
        ctx.set_output("n1", "k", "v")
        snap = ctx.snapshot()
        assert "node_outputs" in snap
        assert "n1" in snap["node_outputs"]


class TestWorkflowExecutor:
    """工作流执行引擎测试"""

    def _make_executor(self):
        exec = WorkflowExecutor(max_workers=4, default_retry=0)

        def mock_handler(inputs, context, config):
            return {"result": inputs.get("value", "default")}

        exec.register_handler("mock", mock_handler)
        return exec

    def test_simple_chain_execution(self):
        exec = self._make_executor()
        dag = DAG.from_edges(["n1", "n2"], [("n1", "n2")])
        configs = {
            "n1": {"type": "mock", "inputs": {"value": "step1"}},
            "n2": {"type": "mock", "inputs": {"value": "step2"}},
        }
        result = exec.execute("wf1", dag, configs)
        assert result.status == ExecutionStatus.SUCCESS
        assert result.node_results["n1"].status == ExecutionStatus.SUCCESS
        assert result.node_results["n2"].status == ExecutionStatus.SUCCESS
        assert result.node_results["n1"].output["result"] == "step1"

    def test_parallel_branch_execution(self):
        exec = self._make_executor()
        dag = DAG()
        dag.add_node("root")
        dag.add_node("b1")
        dag.add_node("b2")
        dag.add_node("merge")
        dag.add_edge("root", "b1")
        dag.add_edge("root", "b2")
        dag.add_edge("b1", "merge")
        dag.add_edge("b2", "merge")
        configs = {
            "root": {"type": "mock", "inputs": {"value": "start"}},
            "b1": {"type": "mock", "inputs": {"value": "branch1"}},
            "b2": {"type": "mock", "inputs": {"value": "branch2"}},
            "merge": {"type": "mock", "inputs": {"value": "end"}},
        }
        result = exec.execute("wf2", dag, configs)
        assert result.status == ExecutionStatus.SUCCESS
        assert result.node_results["b1"].output["result"] == "branch1"
        assert result.node_results["b2"].output["result"] == "branch2"

    def test_invalid_dag_fails(self):
        exec = self._make_executor()
        dag = DAG()
        dag.add_node("a")
        dag.add_node("b")
        dag.add_edge("a", "b")
        dag.add_edge("b", "a")  # cycle
        configs = {
            "a": {"type": "mock", "inputs": {}},
            "b": {"type": "mock", "inputs": {}},
        }
        result = exec.execute("wf3", dag, configs)
        assert result.status == ExecutionStatus.FAILED

    def test_unknown_node_type_fails(self):
        exec = WorkflowExecutor()
        dag = DAG.from_edges(["n1"], [])
        configs = {"n1": {"type": "unknown_type", "inputs": {}}}
        result = exec.execute("wf4", dag, configs)
        assert result.status == ExecutionStatus.FAILED

    def test_variable_passing_between_nodes(self):
        exec = WorkflowExecutor()

        def handler_a(inputs, ctx, config):
            return {"value": 42}

        def handler_b(inputs, ctx, config):
            # This handler receives already-resolved inputs from context
            return {"received": inputs.get("upstream", "none")}

        exec.register_handler("type_a", handler_a)
        exec.register_handler("type_b", handler_b)

        dag = DAG.from_edges(["a", "b"], [("a", "b")])
        configs = {
            "a": {"type": "type_a", "inputs": {}},
            "b": {"type": "type_b", "inputs": {"upstream": "{{a.output.value}}"}},
        }
        result = exec.execute("wf5", dag, configs)
        assert result.status == ExecutionStatus.SUCCESS
        assert result.node_results["b"].output["received"] == 42

    def test_elapsed_ms_populated(self):
        exec = self._make_executor()
        dag = DAG.from_edges(["n1"], [])
        configs = {"n1": {"type": "mock", "inputs": {}}}
        result = exec.execute("wf6", dag, configs)
        assert result.elapsed_ms > 0
        assert result.node_results["n1"].elapsed_ms >= 0
