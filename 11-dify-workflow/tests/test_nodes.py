"""节点测试"""

import pytest
from engine.context import ExecutionContext
from engine.errors import NodeExecutionError
from nodes.code_node import CodeNode
from nodes.condition_node import ConditionNode
from nodes.loop_node import LoopNode
from nodes.aggregate_node import AggregateNode
from nodes.timer_node import TimerNode
from nodes.llm_node import LLMNode
from nodes.sub_workflow_node import SubWorkflowNode


class TestLLMNode:
    def test_mock_llm_returns_text(self):
        node = LLMNode()
        ctx = ExecutionContext()
        result = node.execute(
            {"prompt": "Hello world", "model": "test"},
            ctx,
            {},
        )
        assert "text" in result
        assert "[LLM Mock]" in result["text"]
        assert result["model"] == "test"

    def test_custom_llm_client(self):
        def mock_client(prompt, **kwargs):
            return f"Custom: {prompt}"

        node = LLMNode(llm_client=mock_client)
        ctx = ExecutionContext()
        result = node.execute({"prompt": "hi"}, ctx, {})
        assert result["text"] == "Custom: hi"

    def test_usage_info(self):
        node = LLMNode()
        ctx = ExecutionContext()
        result = node.execute({"prompt": "abc"}, ctx, {})
        assert "usage" in result
        assert result["usage"]["prompt_tokens"] == 3


class TestCodeNode:
    def test_simple_calculation(self):
        node = CodeNode()
        ctx = ExecutionContext()
        result = node.execute(
            {"code": "result = 2 + 3", "x": 5},
            ctx,
            {},
        )
        assert result["result"] == 5

    def test_stdout_captured(self):
        node = CodeNode()
        ctx = ExecutionContext()
        result = node.execute(
            {"code": "print('hello from code')\nresult = 'done'"},
            ctx,
            {},
        )
        assert "hello from code" in result["stdout"]
        assert result["result"] == "done"

    def test_input_access(self):
        node = CodeNode()
        ctx = ExecutionContext()
        result = node.execute(
            {"code": "result = input.get('value', 0) * 2", "value": 21},
            ctx,
            {},
        )
        assert result["result"] == 42

    def test_blocked_exec_raises(self):
        node = CodeNode()
        ctx = ExecutionContext()
        with pytest.raises(NodeExecutionError, match="exec"):
            node.execute({"code": "exec('print(1)')"}, ctx, {})

    def test_empty_code_raises(self):
        node = CodeNode()
        ctx = ExecutionContext()
        with pytest.raises(NodeExecutionError, match="code"):
            node.execute({"code": ""}, ctx, {})


class TestConditionNode:
    def test_eq_match(self):
        node = ConditionNode()
        ctx = ExecutionContext()
        result = node.execute(
            {
                "conditions": [
                    {"left": "hello", "op": "eq", "right": "hello", "branch": "match"},
                ],
            },
            ctx,
            {},
        )
        assert result["branch"] == "match"

    def test_no_match_returns_default(self):
        node = ConditionNode()
        ctx = ExecutionContext()
        result = node.execute(
            {
                "conditions": [
                    {"left": "a", "op": "eq", "right": "b", "branch": "nope"},
                ],
                "default_branch": "fallback",
            },
            ctx,
            {},
        )
        assert result["branch"] == "fallback"

    def test_gt_operator(self):
        node = ConditionNode()
        ctx = ExecutionContext()
        result = node.execute(
            {"conditions": [{"left": 10, "op": "gt", "right": 5, "branch": "yes"}]},
            ctx,
            {},
        )
        assert result["branch"] == "yes"

    def test_contains_operator(self):
        node = ConditionNode()
        ctx = ExecutionContext()
        result = node.execute(
            {"conditions": [{"left": "hello world", "op": "contains", "right": "world", "branch": "found"}]},
            ctx,
            {},
        )
        assert result["branch"] == "found"

    def test_regex_match(self):
        node = ConditionNode()
        ctx = ExecutionContext()
        result = node.execute(
            {"conditions": [{"left": "test123", "op": "regex_match", "right": r"test\d+", "branch": "matched"}]},
            ctx,
            {},
        )
        assert result["branch"] == "matched"

    def test_empty_conditions_raises(self):
        node = ConditionNode()
        ctx = ExecutionContext()
        with pytest.raises(NodeExecutionError):
            node.execute({"conditions": []}, ctx, {})


class TestLoopNode:
    def test_for_loop(self):
        node = LoopNode()
        ctx = ExecutionContext()
        result = node.execute(
            {"loop_type": "for", "items": [1, 2, 3]},
            ctx,
            {},
        )
        assert result["iterations"] == 3
        assert result["loop_type"] == "for"
        assert len(result["results"]) == 3

    def test_while_loop(self):
        node = LoopNode()
        ctx = ExecutionContext()
        result = node.execute(
            {"loop_type": "while", "condition": "iteration < 3"},
            ctx,
            {},
        )
        assert result["iterations"] == 3

    def test_empty_list(self):
        node = LoopNode()
        ctx = ExecutionContext()
        result = node.execute(
            {"loop_type": "for", "items": []},
            ctx,
            {},
        )
        assert result["iterations"] == 0


class TestAggregateNode:
    def test_merge_strategy(self):
        node = AggregateNode()
        ctx = ExecutionContext()
        result = node.execute(
            {"strategy": "merge", "values": [{"a": 1}, {"b": 2}]},
            ctx,
            {},
        )
        assert result["aggregated"] == {"a": 1, "b": 2}

    def test_list_strategy(self):
        node = AggregateNode()
        ctx = ExecutionContext()
        result = node.execute(
            {"strategy": "list", "values": [1, 2, 3]},
            ctx,
            {},
        )
        assert result["aggregated"] == [1, 2, 3]

    def test_sum_strategy(self):
        node = AggregateNode()
        ctx = ExecutionContext()
        result = node.execute(
            {"strategy": "sum", "values": [1, 2, 3, 4, 5]},
            ctx,
            {},
        )
        assert result["aggregated"] == 15

    def test_concat_strategy(self):
        node = AggregateNode()
        ctx = ExecutionContext()
        result = node.execute(
            {"strategy": "concat", "values": ["a", "b", "c"]},
            ctx,
            {},
        )
        assert result["aggregated"] == "abc"


class TestTimerNode:
    def test_zero_delay(self):
        node = TimerNode()
        ctx = ExecutionContext()
        result = node.execute(
            {"delay_seconds": 0},
            ctx,
            {"skip_wait": True},
        )
        assert result["delayed"] == 0

    def test_skip_wait(self):
        node = TimerNode()
        ctx = ExecutionContext()
        result = node.execute(
            {"delay_seconds": 9999, "skip_wait": True},
            ctx,
            {},
        )
        assert result["skipped"] is True


class TestSubWorkflowNode:
    def test_mock_mode(self):
        node = SubWorkflowNode()
        ctx = ExecutionContext()
        result = node.execute(
            {"workflow_id": "test-wf"},
            ctx,
            {},
        )
        assert result["status"] == "mock_success"
        assert "test-wf" in result["output"]["result"]

    def test_missing_workflow_id_raises(self):
        node = SubWorkflowNode()
        ctx = ExecutionContext()
        with pytest.raises(NodeExecutionError):
            node.execute({}, ctx, {})
