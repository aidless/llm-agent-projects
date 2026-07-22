"""Instrumentation 模块测试 - LLM/Agent/RAG 埋点。"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from storage.memory_store import get_store
from tracing.tracer import reset_tracer
from instrumentation.llm import trace_llm, trace_llm_call, calculate_cost
from instrumentation.agent import trace_agent, trace_tool_call
from instrumentation.rag import trace_rag, trace_retrieval
from metrics.registry import get_registry


class TestLLMInstrumentation:
    """LLM 埋点测试。"""

    def test_trace_llm_decorator(self):
        @trace_llm(model="gpt-4")
        def fake_llm(prompt: str):
            return {
                "llm_input": prompt,
                "llm_output": "Hello!",
                "prompt_tokens": 10,
                "completion_tokens": 5,
            }

        result = fake_llm("Hi")
        assert result["llm_output"] == "Hello!"
        assert result["prompt_tokens"] == 10

        # 检查 span 已存储
        store = get_store()
        traces = store.list_traces()
        assert len(traces) == 1
        spans = traces[0]["spans"]
        assert any(s["kind"] == "llm" for s in spans)

    def test_trace_llm_context_manager(self):
        with trace_llm_call(model="gpt-3.5-turbo") as span:
            span.set_llm(
                input_text="Hello",
                output_text="Hi!",
                model_name="gpt-3.5-turbo",
                prompt_tokens=15,
                completion_tokens=8,
            )
        assert span.is_finished
        assert span.total_tokens == 23

    def test_llm_metrics_recorded(self):
        @trace_llm(model="gpt-4")
        def fake_llm(prompt: str):
            return {
                "llm_input": prompt,
                "llm_output": "response",
                "prompt_tokens": 10,
                "completion_tokens": 5,
            }

        fake_llm("test")
        registry = get_registry()
        metrics = registry.list_metrics()
        names = [m["name"] for m in metrics]
        assert "llm.requests.total" in names
        assert "llm.requests.success" in names

    def test_calculate_cost(self):
        cost = calculate_cost("gpt-4", 1000, 500)
        assert cost > 0
        # gpt-4: prompt $0.03/1K, completion $0.06/1K
        expected = 1.0 * 0.03 + 0.5 * 0.06
        assert abs(cost - expected) < 1e-9

    def test_llm_error_metrics(self):
        @trace_llm(model="gpt-4")
        def failing_llm(prompt: str):
            raise RuntimeError("API error")

        with pytest.raises(RuntimeError):
            failing_llm("test")

        registry = get_registry()
        all_metrics = registry.get_all_metrics()
        error_counter = all_metrics.get("llm.requests.error")
        assert error_counter is not None
        assert error_counter.value >= 1


class TestAgentInstrumentation:
    """Agent 埋点测试。"""

    def test_trace_agent_context_manager(self):
        with trace_agent("TestAgent", agent_type="react", input_text="Find X") as span:
            span.set_attribute("agent.output", "Found X")
        assert span.is_finished
        assert span.attributes.get("agent.type") == "react"

    def test_trace_tool_call(self):
        with trace_agent("MyAgent") as agent_span:
            with trace_tool_call("calculator", {"expr": "2+2"}) as tool_span:
                tool_span.set_attribute("tool.output", "4")
        assert tool_span.is_finished
        assert tool_span.attributes["tool.name"] == "calculator"

    def test_agent_metrics(self):
        with trace_agent("TestAgent"):
            pass
        registry = get_registry()
        all_metrics = registry.get_all_metrics()
        assert "agent.runs" in all_metrics


class TestRAGInstrumentation:
    """RAG 埋点测试。"""

    def test_trace_rag(self):
        with trace_rag(query="What is X?", top_k=5) as span:
            span.set_attribute("rag.retrieved_docs", 3)
        assert span.is_finished
        assert span.attributes["rag.top_k"] == 5

    def test_trace_retrieval(self):
        with trace_retrieval(retriever_name="faiss", query="test") as span:
            span.set_attribute("retrieval.result_count", 10)
        assert span.is_finished
        assert span.kind.value == "retrieval"

    def test_rag_metrics(self):
        with trace_rag(query="test"):
            pass
        registry = get_registry()
        all_metrics = registry.get_all_metrics()
        assert "rag.queries" in all_metrics