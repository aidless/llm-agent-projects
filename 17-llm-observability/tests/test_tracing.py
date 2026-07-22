"""Tracing 模块测试 - Tracer, Span, Context, Sampler, Propagator。"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from tracing.tracer import Tracer, get_tracer, reset_tracer
from tracing.span import Span, SpanKind, SpanStatus
from tracing.context import (
    TraceContext,
    get_current_trace_id,
    get_current_span_id,
    set_context,
    clear_context,
)
from tracing.sampler import (
    AlwaysOnSampler,
    AlwaysOffSampler,
    RatioSampler,
    ParentBasedSampler,
)
from tracing.propagator import Propagator
from storage.memory_store import get_store


class TestSpan:
    """Span 基本功能测试。"""

    def test_span_creation(self):
        span = Span(name="test-operation")
        assert span.name == "test-operation"
        assert span.span_id is not None
        assert len(span.span_id) == 16
        assert span.trace_id is not None
        assert len(span.trace_id) == 32
        assert span.status == SpanStatus.UNSET
        assert span.end_time is None

    def test_span_set_attribute(self):
        span = Span(name="test")
        span.set_attribute("key1", "value1")
        span.set_attributes({"key2": 42, "key3": True})
        assert span.attributes["key1"] == "value1"
        assert span.attributes["key2"] == 42
        assert span.attributes["key3"] is True

    def test_span_set_status(self):
        span = Span(name="test")
        span.set_ok()
        assert span.status == SpanStatus.OK
        span.set_error("something failed")
        assert span.status == SpanStatus.ERROR
        assert span.status_message == "something failed"

    def test_span_duration(self):
        span = Span(name="test")
        time.sleep(0.01)
        span.end()
        assert span.duration_ms > 0
        assert span.is_finished

    def test_span_events(self):
        span = Span(name="test")
        span.add_event("milestone", {"step": 1})
        assert len(span.events) == 1
        assert span.events[0].name == "milestone"
        assert span.events[0].attributes["step"] == 1

    def test_span_llm(self):
        span = Span(name="test")
        span.set_llm(
            input_text="Hello",
            output_text="Hi there",
            model_name="gpt-4",
            prompt_tokens=10,
            completion_tokens=20,
            prompt_template="Say {input}",
            prompt_template_version="v2",
        )
        assert span.prompt_tokens == 10
        assert span.completion_tokens == 20
        assert span.total_tokens == 30
        assert span.model_name == "gpt-4"
        assert span.prompt_template == "Say {input}"

    def test_span_stream_chunks(self):
        span = Span(name="test")
        span.add_stream_chunk(0, "Hello")
        span.add_stream_chunk(1, " World")
        assert len(span.stream_chunks) == 2
        assert span.stream_chunks[0]["chunk_index"] == 0

    def test_span_to_dict(self):
        span = Span(name="test", kind=SpanKind.LLM)
        d = span.to_dict()
        assert d["name"] == "test"
        assert d["kind"] == "llm"
        assert "span_id" in d
        assert "trace_id" in d


class TestContext:
    """上下文传播测试。"""

    def test_set_and_get_context(self):
        set_context("trace-123", "span-456")
        assert get_current_trace_id() == "trace-123"
        assert get_current_span_id() == "span-456"
        clear_context()
        assert get_current_trace_id() is None
        assert get_current_span_id() is None

    def test_trace_context_dataclass(self):
        ctx = TraceContext(trace_id="t1", span_id="s1", flags=1)
        assert ctx.trace_id == "t1"
        assert ctx.span_id == "s1"

    def test_trace_context_activate(self):
        ctx = TraceContext(trace_id="t2", span_id="s2").activate()
        assert get_current_trace_id() == "t2"
        assert get_current_span_id() == "s2"
        clear_context()

    def test_trace_context_current(self):
        set_context("t3", "s3")
        ctx = TraceContext.current()
        assert ctx.trace_id == "t3"
        assert ctx.span_id == "s3"
        clear_context()


class TestSampler:
    """采样策略测试。"""

    def test_always_on(self):
        sampler = AlwaysOnSampler()
        assert sampler.should_sample("t1", "op") is True

    def test_always_off(self):
        sampler = AlwaysOffSampler()
        assert sampler.should_sample("t1", "op") is False

    def test_ratio_sampler_valid_ratio(self):
        sampler = RatioSampler(0.5)
        # 概率测试：采样多次，不应该全部为 True 或 False
        results = [sampler.should_sample(f"t{i}", "op") for i in range(100)]
        assert any(results)  # 至少有一个 True
        assert not all(results)  # 至少有一个 False

    def test_ratio_sampler_invalid_ratio(self):
        with pytest.raises(ValueError):
            RatioSampler(1.5)

    def test_parent_based_sampler(self):
        delegate = AlwaysOffSampler()
        sampler = ParentBasedSampler(delegate)
        # 父已采样 -> 子也采样
        assert sampler.should_sample("t1", "op", parent_sampled=True) is True
        # 父未采样 -> 使用 delegate (AlwaysOff -> False)
        assert sampler.should_sample("t1", "op", parent_sampled=False) is False


class TestPropagator:
    """跨服务传播测试。"""

    def test_inject_and_extract(self):
        propagator = Propagator()
        ctx = TraceContext(trace_id="abc123", span_id="def456", flags=1, baggage={"env": "prod"})
        carrier = {}
        propagator.inject(ctx, carrier)
        assert "traceparent" in carrier
        assert carrier["baggage"] == "env=prod"

        extracted = propagator.extract(carrier)
        assert extracted.trace_id == "abc123"
        assert extracted.span_id == "def456"
        assert extracted.baggage == {"env": "prod"}

    def test_extract_empty_carrier(self):
        propagator = Propagator()
        ctx = propagator.extract({})
        assert ctx.trace_id is None
        assert ctx.span_id is None


class TestTracer:
    """Tracer 核心功能测试。"""

    def test_create_tracer(self):
        tracer = Tracer("my-service")
        assert tracer.service_name == "my-service"

    def test_create_span(self):
        tracer = Tracer("my-service")
        span = tracer.create_span("operation")
        assert span.name == "operation"
        assert span.trace_id is not None
        assert span.span_id is not None

    def test_start_span_context_manager(self):
        tracer = Tracer("my-service")
        with tracer.start_span("op1") as span:
            span.set_attribute("key", "value")
            span.set_ok()
        assert span.is_finished
        assert span.status == SpanStatus.OK

    def test_span_parent_child(self):
        tracer = Tracer("my-service")
        with tracer.start_span("parent") as parent:
            with tracer.start_span("child") as child:
                assert child.parent_span_id == parent.span_id
                assert child.trace_id == parent.trace_id

    def test_span_error_in_context_manager(self):
        tracer = Tracer("my-service")
        with pytest.raises(ValueError):
            with tracer.start_span("failing") as span:
                raise ValueError("test error")
        assert span.status == SpanStatus.ERROR
        assert "test error" in span.status_message

    def test_span_stored(self):
        tracer = Tracer("my-service")
        with tracer.start_span("stored") as span:
            pass
        store = get_store()
        stored = store.get_span(span.span_id)
        assert stored is not None
        assert stored["name"] == "stored"

    def test_trace_created(self):
        tracer = Tracer("my-service")
        with tracer.start_span("root") as span:
            pass
        store = get_store()
        trace = store.get_trace(span.trace_id)
        assert trace is not None
        assert trace["span_count"] >= 1
