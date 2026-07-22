"""Tracer 核心 - 创建和管理 Span，实现上下文传播和采样。"""

import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

from storage.memory_store import get_store
from tracing.context import TraceContext, clear_context, set_context
from tracing.sampler import AlwaysOnSampler, ParentBasedSampler, Sampler
from tracing.span import Span, SpanKind

_DEFAULT_TRACER: Optional["Tracer"] = None


class Tracer:
    """追踪器 - 创建 Span 并管理其生命周期。

    用法::

        tracer = Tracer("my-service")

        # 上下文管理器方式
        with tracer.start_span("operation") as span:
            span.set_attribute("key", "value")

        # 手动方式
        span = tracer.start_span("operation")
        try:
            ...
        finally:
            span.end()
    """

    def __init__(
        self,
        service_name: str = "unknown",
        sampler: Optional[Sampler] = None,
    ) -> None:
        self.service_name = service_name
        self.sampler = sampler or ParentBasedSampler(AlwaysOnSampler())

    # ── 创建 Span ──────────────────────────────────────────

    @contextmanager
    def start_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
        parent_span_id: Optional[str] = None,
        trace_id: Optional[str] = None,
    ):
        """上下文管理器方式创建 Span。"""
        span = self.create_span(
            name=name,
            kind=kind,
            attributes=attributes,
            parent_span_id=parent_span_id,
            trace_id=trace_id,
        )
        try:
            yield span
        except Exception as exc:
            span.set_error(str(exc))
            raise
        finally:
            span.end()
            self._on_span_finished(span)

    def create_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
        parent_span_id: Optional[str] = None,
        trace_id: Optional[str] = None,
    ) -> Span:
        """创建一个新的 Span 并自动建立父子关系。

        如果未显式提供 trace_id / parent_span_id，则从 contextvars 中继承。
        """
        ctx = TraceContext.current()

        effective_trace_id = trace_id or ctx.trace_id
        if effective_trace_id is None:
            effective_trace_id = self._new_trace_id()

        effective_parent = parent_span_id or ctx.span_id

        span = Span(
            name=name,
            trace_id=effective_trace_id,
            parent_span_id=effective_parent,
            kind=kind,
            attributes=attributes or {},
        )

        # 设置上下文
        set_context(effective_trace_id, span.span_id)

        return span

    def start_as_current_span(
        self,
        name: str,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
    ):
        """start_span 的别名，兼容 OpenTelemetry 命名。"""
        return self.start_span(name=name, kind=kind, attributes=attributes)

    # ── 采样 ───────────────────────────────────────────────

    def should_sample(self, trace_id: str, name: str) -> bool:
        return self.sampler.should_sample(trace_id, name)

    # ── 内部方法 ───────────────────────────────────────────

    def _on_span_finished(self, span: Span) -> None:
        """Span 结束后的回调：存储数据并恢复上下文。"""
        # 采样检查 - root span 检查
        if span.parent_span_id is None:
            if not self.sampler.should_sample(span.trace_id, span.name):
                return

        store = get_store()

        # 存储 span
        span_dict = span.to_dict()
        span_dict["service_name"] = self.service_name
        store.store_span(span_dict)

        # 存储 span events
        for event in span.events:
            store.add_span_event(span.span_id, {
                "name": event.name,
                "timestamp": event.timestamp,
                "attributes": event.attributes,
            })

        # 如果是 root span 或其 trace 尚未存储，则创建/更新 trace 记录
        self._upsert_trace(span, store)

    def _upsert_trace(self, span: Span, store: Any) -> None:
        """创建或更新 Trace 汇总记录。"""
        existing = store.get_trace(span.trace_id)
        spans = store.get_trace_spans(span.trace_id)

        if existing:
            # 更新 trace 的结束时间（取最后一个 span 的结束时间）
            if span.end_time:
                existing["end_time"] = max(
                    existing.get("end_time") or 0, span.end_time
                )
                existing["duration_ms"] = (
                    (existing["end_time"] - existing["start_time"]) * 1000
                )
            existing["span_count"] = len(spans)
            existing["spans"] = [s for s in spans if s.get("span_id") != span.span_id] + [span.to_dict()]
            store.store_trace(existing)
        else:
            # 新建 trace
            trace = {
                "trace_id": span.trace_id,
                "name": span.name,
                "service_name": self.service_name,
                "start_time": span.start_time,
                "end_time": span.end_time or time.time(),
                "duration_ms": span.duration_ms,
                "span_count": 1,
                "spans": [span.to_dict()],
                "status": span.status.value,
            }
            store.store_trace(trace)

    @staticmethod
    def _new_trace_id() -> str:
        from tracing.span import _gen_trace_id
        return _gen_trace_id()


# ── 全局 Tracer ────────────────────────────────────────────

def get_tracer(service_name: str = "default") -> Tracer:
    """获取全局 Tracer 实例。"""
    global _DEFAULT_TRACER
    if _DEFAULT_TRACER is None:
        _DEFAULT_TRACER = Tracer(service_name=service_name)
    return _DEFAULT_TRACER


def reset_tracer() -> None:
    """重置全局 Tracer（仅用于测试）。"""
    global _DEFAULT_TRACER
    _DEFAULT_TRACER = None
    clear_context()