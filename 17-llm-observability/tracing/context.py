"""上下文传播 - 使用 contextvars 在异步环境中传播 Trace 上下文。"""

from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional

_current_span: ContextVar[Optional[str]] = ContextVar("current_span", default=None)
_current_trace: ContextVar[Optional[str]] = ContextVar("current_trace", default=None)
_trace_flags: ContextVar[int] = ContextVar("trace_flags", default=1)


@dataclass
class TraceContext:
    """封装当前追踪上下文，方便批量操作。"""

    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    flags: int = 1
    # 附加 baggage (跨服务透传的键值对)
    baggage: dict = field(default_factory=dict)

    # ── contextvars 读写 ──────────────────────────────────

    def activate(self) -> "TraceContext":
        """将当前上下文写入 contextvars，返回 self 以便链式调用。"""
        _current_trace.set(self.trace_id)
        _current_span.set(self.span_id)
        _trace_flags.set(self.flags)
        return self

    @classmethod
    def current(cls) -> "TraceContext":
        """从 contextvars 读取当前上下文。"""
        return cls(
            trace_id=_current_trace.get(),
            span_id=_current_span.get(),
            flags=_trace_flags.get(),
        )


def get_current_trace_id() -> Optional[str]:
    return _current_trace.get()


def get_current_span_id() -> Optional[str]:
    return _current_span.get()


def set_context(trace_id: Optional[str], span_id: Optional[str]) -> None:
    _current_trace.set(trace_id)
    _current_span.set(span_id)


def clear_context() -> None:
    _current_trace.set(None)
    _current_span.set(None)
    _trace_flags.set(1)