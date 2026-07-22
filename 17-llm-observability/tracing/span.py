"""Span 实现 - 表示一次命名的、定时的操作。"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class SpanKind(str, Enum):
    """Span 类型枚举。"""

    LLM = "llm"
    TOOL = "tool"
    AGENT = "agent"
    RETRIEVAL = "retrieval"
    CHAIN = "chain"
    INTERNAL = "internal"
    CLIENT = "client"
    SERVER = "server"
    PRODUCER = "producer"
    CONSUMER = "consumer"


class SpanStatus(str, Enum):
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


@dataclass
class SpanEvent:
    """Span 生命周期内的事件。"""

    name: str
    timestamp: float = field(default_factory=time.time)
    attributes: Dict[str, Any] = field(default_factory=dict)


class Span:
    """表示追踪中的一次操作。

    通过 Tracer.create_span() 创建，不应直接实例化。

    用法::

        with tracer.start_span("my_operation") as span:
            span.set_attribute("key", "value")
            span.add_event("milestone")
    """

    def __init__(
        self,
        name: str,
        span_id: Optional[str] = None,
        trace_id: Optional[str] = None,
        parent_span_id: Optional[str] = None,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Optional[Dict[str, Any]] = None,
        start_time: Optional[float] = None,
    ) -> None:
        self.name = name
        self.span_id: str = span_id or _gen_span_id()
        self.trace_id: str = trace_id or _gen_trace_id()
        self.parent_span_id = parent_span_id
        self.kind = kind
        self.attributes: Dict[str, Any] = attributes or {}
        self.status: SpanStatus = SpanStatus.UNSET
        self.status_message: str = ""

        self.start_time: float = start_time or time.time()
        self.end_time: Optional[float] = None
        self.events: List[SpanEvent] = []

        # LLM 特化字段
        self.llm_input: Optional[Any] = None
        self.llm_output: Optional[Any] = None
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.total_tokens: int = 0
        self.model_name: Optional[str] = None
        self.prompt_template: Optional[str] = None
        self.prompt_template_version: Optional[str] = None
        self.stream_chunks: List[Dict[str, Any]] = []

    # ── 属性操作 ───────────────────────────────────────────

    def set_attribute(self, key: str, value: Any) -> "Span":
        """设置 Span 属性。"""
        self.attributes[key] = value
        return self

    def set_attributes(self, attrs: Dict[str, Any]) -> "Span":
        """批量设置 Span 属性。"""
        self.attributes.update(attrs)
        return self

    # ── 状态操作 ───────────────────────────────────────────

    def set_status(self, status: SpanStatus, message: str = "") -> "Span":
        """设置 Span 状态。"""
        self.status = status
        self.status_message = message
        return self

    def set_ok(self) -> "Span":
        return self.set_status(SpanStatus.OK)

    def set_error(self, message: str = "") -> "Span":
        return self.set_status(SpanStatus.ERROR, message)

    # ── 事件操作 ───────────────────────────────────────────

    def add_event(self, name: str, attributes: Optional[Dict[str, Any]] = None) -> "Span":
        """添加事件到 Span。"""
        self.events.append(SpanEvent(name=name, attributes=attributes or {}))
        return self

    # ── LLM 特化 ───────────────────────────────────────────

    def set_llm(
        self,
        input_text: Any,
        output_text: Any,
        model_name: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        prompt_template: Optional[str] = None,
        prompt_template_version: Optional[str] = None,
    ) -> "Span":
        """设置 LLM 相关属性。"""
        self.llm_input = input_text
        self.llm_output = output_text
        self.model_name = model_name
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens
        self.prompt_template = prompt_template
        self.prompt_template_version = prompt_template_version
        self.set_attribute("llm.model", model_name)
        self.set_attribute("llm.prompt_tokens", prompt_tokens)
        self.set_attribute("llm.completion_tokens", completion_tokens)
        self.set_attribute("llm.total_tokens", self.total_tokens)
        return self

    def add_stream_chunk(self, chunk_index: int, content: str, timestamp: Optional[float] = None) -> None:
        """记录流式响应的一个 chunk。"""
        self.stream_chunks.append({
            "chunk_index": chunk_index,
            "content": content,
            "timestamp": timestamp or time.time(),
        })

    # ── 生命周期 ───────────────────────────────────────────

    def end(self, end_time: Optional[float] = None) -> None:
        """结束 Span 并记录结束时间。"""
        self.end_time = end_time or time.time()

    @property
    def duration_ms(self) -> float:
        """Span 持续时间（毫秒）。"""
        if self.end_time is None:
            return (time.time() - self.start_time) * 1000
        return (self.end_time - self.start_time) * 1000

    @property
    def is_finished(self) -> bool:
        return self.end_time is not None

    # ── 序列化 ─────────────────────────────────────────────

    def to_dict(self) -> Dict[str, Any]:
        """将 Span 序列化为字典。"""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "kind": self.kind.value,
            "status": self.status.value,
            "status_message": self.status_message,
            "attributes": self.attributes,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "duration_ms": self.duration_ms,
            "events": [
                {"name": e.name, "timestamp": e.timestamp, "attributes": e.attributes}
                for e in self.events
            ],
            "llm_input": self.llm_input,
            "llm_output": self.llm_output,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "model_name": self.model_name,
            "prompt_template": self.prompt_template,
            "prompt_template_version": self.prompt_template_version,
            "stream_chunk_count": len(self.stream_chunks),
        }


def _gen_trace_id() -> str:
    return uuid.uuid4().hex


def _gen_span_id() -> str:
    return uuid.uuid4().hex[:16]