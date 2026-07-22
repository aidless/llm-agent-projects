"""跨服务传播 - 模拟 HTTP header 中的 Trace 上下文注入和提取。"""

from dataclasses import dataclass, field
from typing import Dict, Optional

from tracing.context import TraceContext

# 模拟 W3C Trace Context 规范的 header 名
TRACE_PARENT_HEADER = "traceparent"
TRACE_STATE_HEADER = "tracestate"
BAGGAGE_HEADER = "baggage"


@dataclass
class Propagator:
    """基于 W3C Trace Context 规范的传播器（模拟实现）。

    traceparent 格式: {version}-{trace_id}-{span_id}-{flags}
    示例: 00-0af7651916cd43dd8448eb211c80319c-b7ad6b7169203331-01
    """

    version: str = "00"

    def inject(self, context: TraceContext, carrier: Dict[str, str]) -> Dict[str, str]:
        """将 TraceContext 注入到 carrier (如 HTTP headers) 中。"""
        if context.trace_id and context.span_id:
            carrier[TRACE_PARENT_HEADER] = (
                f"{self.version}-{context.trace_id}-{context.span_id}"
                f"-{context.flags:02x}"
            )
        if context.baggage:
            baggage_str = ",".join(
                f"{k}={v}" for k, v in context.baggage.items()
            )
            carrier[BAGGAGE_HEADER] = baggage_str
        return carrier

    def extract(self, carrier: Dict[str, str]) -> TraceContext:
        """从 carrier 中提取 TraceContext。"""
        tp = carrier.get(TRACE_PARENT_HEADER)
        if not tp:
            return TraceContext()

        parts = tp.split("-")
        if len(parts) < 4:
            return TraceContext()

        try:
            flags = int(parts[3], 16)
        except ValueError:
            flags = 1

        baggage = {}
        raw_baggage = carrier.get(BAGGAGE_HEADER, "")
        if raw_baggage:
            for item in raw_baggage.split(","):
                if "=" in item:
                    k, v = item.split("=", 1)
                    baggage[k.strip()] = v.strip()

        return TraceContext(
            trace_id=parts[1],
            span_id=parts[2],
            flags=flags,
            baggage=baggage,
        )