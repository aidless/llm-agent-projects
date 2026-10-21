"""Integration layer: Use 17-llm-observability tracer in demo_pipeline.py.

Provides a real tracer wrapper that uses 17-llm-observability's Tracer/Span
classes to instrument the demo pipeline, replacing the Mock PipelineContext.

Falls back to Mock implementation if 17-llm-observability is not available
or fails to import (e.g., during quick tests).
"""

import os
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, List, Optional

# Add 17-llm-observability to path
OBSERVABILITY_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "tracing",
)
sys.path.insert(0, os.path.dirname(OBSERVABILITY_PATH))

try:
    from tracing.tracer import Tracer
    from tracing.span import SpanKind

    OBSERVABILITY_AVAILABLE = True
except ImportError as e:
    OBSERVABILITY_AVAILABLE = False
    IMPORT_ERROR = str(e)


class DemoTracer:
    """Real tracer for demo pipeline using 17-llm-observability.

    Provides a compatible interface with the Mock PipelineContext used in
    demo_pipeline.py, so the same code can use either implementation.
    """

    def __init__(self, service_name: str = "demo-pipeline"):
        if not OBSERVABILITY_AVAILABLE:
            raise RuntimeError(
                f"17-llm-observability not available: {IMPORT_ERROR}\n"
                "Falling back to Mock implementation."
            )
        self.service_name = service_name
        self.tracer = Tracer(service_name=service_name)
        self.trace_id: Optional[str] = None
        self.spans: List[Dict[str, Any]] = []
        self.tokens_input: int = 0
        self.tokens_output: int = 0
        self.cost_estimate: float = 0.0
        self.safety_checks: List[Dict[str, Any]] = []
        self.retrieved_docs: List[Dict[str, Any]] = []
        self.errors: List[str] = []
        self._span_stack: List[Any] = []

    def start_root_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        """Start the root trace span."""
        self.tracer.start_as_current_span(name, attributes=attributes or {})
        # Use the tracer's API to get trace_id
        from tracing.context import TraceContext, get_current_trace_id

        self.trace_id = get_current_trace_id() or f"trace-{int(time.time() * 1000)}"
        return self

    def record_step(self, name: str, duration_ms: float, attrs: Dict[str, Any] = None):
        """Record a step span (already in trace context)."""
        self.spans.append(
            {"name": name, "duration_ms": round(duration_ms, 2), "attrs": attrs or {}}
        )

    def estimate_cost(self, model: str = "gpt-4o-mini"):
        """Estimate cost from token usage (gpt-4o-mini: $0.15/M input, $0.6/M output)."""
        cost = (self.tokens_input / 1_000_000) * 0.15 + (self.tokens_output / 1_000_000) * 0.6
        self.cost_estimate = round(cost, 6)

    def report(self) -> Dict[str, Any]:
        """Generate observability report."""
        stage_durations = {s["name"]: s["duration_ms"] for s in self.spans}
        total_duration = sum(stage_durations.values())
        return {
            "trace_id": self.trace_id,
            "service": self.service_name,
            "total_duration_ms": round(total_duration, 2),
            "stage_durations": stage_durations,
            "tokens": {
                "input": self.tokens_input,
                "output": self.tokens_output,
                "total": self.tokens_input + self.tokens_output,
            },
            "cost_estimate_usd": self.cost_estimate,
            "errors": self.errors,
            "safety_checks": self.safety_checks,
            "num_retrieved_docs": len(self.retrieved_docs),
        }


# Fallback to Mock if 17-llm-observability is not available
class MockTracer:
    """Mock tracer used when 17-llm-observability cannot be imported.

    Provides a compatible interface so demo_pipeline.py works either way.
    """

    def __init__(self, service_name: str = "demo-pipeline"):
        self.service_name = service_name
        self.trace_id: Optional[str] = f"trace-{int(time.time() * 1000)}"
        self.spans: List[Dict[str, Any]] = []
        self.tokens_input: int = 0
        self.tokens_output: int = 0
        self.cost_estimate: float = 0.0
        self.safety_checks: List[Dict[str, Any]] = []
        self.retrieved_docs: List[Dict[str, Any]] = []
        self.errors: List[str] = []
        self._start_time: float = time.time()

    def start_root_span(self, name: str, attributes: Optional[Dict[str, Any]] = None):
        return self

    def record_step(self, name: str, duration_ms: float, attrs: Dict[str, Any] = None):
        self.spans.append(
            {"name": name, "duration_ms": round(duration_ms, 2), "attrs": attrs or {}}
        )

    def estimate_cost(self, model: str = "gpt-4o-mini"):
        cost = (self.tokens_input / 1_000_000) * 0.15 + (self.tokens_output / 1_000_000) * 0.6
        self.cost_estimate = round(cost, 6)

    def report(self) -> Dict[str, Any]:
        stage_durations = {s["name"]: s["duration_ms"] for s in self.spans}
        total_duration = sum(stage_durations.values()) or (time.time() - self._start_time) * 1000
        return {
            "trace_id": self.trace_id,
            "service": self.service_name,
            "total_duration_ms": round(total_duration, 2),
            "stage_durations": stage_durations,
            "tokens": {
                "input": self.tokens_input,
                "output": self.tokens_output,
                "total": self.tokens_input + self.tokens_output,
            },
            "cost_estimate_usd": self.cost_estimate,
            "errors": self.errors,
            "safety_checks": self.safety_checks,
            "num_retrieved_docs": len(self.retrieved_docs),
        }


def get_tracer(service_name: str = "demo-pipeline"):
    """Factory: returns DemoTracer if available, else MockTracer."""
    if OBSERVABILITY_AVAILABLE:
        try:
            return DemoTracer(service_name)
        except Exception:
            return MockTracer(service_name)
    return MockTracer(service_name)


__all__ = [
    "DemoTracer",
    "MockTracer",
    "get_tracer",
    "OBSERVABILITY_AVAILABLE",
]


if __name__ == "__main__":
    # Self-test
    print("=== Tracer Integration Self-Test ===")
    print(f"17-llm-observability available: {OBSERVABILITY_AVAILABLE}")
    if not OBSERVABILITY_AVAILABLE:
        print(f"  Import error: {IMPORT_ERROR}")

    tracer = get_tracer("self-test")
    print(f"Using: {type(tracer).__name__}")

    tracer.start_root_span("self-test")
    t0 = time.time()
    time.sleep(0.05)
    tracer.record_step("step-1", (time.time() - t0) * 1000, {"key": "value"})
    tracer.tokens_input = 100
    tracer.tokens_output = 50
    tracer.estimate_cost()
    report = tracer.report()
    print(f"Report: trace_id={report['trace_id']}, total={report['total_duration_ms']}ms, cost=${report['cost_estimate_usd']}")