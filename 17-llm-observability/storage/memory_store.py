"""内存存储后端 - 存储所有 Trace、Span、Metrics、Log 和 Feedback 数据。"""

import time
import threading
from collections import defaultdict
from typing import Any, Dict, List, Optional


class MemoryStore:
    """线程安全的内存存储，提供 Trace/Span/Metrics/Log/Feedback 的 CRUD 操作。"""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # traces: trace_id -> trace dict
        self._traces: Dict[str, Dict[str, Any]] = {}
        # spans: span_id -> span dict
        self._spans: Dict[str, Dict[str, Any]] = {}
        # trace_id -> [span_id, ...]
        self._trace_spans: Dict[str, List[str]] = defaultdict(list)
        # metrics: metric_name -> list of data points
        self._metrics: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        # logs
        self._logs: List[Dict[str, Any]] = []
        # feedbacks
        self._feedbacks: Dict[str, Dict[str, Any]] = {}
        # span_id -> [event, ...]
        self._span_events: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    # ── Trace ──────────────────────────────────────────────

    def store_trace(self, trace: Dict[str, Any]) -> None:
        with self._lock:
            self._traces[trace["trace_id"]] = trace

    def get_trace(self, trace_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._traces.get(trace_id)

    def list_traces(
        self,
        limit: int = 20,
        offset: int = 0,
        *,
        span_type: Optional[str] = None,
        min_duration: Optional[float] = None,
        max_duration: Optional[float] = None,
        has_error: Optional[bool] = None,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            traces = list(self._traces.values())
        if span_type:
            traces = [
                t for t in traces
                if any(s.get("kind") == span_type for s in t.get("spans", []))
            ]
        if min_duration is not None:
            traces = [t for t in traces if (t.get("duration_ms") or 0) >= min_duration]
        if max_duration is not None:
            traces = [t for t in traces if (t.get("duration_ms") or 0) <= max_duration]
        if has_error is not None:
            traces = [
                t for t in traces
                if has_error == any(s.get("status") == "error" for s in t.get("spans", []))
            ]
        traces.sort(key=lambda t: t.get("start_time", 0), reverse=True)
        return traces[offset: offset + limit]

    def trace_count(self) -> int:
        with self._lock:
            return len(self._traces)

    # ── Span ───────────────────────────────────────────────

    def store_span(self, span: Dict[str, Any]) -> None:
        with self._lock:
            self._spans[span["span_id"]] = span
            self._trace_spans[span["trace_id"]].append(span["span_id"])

    def get_span(self, span_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._spans.get(span_id)

    def get_trace_spans(self, trace_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            span_ids = self._trace_spans.get(trace_id, [])
            return [self._spans[sid] for sid in span_ids if sid in self._spans]

    # ── Span Events ────────────────────────────────────────

    def add_span_event(self, span_id: str, event: Dict[str, Any]) -> None:
        with self._lock:
            self._span_events[span_id].append(event)

    def get_span_events(self, span_id: str) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._span_events.get(span_id, []))

    # ── Metrics ────────────────────────────────────────────

    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        point: Dict[str, Any] = {
            "name": name,
            "value": value,
            "timestamp": time.time(),
            "tags": tags or {},
        }
        with self._lock:
            self._metrics[name].append(point)

    def get_metrics(
        self, name: str, window_seconds: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        with self._lock:
            points = list(self._metrics.get(name, []))
        if window_seconds is not None:
            cutoff = time.time() - window_seconds
            points = [p for p in points if p["timestamp"] >= cutoff]
        return points

    def list_metric_names(self) -> List[str]:
        with self._lock:
            return list(self._metrics.keys())

    # ── Logs ───────────────────────────────────────────────

    def store_log(self, log_entry: Dict[str, Any]) -> None:
        with self._lock:
            self._logs.append(log_entry)

    def query_logs(
        self,
        limit: int = 50,
        offset: int = 0,
        *,
        trace_id: Optional[str] = None,
        level: Optional[str] = None,
        message_contains: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        with self._lock:
            logs = list(self._logs)
        if trace_id:
            logs = [l for l in logs if l.get("trace_id") == trace_id]
        if level:
            logs = [l for l in logs if l.get("level") == level]
        if message_contains:
            logs = [l for l in logs if message_contains in l.get("message", "")]
        logs.sort(key=lambda l: l.get("timestamp", 0), reverse=True)
        return logs[offset: offset + limit]

    def log_count(self) -> int:
        with self._lock:
            return len(self._logs)

    # ── Feedback ───────────────────────────────────────────

    def store_feedback(self, feedback: Dict[str, Any]) -> None:
        with self._lock:
            self._feedbacks[feedback["feedback_id"]] = feedback

    def get_feedback(self, feedback_id: str) -> Optional[Dict[str, Any]]:
        with self._lock:
            return self._feedbacks.get(feedback_id)

    def list_feedbacks(
        self, trace_id: Optional[str] = None, limit: int = 20, offset: int = 0
    ) -> List[Dict[str, Any]]:
        with self._lock:
            fbs = list(self._feedbacks.values())
        if trace_id:
            fbs = [f for f in fbs if f.get("trace_id") == trace_id]
        fbs.sort(key=lambda f: f.get("created_at", 0), reverse=True)
        return fbs[offset: offset + limit]

    def feedback_count(self, trace_id: Optional[str] = None) -> int:
        with self._lock:
            fbs = list(self._feedbacks.values())
        if trace_id:
            fbs = [f for f in fbs if f.get("trace_id") == trace_id]
        return len(fbs)

    # ── Utility ────────────────────────────────────────────

    def clear(self) -> None:
        with self._lock:
            self._traces.clear()
            self._spans.clear()
            self._trace_spans.clear()
            self._metrics.clear()
            self._logs.clear()
            self._feedbacks.clear()
            self._span_events.clear()


# 全局单例
_global_store: Optional[MemoryStore] = None


def get_store() -> MemoryStore:
    """获取全局 MemoryStore 单例。"""
    global _global_store
    if _global_store is None:
        _global_store = MemoryStore()
    return _global_store


def reset_store() -> None:
    """重置全局单例（仅用于测试）。"""
    global _global_store
    if _global_store is not None:
        _global_store.clear()