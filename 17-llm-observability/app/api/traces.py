"""Trace 查询 API - 提供检索和过滤 Trace 的接口。"""

import math
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query

from app.models import (
    LatencyAnalysis,
    TraceDetailResponse,
    TraceListResponse,
    TraceModel,
    SpanModel,
    SpanKindEnum,
)
from storage.memory_store import get_store

router = APIRouter(prefix="/api/v1/traces", tags=["traces"])


@router.get("", response_model=TraceListResponse)
def list_traces(
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    span_type: Optional[str] = Query(None),
    min_duration: Optional[float] = Query(None),
    max_duration: Optional[float] = Query(None),
    has_error: Optional[bool] = Query(None),
):
    """查询 Trace 列表，支持多种过滤条件。"""
    store = get_store()
    traces = store.list_traces(
        limit=limit,
        offset=offset,
        span_type=span_type,
        min_duration=min_duration,
        max_duration=max_duration,
        has_error=has_error,
    )
    total = store.trace_count()
    return TraceListResponse(
        total=total,
        traces=[_to_trace_model(t) for t in traces],
        limit=limit,
        offset=offset,
    )


@router.get("/{trace_id}", response_model=TraceDetailResponse)
def get_trace(trace_id: str):
    """获取单个 Trace 详情。"""
    store = get_store()
    trace = store.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")
    return TraceDetailResponse(trace=_to_trace_model(trace))


@router.get("/{trace_id}/spans", response_model=List[SpanModel])
def get_trace_spans(trace_id: str):
    """获取 Trace 下的所有 Span。"""
    store = get_store()
    trace = store.get_trace(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")
    spans = store.get_trace_spans(trace_id)
    return [_to_span_model(s) for s in spans]


@router.get("/analysis/latency", response_model=LatencyAnalysis)
def analyze_latency(
    span_type: Optional[str] = Query("llm"),
):
    """延迟分析 - 计算 P50/P95/P99 等分位值。"""
    store = get_store()
    traces = store.list_traces(limit=10000, span_type=span_type)

    durations = []
    for t in traces:
        for s in t.get("spans", []):
            if span_type and s.get("kind") != span_type:
                continue
            d = s.get("duration_ms", 0)
            if d > 0:
                durations.append(d)

    if not durations:
        return LatencyAnalysis(
            avg_ms=0, p50_ms=0, p95_ms=0, p99_ms=0,
            min_ms=0, max_ms=0, count=0,
        )

    sorted_d = sorted(durations)
    n = len(sorted_d)

    def percentile(p: float) -> float:
        idx = (p / 100.0) * (n - 1)
        lower = int(math.floor(idx))
        upper = min(lower + 1, n - 1)
        frac = idx - lower
        return sorted_d[lower] + frac * (sorted_d[upper] - sorted_d[lower])

    return LatencyAnalysis(
        avg_ms=sum(durations) / n,
        p50_ms=percentile(50),
        p95_ms=percentile(95),
        p99_ms=percentile(99),
        min_ms=sorted_d[0],
        max_ms=sorted_d[-1],
        count=n,
    )


@router.get("/analysis/errors")
def analyze_errors():
    """错误率分析。"""
    store = get_store()
    traces = store.list_traces(limit=10000)

    total_spans = 0
    error_count = 0
    error_breakdown: dict = {}

    for t in traces:
        for s in t.get("spans", []):
            total_spans += 1
            if s.get("status") == "error":
                error_count += 1
                kind = s.get("kind", "unknown")
                error_breakdown[kind] = error_breakdown.get(kind, 0) + 1

    return {
        "total_requests": total_spans,
        "error_count": error_count,
        "error_rate": error_count / total_spans if total_spans > 0 else 0,
        "error_breakdown": error_breakdown,
    }


@router.get("/analysis/cost")
def analyze_cost():
    """成本分析。"""
    store = get_store()
    traces = store.list_traces(limit=10000)

    total_cost = 0.0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    request_count = 0
    by_model: dict = {}

    for t in traces:
        for s in t.get("spans", []):
            if s.get("kind") != "llm":
                continue
            request_count += 1
            cost = s.get("attributes", {}).get("llm.cost_usd", 0)
            total_cost += cost
            pt = s.get("prompt_tokens", 0)
            ct = s.get("completion_tokens", 0)
            total_prompt_tokens += pt
            total_completion_tokens += ct
            model = s.get("model_name") or s.get("attributes", {}).get("llm.model", "unknown")
            if model not in by_model:
                by_model[model] = {"cost_usd": 0, "request_count": 0, "total_tokens": 0}
            by_model[model]["cost_usd"] += cost
            by_model[model]["request_count"] += 1
            by_model[model]["total_tokens"] += pt + ct

    return {
        "total_cost_usd": round(total_cost, 6),
        "avg_cost_per_request": round(total_cost / request_count, 6) if request_count > 0 else 0,
        "total_tokens": total_prompt_tokens + total_completion_tokens,
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "by_model": by_model,
    }


@router.get("/analysis/trends")
def analyze_trends(
    window: str = Query("1h"),
):
    """使用趋势分析。"""
    store = get_store()
    import time
    window_seconds = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600}.get(window, 3600)
    cutoff = time.time() - window_seconds

    traces = store.list_traces(limit=10000)
    traces = [t for t in traces if t.get("start_time", 0) >= cutoff]

    request_count = len(traces)
    token_count = 0
    durations = []

    for t in traces:
        for s in t.get("spans", []):
            if s.get("kind") == "llm":
                token_count += s.get("total_tokens", 0)
            durations.append(s.get("duration_ms", 0))

    return {
        "window": window,
        "window_seconds": window_seconds,
        "request_count": request_count,
        "token_count": token_count,
        "avg_latency_ms": sum(durations) / len(durations) if durations else 0,
    }


# ── 辅助函数 ───────────────────────────────────────────────


def _to_trace_model(t: dict) -> TraceModel:
    spans = []
    for s in t.get("spans", []):
        spans.append(_to_span_model(s))
    return TraceModel(
        trace_id=t["trace_id"],
        name=t.get("name", ""),
        service_name=t.get("service_name", ""),
        start_time=t.get("start_time", 0),
        end_time=t.get("end_time"),
        duration_ms=t.get("duration_ms", 0),
        span_count=t.get("span_count", len(spans)),
        spans=spans,
        status=t.get("status", "unset"),
    )


def _to_span_model(s: dict) -> SpanModel:
    events = []
    for e in s.get("events", []):
        events.append({
            "name": e.get("name", ""),
            "timestamp": e.get("timestamp", 0),
            "attributes": e.get("attributes", {}),
        })
    return SpanModel(
        span_id=s["span_id"],
        trace_id=s["trace_id"],
        parent_span_id=s.get("parent_span_id"),
        name=s.get("name", ""),
        kind=s.get("kind", "internal"),
        status=s.get("status", "unset"),
        status_message=s.get("status_message", ""),
        attributes=s.get("attributes", {}),
        start_time=s.get("start_time", 0),
        end_time=s.get("end_time"),
        duration_ms=s.get("duration_ms", 0),
        events=events,
        llm_input=s.get("llm_input"),
        llm_output=s.get("llm_output"),
        prompt_tokens=s.get("prompt_tokens", 0),
        completion_tokens=s.get("completion_tokens", 0),
        total_tokens=s.get("total_tokens", 0),
        model_name=s.get("model_name"),
        prompt_template=s.get("prompt_template"),
        prompt_template_version=s.get("prompt_template_version"),
        stream_chunk_count=s.get("stream_chunk_count", 0),
        service_name=s.get("service_name"),
    )