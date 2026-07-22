"""Pydantic 模型 - API 请求和响应的数据模型。"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Trace 模型 ─────────────────────────────────────────────


class SpanKindEnum(str, Enum):
    LLM = "llm"
    TOOL = "tool"
    AGENT = "agent"
    RETRIEVAL = "retrieval"
    CHAIN = "chain"
    INTERNAL = "internal"
    CLIENT = "client"
    SERVER = "server"


class SpanStatusEnum(str, Enum):
    UNSET = "unset"
    OK = "ok"
    ERROR = "error"


class SpanEventModel(BaseModel):
    name: str
    timestamp: float
    attributes: Dict[str, Any] = Field(default_factory=dict)


class SpanModel(BaseModel):
    span_id: str
    trace_id: str
    parent_span_id: Optional[str] = None
    name: str
    kind: SpanKindEnum = SpanKindEnum.INTERNAL
    status: SpanStatusEnum = SpanStatusEnum.UNSET
    status_message: str = ""
    attributes: Dict[str, Any] = Field(default_factory=dict)
    start_time: float
    end_time: Optional[float] = None
    duration_ms: float = 0.0
    events: List[SpanEventModel] = Field(default_factory=list)
    llm_input: Optional[Any] = None
    llm_output: Optional[Any] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    model_name: Optional[str] = None
    prompt_template: Optional[str] = None
    prompt_template_version: Optional[str] = None
    stream_chunk_count: int = 0
    service_name: Optional[str] = None


class TraceModel(BaseModel):
    trace_id: str
    name: str
    service_name: str = ""
    start_time: float
    end_time: Optional[float] = None
    duration_ms: float = 0.0
    span_count: int = 0
    spans: List[SpanModel] = Field(default_factory=list)
    status: str = "unset"


class TraceListResponse(BaseModel):
    total: int
    traces: List[TraceModel]
    limit: int
    offset: int


class TraceDetailResponse(BaseModel):
    trace: TraceModel


# ── Metrics 模型 ───────────────────────────────────────────


class MetricTypeEnum(str, Enum):
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"


class MetricInfo(BaseModel):
    name: str
    type: MetricTypeEnum
    description: str = ""


class MetricListResponse(BaseModel):
    metrics: List[MetricInfo]


class MetricDataPoint(BaseModel):
    value: float
    timestamp: float
    tags: Dict[str, str] = Field(default_factory=dict)


class MetricDetailResponse(BaseModel):
    name: str
    points: List[MetricDataPoint]
    summary: Optional[Dict[str, Any]] = None


class AggregationResponse(BaseModel):
    aggregations: List[Dict[str, Any]]
    window: str


# ── Log 模型 ───────────────────────────────────────────────


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogEntryModel(BaseModel):
    timestamp: float
    level: str
    logger: str = ""
    message: str
    module: str = ""
    function: str = ""
    line: int = 0
    trace_id: Optional[str] = None
    span_id: Optional[str] = None
    exception: Optional[Dict[str, Any]] = None


class LogListResponse(BaseModel):
    total: int
    logs: List[LogEntryModel]
    limit: int
    offset: int


# ── Feedback 模型 ──────────────────────────────────────────


class FeedbackTypeEnum(str, Enum):
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    RATING = "rating"
    CORRECTION = "correction"
    FLAG = "flag"


class FeedbackCreateRequest(BaseModel):
    trace_id: str
    feedback_type: FeedbackTypeEnum
    value: Optional[float] = None
    comment: str = ""
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class FeedbackModel(BaseModel):
    feedback_id: str
    trace_id: str
    feedback_type: FeedbackTypeEnum
    value: Optional[float] = None
    comment: str = ""
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: float


class FeedbackListResponse(BaseModel):
    total: int
    feedbacks: List[FeedbackModel]


class FeedbackStatsResponse(BaseModel):
    total: int
    thumbs_up: int
    thumbs_down: int
    thumbs_up_ratio: float
    avg_rating: Optional[float] = None
    rating_count: int


# ── Analysis 模型 ──────────────────────────────────────────


class LatencyAnalysis(BaseModel):
    avg_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    count: int


class ErrorAnalysis(BaseModel):
    total_requests: int
    error_count: int
    error_rate: float
    error_breakdown: Dict[str, int] = Field(default_factory=dict)


class CostAnalysis(BaseModel):
    total_cost_usd: float
    avg_cost_per_request: float
    total_tokens: int
    total_prompt_tokens: int
    total_completion_tokens: int
    by_model: Dict[str, Dict[str, Any]] = Field(default_factory=dict)


class UsageTrend(BaseModel):
    period: str
    request_count: int
    token_count: int
    avg_latency_ms: float
