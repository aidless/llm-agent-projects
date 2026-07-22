"""Pydantic 模型 - API 请求/响应模型"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ========== 推理 API ==========

class ChatMessage(BaseModel):
    role: str = Field(..., description="消息角色: system/user/assistant")
    content: str = Field(..., description="消息内容")


class ChatCompletionRequest(BaseModel):
    """OpenAI 兼容的 Chat Completion 请求"""
    model: str = Field(default="mock-llm-7b", description="模型名称")
    messages: List[ChatMessage] = Field(..., description="对话消息列表")
    max_tokens: int = Field(default=128, ge=1, le=4096, description="最大生成 token 数")
    temperature: float = Field(default=1.0, ge=0.0, le=2.0, description="温度")
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-P 采样")
    stream: bool = Field(default=False, description="是否流式输出")
    priority: int = Field(default=0, ge=0, le=10, description="请求优先级")


class ChatCompletionChoice(BaseModel):
    index: int = Field(default=0)
    message: ChatMessage
    finish_reason: str = Field(default="stop")


class UsageInfo(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(BaseModel):
    """OpenAI 兼容的 Chat Completion 响应"""
    id: str
    object: str = Field(default="chat.completion")
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: UsageInfo


class CompletionRequest(BaseModel):
    """文本补全请求"""
    model: str = Field(default="mock-llm-7b")
    prompt: str = Field(..., description="输入提示")
    max_tokens: int = Field(default=128, ge=1, le=4096)
    temperature: float = Field(default=1.0, ge=0.0, le=2.0)
    stream: bool = Field(default=False)
    priority: int = Field(default=0, ge=0, le=10)


class CompletionResponse(BaseModel):
    """文本补全响应"""
    id: str
    object: str = Field(default="text_completion")
    created: int
    model: str
    choices: List[Dict[str, Any]]
    usage: UsageInfo


# ========== 模型/配置 ==========

class ModelInfo(BaseModel):
    id: str
    object: str = Field(default="model")
    owned_by: str = Field(default="local")
    quantization: str = Field(default="fp16")


class ModelListResponse(BaseModel):
    object: str = Field(default="list")
    data: List[ModelInfo]


class QuantizationConfigRequest(BaseModel):
    """量化配置请求"""
    format: str = Field(..., description="量化格式")
    num_parameters: int = Field(default=7_000_000_000)
    num_layers: int = Field(default=32)
    hidden_size: int = Field(default=4096)


class QuantizationCompareRequest(BaseModel):
    """量化对比请求"""
    num_parameters: int = Field(default=7_000_000_000)
    num_layers: int = Field(default=32)
    hidden_size: int = Field(default=4096)


# ========== 监控 ==========

class MetricsResponse(BaseModel):
    timestamp: float
    metrics: Dict[str, Any]
    summary: Dict[str, Any]


class GPUReportResponse(BaseModel):
    timestamp: float
    gpu_utilization: float
    gpu_memory_usage_gb: float
    kv_cache_utilization: float
    active_requests: int


class InferenceReportResponse(BaseModel):
    timestamp: float
    total_requests: float
    total_tokens: float
    performance: Dict[str, float]


# ========== 缓存 ==========

class CacheStatsResponse(BaseModel):
    total_allocations: int
    total_evictions: int
    total_hits: int
    total_misses: int
    hit_rate: float
    current_usage_bytes: int
    peak_usage_bytes: int


class PagedCacheUsageResponse(BaseModel):
    block_size: int
    total_blocks: int
    total_layers: int
    used_blocks: int
    available_blocks: int
    overall_utilization: float
    num_active_requests: int


# ========== 代理 ==========

class BackendStatus(BaseModel):
    name: str
    url: str
    healthy: bool
    weight: int
    active_connections: int
    total_requests: int
    total_errors: int
    avg_latency_ms: float


class RouterStatsResponse(BaseModel):
    strategy: str
    total_backends: int
    available_backends: int
    total_requests: int
    total_errors: int
    error_rate: float


class CircuitBreakerStatus(BaseModel):
    name: str
    state: str
    failure_count: int
    success_count: int
    events: List[Dict[str, Any]]


# ========== 通用 ==========

class HealthResponse(BaseModel):
    status: str = Field(default="healthy")
    version: str = Field(default="1.0.0")
    uptime_seconds: float = 0.0


class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
