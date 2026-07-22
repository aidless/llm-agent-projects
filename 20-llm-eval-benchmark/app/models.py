"""
app/models.py - Pydantic 数据模型

定义所有 API 请求和响应的数据结构。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field, ConfigDict


# ==================== 评测相关 ====================

class _BaseModel(BaseModel):
    """基类，统一关闭 model_ 命名空间保护."""
    model_config = ConfigDict(protected_namespaces=())


class EvalRequest(_BaseModel):
    """评测请求."""
    model_name: str = Field(..., description="模型名称", examples=["gpt-4"])
    benchmark: str = Field(..., description="基准名称", examples=["mmlu"])
    max_concurrent: int = Field(5, ge=1, le=50, description="最大并发数")
    timeout_seconds: float = Field(30.0, ge=1.0, le=300.0, description="超时时间(秒)")
    max_retries: int = Field(2, ge=0, le=10, description="最大重试次数")
    question_limit: Optional[int] = Field(None, ge=1, description="题目数量限制")
    enable_cache: bool = Field(True, description="是否启用缓存")


class EvalProgressResponse(_BaseModel):
    """评测进度响应."""
    total: int = 0
    completed: int = 0
    correct: int = 0
    errors: int = 0
    accuracy: float = 0.0
    progress_pct: float = 0.0
    elapsed_seconds: float = 0.0


class BenchmarkResultResponse(_BaseModel):
    """单个评测结果."""
    question_id: int
    question: str
    expected: Optional[Any] = None
    predicted: Optional[Any] = None
    correct: bool = False
    score: float = 0.0
    error: Optional[str] = None


class ScoreDetail(_BaseModel):
    """评分详情."""
    metric: str
    value: float
    confidence_interval: Optional[Dict[str, float]] = None
    details: Optional[Dict[str, Any]] = None


class EvalResponse(_BaseModel):
    """评测完成响应."""
    model_name: str
    benchmark: str
    total_questions: int
    correct: int
    accuracy: float
    eval_time_seconds: float
    scores: Dict[str, ScoreDetail] = {}
    results: List[BenchmarkResultResponse] = []


# ==================== 模型管理 ====================

class ModelRegisterRequest(_BaseModel):
    """模型注册请求."""
    name: str = Field(..., description="模型名称")
    api_endpoint: str = Field("", description="API 端点")
    parameters: Optional[str] = Field(None, description="参数量 (如 '7B', '13B')")
    provider: str = Field("unknown", description="提供商")
    description: str = Field("", description="模型描述")
    version: str = Field("v1", description="初始版本")


class ModelUpdateRequest(_BaseModel):
    """模型更新请求."""
    api_endpoint: Optional[str] = None
    parameters: Optional[str] = None
    provider: Optional[str] = None
    description: Optional[str] = None
    version: Optional[str] = None


class ModelResponse(_BaseModel):
    """模型信息响应."""
    name: str
    api_endpoint: str = ""
    parameters: Optional[str] = None
    provider: str = "unknown"
    description: str = ""
    latest_version: Optional[str] = None
    versions: List[Dict[str, Any]] = []


# ==================== 排行榜 ====================

class LeaderboardEntryResponse(_BaseModel):
    """排行榜条目."""
    rank: int
    model_name: str
    overall_score: float
    overall_score_pct: str
    benchmark_scores: Dict[str, float] = {}


class LeaderboardResponse(_BaseModel):
    """排行榜响应."""
    total_models: int = 0
    available_benchmarks: List[str] = []
    overall_ranking: List[LeaderboardEntryResponse] = []


# ==================== 报告 ====================

class ReportRequest(_BaseModel):
    """报告生成请求."""
    format: str = Field("json", description="报告格式: json, markdown, csv")
    model_names: Optional[List[str]] = Field(None, description="指定模型")
    benchmark_name: Optional[str] = Field(None, description="指定基准")
    include_details: bool = Field(False, description="是否包含详细结果")
    generate_charts: bool = Field(True, description="是否生成图表")


class ComparisonRequest(_BaseModel):
    """模型对比请求."""
    model_a: str
    model_b: str
    benchmark_name: Optional[str] = None


class ComparisonResponse(_BaseModel):
    """模型对比响应."""
    model_a: str
    model_b: str
    comparisons: List[Dict[str, Any]] = []
    ranking: List[Dict[str, Any]] = []


class HealthResponse(_BaseModel):
    """健康检查响应."""
    status: str = "ok"
    version: str = "1.0.0"
    benchmarks_available: List[str] = []