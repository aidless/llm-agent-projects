"""Pydantic 数据模型。"""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ---- 评估相关 ----

class EvaluationRequest(BaseModel):
    predictions: List[str] = Field(..., description="预测结果列表")
    references: List[str] = Field(..., description="参考答案列表")
    metric_names: Optional[List[str]] = Field(None, description="指标名称列表")


class EvaluationResponse(BaseModel):
    num_samples: int
    metrics_used: List[str]
    results: Dict[str, Any]
    elapsed_seconds: float


# ---- 数据集相关 ----

class DatasetCreateRequest(BaseModel):
    name: str = Field(..., description="数据集名称")
    input_key: str = Field("input", description="输入字段名")
    output_key: str = Field("output", description="输出字段名")
    data: Optional[List[Dict[str, str]]] = Field(None, description="数据")


class DatasetImportRequest(BaseModel):
    file_path: str = Field(..., description="文件路径")
    name: Optional[str] = Field(None, description="数据集名称")
    format: str = Field("json", description="文件格式: json/jsonl/csv")


class DatasetSampleRequest(BaseModel):
    n: int = Field(10, description="采样数量")
    strategy: str = Field("random", description="采样策略: random/first/last")
    seed: Optional[int] = Field(None, description="随机种子")


# ---- Prompt 版本相关 ----

class PromptVersionCreate(BaseModel):
    version_id: str = Field(..., description="版本 ID")
    name: str = Field(..., description="版本名称")
    prompt_template: str = Field(..., description="Prompt 模板")
    description: str = Field("", description="版本描述")


class ABTestRequest(BaseModel):
    version_a_id: str = Field(..., description="版本 A ID")
    version_b_id: str = Field(..., description="版本 B ID")
    predictions_a: List[str] = Field(..., description="版本 A 预测结果")
    predictions_b: List[str] = Field(..., description="版本 B 预测结果")
    references: List[str] = Field(..., description="参考答案")
    metric_names: Optional[List[str]] = Field(None, description="指标名称列表")
    alpha: float = Field(0.05, description="显著性水平")


# ---- LLM Judge 相关 ----

class JudgeRequest(BaseModel):
    questions: List[str] = Field(..., description="问题列表")
    predictions: List[str] = Field(..., description="预测结果列表")
    references: List[str] = Field(..., description="参考答案列表")
    dimensions: Optional[List[str]] = Field(None, description="评估维度列表")
    num_judges: int = Field(1, description="Judge 数量")


# ---- 报告相关 ----

class ReportRequest(BaseModel):
    evaluation_result: Dict[str, Any] = Field(..., description="评估结果")
    format: str = Field("json", description="报告格式: json/markdown")


# ---- Prompt 优化相关 ----

class PromptAnalyzeRequest(BaseModel):
    prompt: str = Field(..., description="待分析的 Prompt")


class FewShotSelectRequest(BaseModel):
    dataset_name: str = Field(..., description="数据集名称")
    query: str = Field("", description="当前查询")
    n: int = Field(3, description="选择示例数量")


class TemplateParameterizeRequest(BaseModel):
    template: str = Field(..., description="Prompt 模板")