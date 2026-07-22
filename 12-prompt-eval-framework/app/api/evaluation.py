"""评估 API 路由。"""

from typing import Any, Dict

from fastapi import APIRouter

from app.models import EvaluationRequest, EvaluationResponse
from metrics import MetricRegistry
from pipeline.evaluator import EvaluationPipeline
from judge import LLMJudge, ConsensusJudge
from judge.llm_judge import JUDGE_TEMPLATES
from app.models import JudgeRequest

router = APIRouter(prefix="/api/v1", tags=["evaluation"])

# 全局实例
registry = MetricRegistry()
pipeline = EvaluationPipeline(metric_registry=registry)


@router.get("/metrics")
def list_metrics():
    """列出所有可用的评估指标。"""
    return {"metrics": registry.list_metrics()}


@router.post("/evaluate", response_model=EvaluationResponse)
def evaluate(request: EvaluationRequest):
    """执行评估。"""
    result = pipeline.evaluate(
        predictions=request.predictions,
        references=request.references,
        metric_names=request.metric_names,
    )
    return result


@router.post("/judge")
def judge_evaluate(request: JudgeRequest) -> Dict[str, Any]:
    """LLM-as-Judge 评估。"""
    if request.num_judges > 1:
        judge = ConsensusJudge(
            num_judges=request.num_judges,
            dimensions=request.dimensions,
        )
        results = judge.judge_batch(
            questions=request.questions,
            predictions=request.predictions,
            references=request.references,
        )
    else:
        judge = LLMJudge(dimensions=request.dimensions)
        results = judge.judge_batch(
            questions=request.questions,
            predictions=request.predictions,
            references=request.references,
        )

    return {
        "num_samples": len(request.predictions),
        "results": results,
        "judge_type": "consensus" if request.num_judges > 1 else "single",
        "num_judges": request.num_judges,
    }


@router.get("/judge/dimensions")
def list_judge_dimensions():
    """列出所有可用的 Judge 评估维度。"""
    return {
        "dimensions": [
            {"key": k, "name": v["name"], "description": v["description"]}
            for k, v in JUDGE_TEMPLATES.items()
        ]
    }