"""报告 API 路由。"""

from fastapi import APIRouter, HTTPException

from app.models import ReportRequest
from pipeline.evaluator import EvaluationPipeline

router = APIRouter(prefix="/api/v1/reports", tags=["report"])

pipeline = EvaluationPipeline()


@router.post("/generate")
def generate_report(request: ReportRequest):
    """生成评估报告。"""
    try:
        report = pipeline.generate_report(
            evaluation_result=request.evaluation_result,
            format=request.format,
        )
        return {
            "format": request.format,
            "report": report,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))