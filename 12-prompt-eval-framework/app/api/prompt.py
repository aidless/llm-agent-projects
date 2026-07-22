"""Prompt 版本管理 API 路由。"""

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from app.models import PromptVersionCreate, ABTestRequest, PromptAnalyzeRequest, FewShotSelectRequest, TemplateParameterizeRequest
from ab_test.runner import ABTestRunner, PromptVersion
from pipeline.optimizer import PromptOptimizer
from datasets import DatasetManager

router = APIRouter(prefix="/api/v1/prompts", tags=["prompt"])

# 全局实例
ab_runner = ABTestRunner()
optimizer = PromptOptimizer()
dataset_manager = DatasetManager()


@router.get("/versions")
def list_versions():
    """列出所有 Prompt 版本。"""
    return {"versions": ab_runner.list_versions()}


@router.post("/versions")
def create_version(request: PromptVersionCreate):
    """创建 Prompt 版本。"""
    version = PromptVersion(
        version_id=request.version_id,
        name=request.name,
        prompt_template=request.prompt_template,
        description=request.description,
    )
    ab_runner.add_version(version)
    return {"message": f"版本 {request.version_id} 已创建", "version_id": request.version_id}


@router.delete("/versions/{version_id}")
def delete_version(version_id: str):
    """删除 Prompt 版本。"""
    try:
        ab_runner.remove_version(version_id)
        return {"message": f"版本 {version_id} 已删除"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/ab-test")
def run_ab_test(request: ABTestRequest):
    """运行 A/B 测试。"""
    try:
        result = ab_runner.run_test(
            version_a_id=request.version_a_id,
            version_b_id=request.version_b_id,
            predictions_a=request.predictions_a,
            predictions_b=request.predictions_b,
            references=request.references,
            metric_names=request.metric_names,
            alpha=request.alpha,
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/analyze")
def analyze_prompt(request: PromptAnalyzeRequest):
    """分析 Prompt 并提供改进建议。"""
    return optimizer.analyze_prompt(request.prompt)


@router.post("/few-shot/select")
def select_few_shot(request: FewShotSelectRequest):
    """自动选择 few-shot 示例。"""
    try:
        dataset = dataset_manager.get_dataset(request.dataset_name)
        examples = optimizer.select_few_shot_examples(
            dataset=dataset["data"],
            input_key=dataset["input_key"],
            output_key=dataset["output_key"],
            query=request.query,
            n=request.n,
        )
        return {"examples": examples, "count": len(examples)}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/template/parameterize")
def parameterize_template(request: TemplateParameterizeRequest):
    """模板变量化。"""
    return optimizer.parameterize_template(request.template)
