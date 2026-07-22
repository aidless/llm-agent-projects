"""数据集管理 API 路由。"""

from typing import Any, Dict

from fastapi import APIRouter, HTTPException

from app.models import DatasetCreateRequest, DatasetImportRequest, DatasetSampleRequest
from datasets import DatasetManager, DataAugmenter

router = APIRouter(prefix="/api/v1/datasets", tags=["dataset"])

# 全局实例
manager = DatasetManager()
augmenter = DataAugmenter()


@router.get("/")
def list_datasets():
    """列出所有数据集。"""
    return {"datasets": manager.list_datasets()}


@router.post("/")
def create_dataset(request: DatasetCreateRequest):
    """创建数据集。"""
    result = manager.create_dataset(
        name=request.name,
        input_key=request.input_key,
        output_key=request.output_key,
        data=request.data,
    )
    return result


@router.post("/import")
def import_dataset(request: DatasetImportRequest):
    """导入数据集。"""
    try:
        if request.format == "json":
            result = manager.import_json(request.file_path, name=request.name)
        elif request.format == "jsonl":
            result = manager.import_jsonl(request.file_path, name=request.name)
        elif request.format == "csv":
            result = manager.import_csv(request.file_path, name=request.name)
        else:
            raise HTTPException(status_code=400, detail=f"不支持的格式: {request.format}")
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/{name}")
def get_dataset(name: str):
    """获取数据集详情。"""
    try:
        return manager.get_dataset(name)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{name}")
def delete_dataset(name: str):
    """删除数据集。"""
    try:
        manager.delete_dataset(name)
        return {"message": f"数据集 {name} 已删除"}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/{name}/sample")
def sample_dataset(name: str, request: DatasetSampleRequest):
    """从数据集采样。"""
    try:
        return manager.sample(name, request.n, request.strategy, request.seed)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/augment")
def augment_data(data: list[Dict[str, str]], methods: list[str] | None = None):
    """数据增强。"""
    result = augmenter.augment_dataset(
        data=data,
        methods=methods,
    )
    return {
        "original_size": len(data),
        "augmented_size": len(result),
        "new_samples": len(result) - len(data),
        "data": result,
    }