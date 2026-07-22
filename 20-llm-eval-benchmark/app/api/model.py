"""
app/api/model.py - 模型管理 API 路由
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from models import ModelManager
from app.models import ModelRegisterRequest, ModelUpdateRequest, ModelResponse

router = APIRouter(prefix="/api/models", tags=["models"])

# 全局模型管理器
model_manager = ModelManager()


@router.post("/register", response_model=ModelResponse)
async def register_model(request: ModelRegisterRequest):
    """注册新模型."""
    try:
        model = model_manager.register(
            name=request.name,
            api_endpoint=request.api_endpoint,
            parameters=request.parameters,
            provider=request.provider,
            description=request.description,
            version=request.version,
        )
        return _to_response(model)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))


@router.get("/list")
async def list_models():
    """列出所有已注册模型."""
    models = model_manager.list_models()
    return {
        "total": len(models),
        "models": [_to_response(m) for m in models],
    }


@router.get("/{model_name}", response_model=ModelResponse)
async def get_model(model_name: str):
    """获取模型信息."""
    model = model_manager.get(model_name)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")
    return _to_response(model)


@router.put("/{model_name}", response_model=ModelResponse)
async def update_model(model_name: str, request: ModelUpdateRequest):
    """更新模型信息."""
    update_data = request.model_dump(exclude_unset=True)
    version = update_data.pop("version", None)

    model = model_manager.update(model_name, **update_data)
    if not model:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")

    if version:
        model_manager.add_version(model_name, version)

    return _to_response(model_manager.get(model_name))


@router.delete("/{model_name}")
async def unregister_model(model_name: str):
    """取消注册模型."""
    if model_manager.unregister(model_name):
        return {"status": "ok", "message": f"Model '{model_name}' unregistered"}
    raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found")


def _to_response(model) -> ModelResponse:
    """将 ModelInfo 转换为 API 响应."""
    return ModelResponse(
        name=model.name,
        api_endpoint=model.api_endpoint,
        parameters=model.parameters,
        provider=model.provider,
        description=model.description,
        latest_version=model.latest_version,
        versions=[
            {"version": v.version, "is_active": v.is_active}
            for v in model.versions
        ],
    )