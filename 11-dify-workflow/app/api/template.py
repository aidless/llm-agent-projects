"""模板 API"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.models import (
    TemplateResponse,
    TemplateCreateRequest,
    WorkflowResponse,
    MessageResponse,
    ListResponse,
)

router = APIRouter(prefix="/api/v1/templates", tags=["templates"])


def get_store():
    from app.main import workflow_store
    return workflow_store


@router.get("", response_model=ListResponse)
def list_templates():
    """列出所有内置模板"""
    from templates.builtin import list_templates as list_builtin
    items = list_builtin()
    return ListResponse(items=items, total=len(items))


@router.get("/{template_id}")
def get_template(template_id: str):
    """获取模板详情"""
    from templates.builtin import get_template
    try:
        return get_template(template_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/create", response_model=WorkflowResponse, status_code=201)
def create_from_template(req: TemplateCreateRequest):
    """从模板创建工作流"""
    from templates.builtin import get_template
    store = get_store()

    try:
        template = get_template(req.template_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # 应用覆盖
    workflow_data = template.copy()
    if req.name:
        workflow_data["name"] = req.name
    workflow_data.update(req.overrides)

    # 修改 ID 以避免与模板 ID 冲突
    workflow_data["id"] = f"user-{workflow_data['id']}"

    try:
        wf = store.create(workflow_data)
        return wf
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
