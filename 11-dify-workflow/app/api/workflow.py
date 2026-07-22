"""工作流 CRUD API"""

from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, HTTPException

from app.models import (
    WorkflowCreate,
    WorkflowUpdate,
    WorkflowResponse,
    MessageResponse,
    ListResponse,
    WorkflowImport,
)
from engine.errors import WorkflowNotFoundError

router = APIRouter(prefix="/api/v1/workflows", tags=["workflows"])


def get_store():
    """延迟导入以避免循环依赖"""
    from app.main import workflow_store
    return workflow_store


@router.post("", response_model=WorkflowResponse, status_code=201)
def create_workflow(data: WorkflowCreate):
    """创建工作流"""
    store = get_store()
    try:
        wf = store.create(data.model_dump())
        return wf
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("", response_model=ListResponse)
def list_workflows(tag: Optional[str] = None):
    """列出所有工作流"""
    store = get_store()
    items = store.list_all(tag=tag)
    return ListResponse(items=items, total=len(items))


@router.get("/{workflow_id}", response_model=WorkflowResponse)
def get_workflow(workflow_id: str):
    """获取工作流详情"""
    store = get_store()
    try:
        return store.get(workflow_id)
    except WorkflowNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.put("/{workflow_id}", response_model=WorkflowResponse)
def update_workflow(workflow_id: str, data: WorkflowUpdate):
    """更新工作流"""
    store = get_store()
    try:
        updates = data.model_dump(exclude_none=True)
        return store.update(workflow_id, updates)
    except WorkflowNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.delete("/{workflow_id}", response_model=MessageResponse)
def delete_workflow(workflow_id: str):
    """删除工作流"""
    store = get_store()
    try:
        store.delete(workflow_id)
        return MessageResponse(message=f"工作流 '{workflow_id}' 已删除")
    except WorkflowNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{workflow_id}/versions", response_model=ListResponse)
def list_workflow_versions(workflow_id: str):
    """列出工作流所有版本"""
    store = get_store()
    try:
        versions = store.list_versions(workflow_id)
        return ListResponse(items=versions, total=len(versions))
    except WorkflowNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/{workflow_id}/export")
def export_workflow(workflow_id: str):
    """导出工作流"""
    store = get_store()
    try:
        return store.export_json(workflow_id)
    except WorkflowNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/import", response_model=WorkflowResponse, status_code=201)
def import_workflow(data: WorkflowImport):
    """导入工作流"""
    store = get_store()
    try:
        wf = store.import_json(data.model_dump()["data"])
        return wf
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))
