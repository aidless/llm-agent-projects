"""Pydantic 模型 - 请求/响应数据结构"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ── 工作流模型 ────────────────────────────────────────────

class NodeConfig(BaseModel):
    type: str = Field(..., description="节点类型")
    inputs: Dict[str, Any] = Field(default_factory=dict)
    retry: int = Field(default=0, ge=0, le=10)


class WorkflowCreate(BaseModel):
    id: Optional[str] = Field(default=None, description="工作流 ID，不提供则从 name 生成")
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field(default="")
    nodes: Dict[str, Any] = Field(default_factory=dict)
    edges: List[Dict[str, str]] = Field(default_factory=list)
    global_inputs: Dict[str, Any] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


class WorkflowUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    nodes: Optional[Dict[str, Any]] = None
    edges: Optional[List[Dict[str, str]]] = None
    global_inputs: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    metadata: Optional[Dict[str, Any]] = None


class WorkflowResponse(BaseModel):
    id: str
    name: str
    description: str = ""
    nodes: Dict[str, Any] = {}
    edges: List[Dict[str, str]] = []
    global_inputs: Dict[str, Any] = {}
    version: int = 1
    created_at: str = ""
    updated_at: str = ""
    tags: List[str] = []
    metadata: Dict[str, Any] = {}


class WorkflowImport(BaseModel):
    data: Dict[str, Any]


class WorkflowExport(BaseModel):
    format: str = Field(default="json", pattern="^(json|yaml)$")


# ── 执行模型 ──────────────────────────────────────────────

class ExecutionRequest(BaseModel):
    workflow_id: str
    inputs: Dict[str, Any] = Field(default_factory=dict)
    env_vars: Dict[str, str] = Field(default_factory=dict)


class NodeResultResponse(BaseModel):
    node_id: str
    status: str
    output: Dict[str, Any] = {}
    error: Optional[str] = None
    elapsed_ms: float = 0.0


class ExecutionResponse(BaseModel):
    execution_id: str
    workflow_id: str
    status: str
    node_results: Dict[str, Any] = {}
    output: Dict[str, Any] = {}
    error: Optional[str] = None
    elapsed_ms: float = 0.0


# ── 模板模型 ──────────────────────────────────────────────

class TemplateResponse(BaseModel):
    id: str
    name: str
    description: str = ""
    tags: List[str] = []
    node_count: int = 0
    edge_count: int = 0


class TemplateCreateRequest(BaseModel):
    template_id: str
    name: Optional[str] = None
    overrides: Dict[str, Any] = Field(default_factory=dict)


# ── 通用响应 ──────────────────────────────────────────────

class MessageResponse(BaseModel):
    message: str
    success: bool = True


class ListResponse(BaseModel):
    items: List[Any] = []
    total: int = 0
