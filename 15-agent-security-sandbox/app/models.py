"""Pydantic 请求/响应模型"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field


class ExecuteRequest(BaseModel):
    """代码执行请求"""
    code: str = Field(..., description="要执行的 Python 代码", min_length=1, max_length=100000)
    policy_name: str = Field(default="medium", description="安全策略名称")
    variables: Optional[Dict[str, Any]] = Field(default=None, description="预定义变量")


class ExecuteResponse(BaseModel):
    """代码执行响应"""
    execution_id: str
    success: bool
    output: str = ""
    error: str = ""
    error_type: str = ""
    result_value: Optional[str] = None
    resource_usage: Dict[str, Any] = {}
    security_violations: List[Dict[str, Any]] = []
    execution_time: float = 0.0


class PolicyCreateRequest(BaseModel):
    """创建策略请求"""
    name: str = Field(..., description="策略名称", min_length=1)
    level: str = Field(default="MEDIUM", description="安全等级 LOW/MEDIUM/HIGH/STRICT")
    description: str = Field(default="", description="策略描述")
    resource_limits: Optional[Dict[str, Any]] = None
    module_policy: Optional[Dict[str, Any]] = None
    filesystem_policy: Optional[Dict[str, Any]] = None
    network_policy: Optional[Dict[str, Any]] = None


class PolicyInheritRequest(BaseModel):
    """继承策略请求"""
    base_name: str = Field(..., description="基础策略名称")
    overrides: Dict[str, Any] = Field(default_factory=dict, description="覆盖配置")
    new_name: str = Field(..., description="新策略名称")
    description: str = Field(default="", description="策略描述")


class PolicyCombineRequest(BaseModel):
    """组合策略请求"""
    policy_names: List[str] = Field(..., description="要组合的策略名称列表")
    combined_name: str = Field(..., description="组合后的策略名称")


class AuditQueryParams(BaseModel):
    """审计查询参数"""
    execution_id: Optional[str] = None
    event_type: Optional[str] = None
    level: Optional[str] = None
    limit: int = Field(default=100, ge=1, le=1000)
    offset: int = Field(default=0, ge=0)


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = "healthy"
    version: str = "1.0.0"
    total_executions: int = 0
    active_policies: int = 0
    log_entries: int = 0
    alerts: int = 0


class ErrorResponse(BaseModel):
    """错误响应"""
    error: str
    detail: str = ""
