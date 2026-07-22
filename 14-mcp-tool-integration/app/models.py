from pydantic import BaseModel, Field
from typing import Any, Optional


class ToolCallRequest(BaseModel):
    """工具调用请求"""
    server_name: str = Field(..., description="服务器名称")
    tool_name: str = Field(..., description="工具名称")
    arguments: dict[str, Any] = Field(default_factory=dict, description="工具参数")
    timeout: Optional[float] = Field(None, description="超时时间(秒)")


class ToolSearchRequest(BaseModel):
    """工具搜索请求"""
    keyword: Optional[str] = Field(None, description="搜索关键词")
    tag: Optional[str] = Field(None, description="按标签过滤")
    server_name: Optional[str] = Field(None, description="按服务器过滤")
    role: str = Field(default="default", description="用户角色")


class AgentExecuteRequest(BaseModel):
    """Agent 执行请求"""
    user_input: str = Field(..., description="用户自然语言输入")
    role: Optional[str] = Field(None, description="用户角色")
    auto_call: bool = Field(default=True, description="是否自动调用")


class PermissionSetRequest(BaseModel):
    """权限设置请求"""
    tool_name: str = Field(..., description="工具名称")
    level: str = Field(..., description="权限级别: allow/deny/require_approval")
    roles: list[str] = Field(default_factory=lambda: ["*"], description="角色列表")
    expires_in: Optional[float] = Field(None, description="过期时间(秒)")


class ServerRegisterRequest(BaseModel):
    """服务器注册请求（用于 API）"""
    server_name: str = Field(..., description="服务器名称")
    builtin: Optional[str] = Field(None, description="使用内置服务器: filesystem/calculator/database")


class ChainExecuteRequest(BaseModel):
    """链式调用请求"""
    steps: list[dict[str, Any]] = Field(..., description="调用步骤列表")
    role: str = Field(default="default", description="用户角色")


class ApiResponse(BaseModel):
    """统一 API 响应"""
    success: bool = True
    data: Any = None
    error: str = ""
