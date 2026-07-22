"""注册中心 API 路由"""

from fastapi import APIRouter, Depends

from registry.tool_registry import ToolRegistry
from registry.permission import PermissionManager
from app.models import (
    ApiResponse,
    PermissionSetRequest,
    ToolSearchRequest,
)

router = APIRouter(prefix="/registry", tags=["registry"])


def get_registry():
    """获取全局注册中心（由 main.py 注入）"""
    from app.main import registry
    return registry


@router.get("/servers", response_model=ApiResponse)
async def list_servers(reg: ToolRegistry = Depends(get_registry)):
    """列出所有已注册的服务器"""
    return ApiResponse(data=reg.list_servers())


@router.get("/tools", response_model=ApiResponse)
async def search_tools(
    keyword: str = None,
    tag: str = None,
    server_name: str = None,
    role: str = "default",
    reg: ToolRegistry = Depends(get_registry),
):
    """搜索工具"""
    tools = reg.list_tools(
        server_name=server_name,
        tag=tag,
        keyword=keyword,
        role=role,
    )
    return ApiResponse(data={"tools": tools, "count": len(tools)})


@router.get("/heartbeat/{server_name}", response_model=ApiResponse)
async def check_heartbeat(server_name: str, reg: ToolRegistry = Depends(get_registry)):
    """检查服务器心跳"""
    status = await reg.check_heartbeat(server_name)
    return ApiResponse(data={
        "server_name": status.server_name,
        "alive": status.alive,
        "last_heartbeat": status.last_heartbeat,
        "response_time_ms": status.response_time_ms,
    })


@router.get("/stats", response_model=ApiResponse)
async def get_usage_stats(
    server_name: str = None,
    tool_name: str = None,
    reg: ToolRegistry = Depends(get_registry),
):
    """获取使用统计"""
    stats = reg.get_usage_stats(server_name=server_name, tool_name=tool_name)
    return ApiResponse(data=stats)


@router.post("/permissions", response_model=ApiResponse)
async def set_permission(
    req: PermissionSetRequest,
    reg: ToolRegistry = Depends(get_registry),
):
    """设置工具权限"""
    perm_mgr: PermissionManager = reg.permissions
    if req.level == "allow":
        perm_mgr.grant(req.tool_name, roles=req.roles, expires_in=req.expires_in)
    elif req.level == "deny":
        perm_mgr.deny(req.tool_name, roles=req.roles, expires_in=req.expires_in)
    elif req.level == "require_approval":
        perm_mgr.set_require_approval(req.tool_name, roles=req.roles, expires_in=req.expires_in)
    else:
        return ApiResponse(success=False, error=f"Invalid permission level: {req.level}")

    return ApiResponse(data={"tool_name": req.tool_name, "level": req.level})


@router.get("/permissions", response_model=ApiResponse)
async def list_permissions(reg: ToolRegistry = Depends(get_registry)):
    """列出所有权限规则"""
    rules = reg.permissions.list_rules()
    return ApiResponse(data={"rules": rules, "count": len(rules)})


@router.get("/permissions/{tool_name}", response_model=ApiResponse)
async def get_tool_permissions(tool_name: str, reg: ToolRegistry = Depends(get_registry)):
    """获取工具权限"""
    rules = reg.permissions.get_tool_permissions(tool_name)
    return ApiResponse(data={"tool_name": tool_name, "rules": rules})