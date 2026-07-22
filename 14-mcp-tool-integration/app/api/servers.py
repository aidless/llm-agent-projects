"""服务器管理 API 路由"""

from fastapi import APIRouter, Depends

from registry.tool_registry import ToolRegistry
from app.models import ApiResponse, ServerRegisterRequest

router = APIRouter(prefix="/servers", tags=["servers"])


def get_registry():
    from app.main import registry
    return registry


def get_client():
    from app.main import client
    return client


@router.get("/list", response_model=ApiResponse)
async def list_servers(reg: ToolRegistry = Depends(get_registry)):
    """列出所有已注册的服务器"""
    return ApiResponse(data={
        "servers": reg.list_servers(),
        "count": reg.server_count,
    })


@router.post("/register", response_model=ApiResponse)
async def register_server(
    req: ServerRegisterRequest,
    reg: ToolRegistry = Depends(get_registry),
):
    """注册服务器"""
    server_name = req.server_name

    # 检查是否已注册
    existing = reg.get_server(server_name)
    if existing:
        return ApiResponse(success=False, error=f"Server already registered: {server_name}")

    # 使用内置服务器
    if req.builtin:
        from server.builtin_servers import (
            CalculatorServer,
            DatabaseServer,
            FileSystemServer,
        )
        builtin_map = {
            "filesystem": FileSystemServer,
            "calculator": CalculatorServer,
            "database": DatabaseServer,
        }
        server_cls = builtin_map.get(req.builtin)
        if server_cls is None:
            return ApiResponse(success=False, error=f"Unknown builtin server: {req.builtin}")

        server = server_cls()
    else:
        return ApiResponse(success=False, error="Must specify a builtin server type")

    result = await reg.register_server(server)
    return ApiResponse(data=result)


@router.delete("/{server_name}", response_model=ApiResponse)
async def unregister_server(server_name: str, reg: ToolRegistry = Depends(get_registry)):
    """注销服务器"""
    success = await reg.unregister_server(server_name)
    if success:
        return ApiResponse(data={"status": "unregistered", "name": server_name})
    return ApiResponse(success=False, error=f"Server not found: {server_name}")


@router.get("/{server_name}/info", response_model=ApiResponse)
async def server_info(server_name: str, reg: ToolRegistry = Depends(get_registry)):
    """获取服务器详情"""
    reg_info = reg.get_server(server_name)
    if reg_info is None:
        return ApiResponse(success=False, error=f"Server not found: {server_name}")
    return ApiResponse(data=reg_info.to_dict())


@router.get("/{server_name}/tools", response_model=ApiResponse)
async def server_tools(server_name: str, reg: ToolRegistry = Depends(get_registry)):
    """获取服务器的工具列表"""
    tools = reg.list_tools(server_name=server_name)
    return ApiResponse(data={"tools": tools, "count": len(tools)})