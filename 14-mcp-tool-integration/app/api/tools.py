"""工具调用 API 路由"""

from fastapi import APIRouter, Depends

from registry.tool_registry import ToolRegistry
from client.mcp_client import MCPClient
from app.models import (
    AgentExecuteRequest,
    ApiResponse,
    ChainExecuteRequest,
    ToolCallRequest,
)
from agent.tool_agent import ToolAgent
from agent.chain_executor import ChainStep

router = APIRouter(prefix="/tools", tags=["tools"])


def get_registry():
    from app.main import registry
    return registry


def get_client():
    from app.main import client
    return client


def get_agent():
    from app.main import agent
    return agent


@router.post("/call", response_model=ApiResponse)
async def call_tool(
    req: ToolCallRequest,
    reg: ToolRegistry = Depends(get_registry),
    client: MCPClient = Depends(get_client),
):
    """调用指定工具"""
    # 权限检查
    perm = reg.permissions.check_permission(req.tool_name)
    from registry.permission import PermissionLevel
    if perm == PermissionLevel.DENY:
        return ApiResponse(success=False, error=f"Permission denied: {req.tool_name}")

    try:
        result = await client.call_tool(
            server_name=req.server_name,
            tool_name=req.tool_name,
            arguments=req.arguments,
            timeout=req.timeout,
        )
        reg.record_usage(
            tool_name=req.tool_name,
            server_name=req.server_name,
            success=not result.is_error,
        )
        return ApiResponse(data=result.to_dict())
    except Exception as e:
        return ApiResponse(success=False, error=str(e))


@router.get("/search", response_model=ApiResponse)
async def search_tools(
    keyword: str = None,
    tag: str = None,
    server_name: str = None,
    role: str = "default",
    reg: ToolRegistry = Depends(get_registry),
):
    """搜索工具"""
    tools = reg.list_tools(
        keyword=keyword,
        tag=tag,
        server_name=server_name,
        role=role,
    )
    return ApiResponse(data={"tools": tools, "count": len(tools)})


@router.post("/agent/execute", response_model=ApiResponse)
async def agent_execute(
    req: AgentExecuteRequest,
    agent: ToolAgent = Depends(get_agent),
):
    """通过 Agent 执行任务"""
    result = await agent.execute(
        user_input=req.user_input,
        role=req.role,
        auto_call=req.auto_call,
    )
    return ApiResponse(data=result.to_dict())


@router.post("/chain/execute", response_model=ApiResponse)
async def chain_execute(
    req: ChainExecuteRequest,
    agent: ToolAgent = Depends(get_agent),
):
    """执行链式调用"""
    steps = []
    for s in req.steps:
        steps.append(ChainStep(
            tool_name=s.get("tool_name", ""),
            arguments=s.get("arguments", {}),
            argument_templates=s.get("argument_templates", {}),
            max_retries=s.get("max_retries", 2),
            timeout=s.get("timeout"),
        ))
    result = await agent.chain_executor.execute_chain(steps, role=req.role)
    return ApiResponse(data=result.to_dict())