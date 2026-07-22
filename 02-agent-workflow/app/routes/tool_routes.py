# ============================================
# 工具路由
# 提供直接调用工具的 API
# ============================================

from fastapi import APIRouter, HTTPException
from loguru import logger

from app.models import ToolCallRequest, ToolResponse
from tools import ALL_TOOLS

router = APIRouter()


@router.get("/tools", summary="获取可用工具列表")
async def list_tools():
    """
    列出所有可用的 Agent 工具及其描述
    """
    tools_info = []
    for tool in ALL_TOOLS:
        tools_info.append({
            "name": tool.name,
            "description": tool.description,
        })
    return {"tools": tools_info, "total": len(tools_info)}


@router.post("/tools/call", response_model=ToolResponse, summary="调用工具")
async def call_tool(request: ToolCallRequest):
    """
    直接调用指定工具

    Args:
        request: 工具调用请求，包含工具名称和参数
    """
    # 查找工具
    tool_map = {t.name: t for t in ALL_TOOLS}
    tool = tool_map.get(request.tool_name)

    if tool is None:
        available = list(tool_map.keys())
        raise HTTPException(
            status_code=404,
            detail=f"工具 '{request.tool_name}' 不存在。可用工具: {available}",
        )

    logger.info(f"[API] 调用工具: {request.tool_name}, 参数: {request.arguments}")

    try:
        result = await tool.ainvoke(request.arguments)
        return ToolResponse(
            tool_name=request.tool_name,
            result=str(result),
            success=True,
        )
    except Exception as e:
        logger.error(f"[API] 工具调用失败: {e}")
        return ToolResponse(
            tool_name=request.tool_name,
            result=f"工具调用失败: {str(e)}",
            success=False,
        )
