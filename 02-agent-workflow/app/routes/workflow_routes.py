# ============================================
# 工作流路由
# 提供工作流执行、状态查询等 API
# ============================================

from fastapi import APIRouter, HTTPException
from datetime import datetime
from loguru import logger

from app.models import (
    TaskRequest,
    TaskResponse,
    WorkflowInfoResponse,
    ChatRequest,
    ChatResponse,
    ErrorResponse,
)

router = APIRouter()

# 任务状态存储（生产环境应使用数据库）
_task_store: dict = {}


@router.get("/workflow/info", response_model=WorkflowInfoResponse, summary="获取工作流信息")
async def get_workflow_info():
    """
    获取当前工作流的详细信息，包括流程步骤和配置
    """
    from app.main import _workflow

    if _workflow is None:
        raise HTTPException(status_code=503, detail="工作流尚未初始化")

    info = _workflow.get_workflow_info()
    return WorkflowInfoResponse(**info)


@router.post("/workflow/run", response_model=TaskResponse, summary="执行研究工作流")
async def run_workflow(request: TaskRequest):
    """
    执行自动研究报告生成工作流

    完整流程：规划 → 搜索 → 分析 → 写作 → 审核 → 生成报告

    Args:
        request: 任务请求，包含任务描述和可选参数
    """
    from app.main import _workflow

    if _workflow is None:
        raise HTTPException(status_code=503, detail="工作流尚未初始化")

    logger.info(f"[API] 收到工作流执行请求: {request.task[:100]}...")

    try:
        # 运行工作流
        result = await _workflow.run(
            task=request.task,
            task_id=request.task_id,
        )

        # 提取关键信息构建响应
        task_id = result.get("task_id", "")
        status = result.get("status", "unknown")
        final_report = result.get("final_report", "")
        score = result.get("score")
        error = result.get("last_error")

        # 存储任务状态
        _task_store[task_id] = {
            "task_id": task_id,
            "status": status,
            "final_report": final_report,
            "score": score,
            "error": error,
            "created_at": result.get("created_at", datetime.now().isoformat()),
            "completed_at": datetime.now().isoformat(),
        }

        return TaskResponse(
            task_id=task_id,
            status=status,
            final_report=final_report,
            score=score,
            error=error,
            created_at=result.get("created_at", ""),
            completed_at=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"[API] 工作流执行失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"工作流执行失败: {str(e)}",
        )


@router.get("/workflow/tasks", summary="获取任务列表")
async def list_tasks():
    """
    获取所有已执行的任务列表
    """
    tasks = list(_task_store.values())
    return {"tasks": tasks, "total": len(tasks)}


@router.get("/workflow/tasks/{task_id}", response_model=TaskResponse, summary="查询任务状态")
async def get_task(task_id: str):
    """
    根据任务 ID 查询任务执行状态和结果
    """
    if task_id not in _task_store:
        raise HTTPException(status_code=404, detail=f"任务 '{task_id}' 不存在")

    task_data = _task_store[task_id]
    return TaskResponse(**task_data)


@router.post("/workflow/chat", response_model=ChatResponse, summary="Agent 对话")
async def chat_with_agent(request: ChatRequest):
    """
    与 Agent 进行简单对话（使用短期记忆）
    """
    from app.main import _workflow
    from agents.planner_agent import PlannerAgent

    if _workflow is None:
        raise HTTPException(status_code=503, detail="工作流尚未初始化")

    session_id = request.session_id or "default"
    short_memory = _workflow.short_term_memory

    # 添加用户消息
    short_memory.add_user_message(request.message)

    # 使用 Planner Agent 简单响应
    planner = PlannerAgent()
    history = short_memory.get_langchain_messages()
    try:
        from langchain_core.messages import SystemMessage
        messages = [SystemMessage(content="你是一个有帮助的 AI 助手。请简洁地回答用户问题。")]
        messages.extend(history[-10:])  # 最近的对话
        response = await planner.llm.ainvoke(messages)

        # 添加助手回复到记忆
        short_memory.add_assistant_message(response.content)

        return ChatResponse(
            message=response.content,
            session_id=session_id,
            turn_count=short_memory.get_turn_count(),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"对话失败: {str(e)}")
