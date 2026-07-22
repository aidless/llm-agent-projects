"""代码执行 API"""

from fastapi import APIRouter, Depends, HTTPException

from app.auth import verify_token
from app.models import ExecuteRequest, ExecuteResponse, ErrorResponse
from sandbox.executor import SandboxExecutor
from policy.engine import engine as policy_engine
from policy.models import SecurityLevel
from audit.logger import AuditLogger
from audit.event_tracker import EventTracker

router = APIRouter(prefix="/api/v1", tags=["execute"])

# 全局实例
audit_logger = AuditLogger()
event_tracker = EventTracker()
executor = SandboxExecutor(audit_logger=audit_logger)


@router.post(
    "/execute",
    response_model=ExecuteResponse,
    responses={400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}, 500: {"model": ErrorResponse}},
    summary="执行代码",
    description="在安全沙箱中执行 Python 代码（需要 Bearer Token 鉴权）",
    dependencies=[Depends(verify_token)],
)
async def execute_code(request: ExecuteRequest):
    """在安全沙箱中执行 Python 代码"""
    # 获取策略
    policy = policy_engine.get_policy(request.policy_name)
    if not policy:
        # 尝试按安全等级查找
        try:
            level = SecurityLevel(request.policy_name.upper())
            policy = policy_engine.get_policy_by_level(level)
        except (ValueError, TypeError):
            pass

    if not policy:
        raise HTTPException(
            status_code=400,
            detail=f"Policy '{request.policy_name}' not found. Available: {[p['name'] for p in policy_engine.list_policies()]}",
        )

    if not policy.enabled:
        raise HTTPException(status_code=400, detail=f"Policy '{request.policy_name}' is disabled")

    # 执行代码
    result = executor.execute(
        code=request.code,
        policy=policy,
        variables=request.variables,
    )

    return ExecuteResponse(**result.to_dict())


@router.get(
    "/execute/{execution_id}",
    response_model=ExecuteResponse,
    summary="获取执行结果",
    description="根据 execution_id 获取执行结果快照（需要 Bearer Token 鉴权）",
    dependencies=[Depends(verify_token)],
)
async def get_execution_result(execution_id: str):
    """获取指定执行的快照"""
    snapshot = audit_logger.get_snapshot(execution_id)
    if not snapshot:
        raise HTTPException(status_code=404, detail=f"Execution '{execution_id}' not found")
    return ExecuteResponse(**snapshot)