"""策略管理 API"""

from typing import Set
from fastapi import APIRouter, HTTPException

from app.models import PolicyCreateRequest, PolicyInheritRequest, PolicyCombineRequest, ErrorResponse
from policy.engine import engine as policy_engine
from policy.models import (
    SecurityPolicy, SecurityLevel, ResourceLimits,
    ModulePolicy, FilesystemPolicy, NetworkPolicy,
)
from audit.logger import AuditLogger

router = APIRouter(prefix="/api/v1", tags=["policy"])

audit_logger = AuditLogger()


@router.get(
    "/policies",
    summary="列出所有策略",
    description="列出所有可用的安全策略",
)
async def list_policies():
    """列出所有安全策略"""
    return {"policies": policy_engine.list_policies()}


@router.get(
    "/policies/{name}",
    summary="获取策略详情",
    description="获取指定策略的完整配置",
)
async def get_policy(name: str):
    """获取策略详情"""
    policy = policy_engine.get_policy(name)
    if not policy:
        raise HTTPException(status_code=404, detail=f"Policy '{name}' not found")
    return policy.to_dict()


@router.post(
    "/policies",
    summary="创建自定义策略",
    description="创建新的安全策略",
    responses={400: {"model": ErrorResponse}},
)
async def create_policy(request: PolicyCreateRequest):
    """创建自定义策略"""
    if policy_engine.get_policy(request.name):
        raise HTTPException(status_code=400, detail=f"Policy '{request.name}' already exists")

    # 构建资源限制
    rl = ResourceLimits()
    if request.resource_limits:
        for k, v in request.resource_limits.items():
            if hasattr(rl, k):
                setattr(rl, k, v)

    # 构建模块策略
    mp = ModulePolicy()
    if request.module_policy:
        for k, v in request.module_policy.items():
            if k in ("allowed_modules", "blocked_modules", "allowed_builtins", "blocked_builtins"):
                if isinstance(v, (list, set)):
                    v = set(v)
                setattr(mp, k, v)
            elif hasattr(mp, k):
                setattr(mp, k, v)

    # 构建文件系统策略
    fp = FilesystemPolicy()
    if request.filesystem_policy:
        for k, v in request.filesystem_policy.items():
            if k in ("allowed_read_dirs", "allowed_write_dirs"):
                if isinstance(v, (list, set)):
                    v = set(v)
                setattr(fp, k, v)
            elif hasattr(fp, k):
                setattr(fp, k, v)

    # 构建网络策略
    np_ = NetworkPolicy()
    if request.network_policy:
        for k, v in request.network_policy.items():
            if k in ("allowed_domains", "blocked_domains", "allowed_ports", "blocked_ports"):
                if isinstance(v, (list, set)):
                    v = set(v)
                setattr(np_, k, v)
            elif hasattr(np_, k):
                setattr(np_, k, v)

    try:
        level = SecurityLevel(request.level)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid level: {request.level}. Use LOW/MEDIUM/HIGH/STRICT")

    policy = SecurityPolicy(
        name=request.name,
        level=level,
        description=request.description,
        resource_limits=rl,
        module_policy=mp,
        filesystem_policy=fp,
        network_policy=np_,
    )

    success = policy_engine.add_policy(policy)
    if not success:
        raise HTTPException(status_code=400, detail="Failed to create policy")

    audit_logger.log_policy_change(request.name, "CREATE", {"level": request.level})

    return {"message": f"Policy '{request.name}' created", "policy": policy.to_dict()}


@router.delete(
    "/policies/{name}",
    summary="删除策略",
    description="删除自定义策略（预设策略不可删除）",
)
async def delete_policy(name: str):
    """删除策略"""
    success = policy_engine.remove_policy(name)
    if not success:
        raise HTTPException(status_code=400, detail=f"Cannot delete policy '{name}' (not found or is preset)")
    audit_logger.log_policy_change(name, "DELETE")
    return {"message": f"Policy '{name}' deleted"}


@router.post(
    "/policies/inherit",
    summary="继承策略",
    description="基于已有策略创建继承策略",
    responses={400: {"model": ErrorResponse}},
)
async def inherit_policy(request: PolicyInheritRequest):
    """继承策略"""
    policy = policy_engine.inherit_policy(
        base_name=request.base_name,
        overrides=request.overrides,
        new_name=request.new_name,
        description=request.description,
    )
    if not policy:
        raise HTTPException(status_code=404, detail=f"Base policy '{request.base_name}' not found")

    audit_logger.log_policy_change(request.new_name, "INHERIT", {"base": request.base_name})

    return {"message": f"Policy '{request.new_name}' created from '{request.base_name}'", "policy": policy.to_dict()}


@router.post(
    "/policies/combine",
    summary="组合策略",
    description="组合多个策略（取最严格配置）",
    responses={400: {"model": ErrorResponse}},
)
async def combine_policies(request: PolicyCombineRequest):
    """组合策略"""
    policy = policy_engine.combine_policies(
        names=request.policy_names,
        combined_name=request.combined_name,
    )
    if not policy:
        raise HTTPException(status_code=400, detail="Failed to combine policies (none found)")

    audit_logger.log_policy_change(
        request.combined_name, "COMBINE", {"sources": request.policy_names}
    )

    return {"message": f"Combined policy '{request.combined_name}' created", "policy": policy.to_dict()}