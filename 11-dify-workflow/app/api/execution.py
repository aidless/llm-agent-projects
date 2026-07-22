"""执行 API"""

from __future__ import annotations

from typing import Optional
from fastapi import APIRouter, HTTPException

from app.models import (
    ExecutionRequest,
    ExecutionResponse,
    ListResponse,
)
from engine.dag import DAG
from engine.errors import WorkflowNotFoundError, DAGValidationError

router = APIRouter(prefix="/api/v1/executions", tags=["executions"])


def get_store():
    from app.main import workflow_store
    return workflow_store


def get_executor():
    from app.main import executor
    return executor


def get_exec_log():
    from app.main import execution_log
    return execution_log


@router.post("", response_model=ExecutionResponse)
def execute_workflow(req: ExecutionRequest):
    """执行工作流"""
    store = get_store()
    executor = get_executor()
    exec_log = get_exec_log()

    # 获取工作流
    try:
        wf = store.get(req.workflow_id)
    except WorkflowNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # 构建 DAG
    nodes_data = wf.get("nodes", {})
    edges_data = wf.get("edges", [])
    node_ids = list(nodes_data.keys())

    try:
        dag = DAG()
        for nid in node_ids:
            dag.add_node(nid)
        for edge in edges_data:
            src = edge.get("source", edge.get("from"))
            tgt = edge.get("target", edge.get("to"))
            dag.add_edge(src, tgt)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"DAG 构建失败: {e}")

    # 验证
    errors = dag.validate()
    if errors:
        raise HTTPException(
            status_code=400,
            detail="; ".join(str(e) for e in errors),
        )

    # 执行
    try:
        result = executor.execute(
            workflow_id=req.workflow_id,
            dag=dag,
            node_configs=nodes_data,
            global_inputs=req.inputs,
            env_vars=req.env_vars,
        )

        # 创建执行日志 (使用引擎的 execution_id)
        log = exec_log.create_log(
            workflow_id=req.workflow_id,
            execution_id=result.execution_id,
            status="running",
            global_inputs=req.inputs,
        )

        # 记录节点日志
        for node_id, nr in result.node_results.items():
            exec_log.add_node_log(
                execution_id=result.execution_id,
                node_id=node_id,
                status=nr.status.value,
                output=nr.output,
                error=nr.error,
                elapsed_ms=nr.elapsed_ms,
            )

        # 更新执行日志
        exec_log.update_log(
            execution_id=log["execution_id"],
            status=result.status.value,
            output=result.output,
            error=result.error,
            elapsed_ms=result.elapsed_ms,
        )

        return ExecutionResponse(
            execution_id=result.execution_id,
            workflow_id=result.workflow_id,
            status=result.status.value,
            node_results={
                nid: {
                    "node_id": nr.node_id,
                    "status": nr.status.value,
                    "output": nr.output,
                    "error": nr.error,
                    "elapsed_ms": nr.elapsed_ms,
                }
                for nid, nr in result.node_results.items()
            },
            output=result.output,
            error=result.error,
            elapsed_ms=result.elapsed_ms,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"执行失败: {e}")


@router.get("", response_model=ListResponse)
def list_executions(
    workflow_id: Optional[str] = None,
    status: Optional[str] = None,
    limit: int = 50,
):
    """列出执行日志"""
    exec_log = get_exec_log()
    logs = exec_log.list_logs(workflow_id=workflow_id, status=status, limit=limit)
    return ListResponse(items=logs, total=len(logs))


@router.get("/{execution_id}")
def get_execution(execution_id: str):
    """获取执行详情"""
    exec_log = get_exec_log()
    try:
        return exec_log.get_log(execution_id)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"执行记录 '{execution_id}' 不存在")
