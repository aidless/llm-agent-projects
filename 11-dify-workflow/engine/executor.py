"""工作流执行引擎 - 串行/并行执行、断点续执、错误重试"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from .context import ExecutionContext
from .dag import DAG
from .errors import (
    NodeExecutionError,
    WorkflowError,
    WorkflowExecutionError,
    WorkflowNotFoundError,
    NodeNotFoundError,
)

logger = logging.getLogger(__name__)


class ExecutionStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


@dataclass
class NodeResult:
    """单个节点的执行结果"""
    node_id: str
    status: ExecutionStatus = ExecutionStatus.PENDING
    output: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    elapsed_ms: float = 0.0

    @property
    def elapsed(self) -> float:
        return self.elapsed_ms


@dataclass
class WorkflowResult:
    """工作流整体执行结果"""
    execution_id: str
    workflow_id: str
    status: ExecutionStatus = ExecutionStatus.PENDING
    node_results: Dict[str, NodeResult] = field(default_factory=dict)
    output: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    @property
    def elapsed_ms(self) -> float:
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time) * 1000
        return 0.0


class WorkflowExecutor:
    """工作流执行引擎"""

    def __init__(self, max_workers: int = 4, default_retry: int = 0):
        self.max_workers = max_workers
        self.default_retry = default_retry
        self._executor = ThreadPoolExecutor(max_workers=max_workers)
        # node_id -> callable(node_config, context) -> Dict
        self._node_handlers: Dict[str, Any] = {}

    def register_handler(self, node_type: str, handler) -> None:
        """注册节点类型处理器"""
        self._node_handlers[node_type] = handler

    def execute(
        self,
        workflow_id: str,
        dag: DAG,
        node_configs: Dict[str, Dict[str, Any]],
        global_inputs: Dict[str, Any] = None,
        env_vars: Dict[str, str] = None,
    ) -> WorkflowResult:
        """
        同步执行工作流。

        Args:
            workflow_id: 工作流 ID
            dag: DAG 图
            node_configs: {node_id: {type, config, ...}} 节点配置
            global_inputs: 全局输入参数
            env_vars: 环境变量

        Returns:
            WorkflowResult
        """
        execution_id = str(uuid.uuid4())[:8]
        result = WorkflowResult(
            execution_id=execution_id,
            workflow_id=workflow_id,
        )
        context = ExecutionContext(env_vars=env_vars)

        # 验证 DAG
        errors = dag.validate()
        if errors:
            result.status = ExecutionStatus.FAILED
            result.error = "; ".join(str(e) for e in errors)
            return result

        # 初始化节点结果
        for node_id in dag.nodes:
            result.node_results[node_id] = NodeResult(node_id=node_id)

        # 设置全局输入为特殊节点 "__global__"
        if global_inputs:
            context.set_outputs("__global__", global_inputs)

        result.status = ExecutionStatus.RUNNING
        result.start_time = time.time()

        try:
            layers = dag.parallel_layers()
            for layer in layers:
                # 同层节点并行执行
                self._execute_layer(layer, node_configs, context, result)
                # 检查是否有节点失败
                if any(
                    nr.status == ExecutionStatus.FAILED
                    for nr in result.node_results.values()
                ):
                    result.status = ExecutionStatus.FAILED
                    break
            else:
                result.status = ExecutionStatus.SUCCESS

        except Exception as exc:
            result.status = ExecutionStatus.FAILED
            result.error = str(exc)
            logger.exception("工作流执行异常")

        result.end_time = time.time()

        # 收集叶子节点的输出作为最终输出
        for leaf in dag.leaves():
            nr = result.node_results.get(leaf)
            if nr and nr.status == ExecutionStatus.SUCCESS:
                result.output.update(nr.output)

        return result

    def _execute_layer(
        self,
        layer: List[str],
        node_configs: Dict[str, Dict[str, Any]],
        context: ExecutionContext,
        result: WorkflowResult,
    ) -> None:
        """并行执行同一层中的所有节点"""

        def _run_node(node_id: str):
            self._execute_node(node_id, node_configs, context, result)

        # 使用线程池并行执行
        futures = []
        for node_id in layer:
            f = self._executor.submit(_run_node, node_id)
            futures.append(f)

        for f in futures:
            f.result()  # 等待完成，异常会重新抛出

    def _execute_node(
        self,
        node_id: str,
        node_configs: Dict[str, Dict[str, Any]],
        context: ExecutionContext,
        result: WorkflowResult,
    ) -> None:
        """执行单个节点（含重试）"""
        if node_id not in node_configs:
            raise NodeNotFoundError(f"节点 '{node_id}' 配置不存在")

        config = node_configs[node_id]
        node_type = config.get("type", "")
        nr = result.node_results[node_id]

        # 解析输入中的模板变量
        resolved_inputs = context.resolve_dict(config.get("inputs", {}))
        context.set_input(node_id, resolved_inputs)

        max_retries = config.get("retry", self.default_retry)
        last_error = None

        for attempt in range(max_retries + 1):
            try:
                nr.status = ExecutionStatus.RUNNING if attempt == 0 else ExecutionStatus.RETRYING
                nr.start_time = time.time()

                handler = self._node_handlers.get(node_type)
                if handler is None:
                    raise NodeExecutionError(
                        f"未注册的节点类型 '{node_type}'",
                        node_id=node_id,
                    )

                output = handler(resolved_inputs, context, config)
                nr.output = output if isinstance(output, dict) else {"result": output}
                context.set_outputs(node_id, nr.output)

                nr.status = ExecutionStatus.SUCCESS
                nr.end_time = time.time()
                nr.elapsed_ms = (nr.end_time - nr.start_time) * 1000
                return

            except Exception as exc:
                last_error = exc
                nr.end_time = time.time()
                nr.elapsed_ms = (nr.end_time - nr.start_time) * 1000

        # 重试耗尽
        nr.status = ExecutionStatus.FAILED
        nr.error = str(last_error)
        if isinstance(last_error, WorkflowError):
            raise last_error
        raise NodeExecutionError(str(last_error), node_id=node_id)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
