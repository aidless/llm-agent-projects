"""子工作流节点 - 调用其他工作流"""

from __future__ import annotations

from typing import Any, Dict, Optional

from engine.errors import NodeExecutionError

from .base import BaseNode


class SubWorkflowNode(BaseNode):
    """子工作流节点 - 嵌套调用其他工作流"""

    node_type = "sub_workflow"

    def __init__(self, workflow_resolver=None):
        """
        Args:
            workflow_resolver: 可选的工作流解析器
                签名: resolver(workflow_id) -> (dag, node_configs)
        """
        self._resolver = workflow_resolver

    def execute(
        self,
        inputs: Dict[str, Any],
        context: "ExecutionContext",
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        workflow_id = inputs.get("workflow_id", "")
        sub_inputs = inputs.get("inputs", {})

        if not workflow_id:
            raise NodeExecutionError("子工作流节点缺少 workflow_id 参数")

        if self._resolver is None:
            # Mock 模式: 返回模拟结果
            return {
                "workflow_id": workflow_id,
                "status": "mock_success",
                "output": {"result": f"[子工作流 Mock] {workflow_id} 执行完成"},
                "message": "工作流解析器未配置，返回模拟结果",
            }

        try:
            sub_dag, sub_node_configs = self._resolver(workflow_id)

            from engine.executor import WorkflowExecutor
            executor = WorkflowExecutor()
            # 注册与父级相同的处理器
            result = executor.execute(
                workflow_id=workflow_id,
                dag=sub_dag,
                node_configs=sub_node_configs,
                global_inputs=sub_inputs,
            )

            return {
                "workflow_id": workflow_id,
                "status": result.status.value,
                "output": result.output,
                "execution_id": result.execution_id,
            }
        except Exception as e:
            raise NodeExecutionError(
                f"子工作流 '{workflow_id}' 执行失败: {e}"
            )


def create_sub_workflow_handler(workflow_resolver=None):
    """工厂函数"""
    node = SubWorkflowNode(workflow_resolver=workflow_resolver)

    def handler(inputs, context, config):
        return node.execute(inputs, context, config)

    handler.node_type = SubWorkflowNode.node_type
    return handler
