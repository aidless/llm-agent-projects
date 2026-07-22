"""节点模块 - 所有节点类型的注册和工厂"""

from nodes.base import BaseNode
from nodes.llm_node import LLMNode, create_llm_handler
from nodes.http_node import HTTPNode, create_http_handler
from nodes.code_node import CodeNode, create_code_handler
from nodes.condition_node import ConditionNode, create_condition_handler
from nodes.loop_node import LoopNode, create_loop_handler
from nodes.aggregate_node import AggregateNode, create_aggregate_handler
from nodes.timer_node import TimerNode, create_timer_handler
from nodes.sub_workflow_node import SubWorkflowNode, create_sub_workflow_handler

# 节点类型注册表
NODE_TYPES = {
    "llm": create_llm_handler,
    "http": create_http_handler,
    "code": create_code_handler,
    "condition": create_condition_handler,
    "loop": create_loop_handler,
    "aggregate": create_aggregate_handler,
    "timer": create_timer_handler,
    "sub_workflow": create_sub_workflow_handler,
}


def register_all_handlers(executor) -> None:
    """将所有内置节点处理器注册到执行引擎"""
    for node_type, factory in NODE_TYPES.items():
        handler = factory()
        executor.register_handler(node_type, handler)


__all__ = [
    "BaseNode",
    "LLMNode", "create_llm_handler",
    "HTTPNode", "create_http_handler",
    "CodeNode", "create_code_handler",
    "ConditionNode", "create_condition_handler",
    "LoopNode", "create_loop_handler",
    "AggregateNode", "create_aggregate_handler",
    "TimerNode", "create_timer_handler",
    "SubWorkflowNode", "create_sub_workflow_handler",
    "NODE_TYPES",
    "register_all_handlers",
]
