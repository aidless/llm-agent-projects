"""条件分支节点 - if-else / switch"""

from __future__ import annotations

import operator
from typing import Any, Dict, List, Optional

from engine.errors import NodeExecutionError

from .base import BaseNode

# 支持的比较运算符
_COMPARISON_OPS = {
    "eq": operator.eq,
    "ne": operator.ne,
    "gt": operator.gt,
    "gte": operator.ge,
    "lt": operator.lt,
    "lte": operator.le,
    "contains": lambda a, b: b in a if isinstance(a, (str, list)) else False,
    "not_contains": lambda a, b: b not in a if isinstance(a, (str, list)) else True,
    "is_empty": lambda a, b: not a,
    "is_not_empty": lambda a, b: bool(a),
    "starts_with": lambda a, b: str(a).startswith(str(b)) if a else False,
    "ends_with": lambda a, b: str(a).endswith(str(b)) if a else False,
    "regex_match": None,  # 特殊处理
}


class ConditionNode(BaseNode):
    """条件分支节点"""

    node_type = "condition"

    def execute(
        self,
        inputs: Dict[str, Any],
        context: "ExecutionContext",
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        conditions = inputs.get("conditions", [])
        # conditions: [{"left": ..., "op": "eq", "right": ..., "branch": "yes"}]
        # 也可以是简单的 if-else 模式

        if not conditions:
            raise NodeExecutionError("条件节点缺少 conditions 参数")

        matched_branch = None
        for cond in conditions:
            left = cond.get("left")
            op = cond.get("op", "eq")
            right = cond.get("right")
            branch = cond.get("branch", "default")

            if self._evaluate(left, op, right):
                matched_branch = branch
                break

        # 默认分支
        if matched_branch is None:
            matched_branch = inputs.get("default_branch", "default")

        return {
            "branch": matched_branch,
            "matched": matched_branch,
            "conditions_evaluated": len(conditions),
        }

    def _evaluate(self, left: Any, op: str, right: Any) -> bool:
        if op == "regex_match":
            import re
            if left is None:
                return False
            return bool(re.search(str(right), str(left)))

        fn = _COMPARISON_OPS.get(op)
        if fn is None:
            raise NodeExecutionError(f"不支持的条件运算符: {op}")

        try:
            return fn(left, right)
        except (TypeError, ValueError) as e:
            raise NodeExecutionError(f"条件评估失败 ({op}): {e}")


def create_condition_handler():
    """工厂函数"""
    node = ConditionNode()

    def handler(inputs, context, config):
        return node.execute(inputs, context, config)

    handler.node_type = ConditionNode.node_type
    return handler
