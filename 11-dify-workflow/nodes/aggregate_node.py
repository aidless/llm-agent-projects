"""聚合节点 - 聚合多个输入"""

from __future__ import annotations

from typing import Any, Dict, List

from .base import BaseNode


class AggregateNode(BaseNode):
    """聚合多个输入为单一输出"""

    node_type = "aggregate"

    def execute(
        self,
        inputs: Dict[str, Any],
        context: "ExecutionContext",
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        strategy = inputs.get("strategy", "merge")  # merge, list, sum, concat, first, last
        values = inputs.get("values", [])
        key = inputs.get("key", "aggregated")

        if not isinstance(values, list):
            values = [values]

        result = self._aggregate(values, strategy)

        return {
            key: result,
            "strategy": strategy,
            "count": len(values),
            "source_count": len(values),
        }

    def _aggregate(self, values: list, strategy: str) -> Any:
        if strategy == "merge":
            # 字典合并
            merged = {}
            for v in values:
                if isinstance(v, dict):
                    merged.update(v)
                else:
                    merged[str(len(merged))] = v
            return merged

        elif strategy == "list":
            return values

        elif strategy == "sum":
            try:
                return sum(v for v in values if isinstance(v, (int, float)))
            except TypeError:
                return 0

        elif strategy == "concat":
            return "".join(str(v) for v in values)

        elif strategy == "first":
            return values[0] if values else None

        elif strategy == "last":
            return values[-1] if values else None

        elif strategy == "join":
            sep = str(values[1]) if len(values) > 1 else ", "
            items = values[0] if len(values) > 0 and isinstance(values[0], list) else values
            return sep.join(str(v) for v in items)

        else:
            return values


def create_aggregate_handler():
    """工厂函数"""
    node = AggregateNode()

    def handler(inputs, context, config):
        return node.execute(inputs, context, config)

    handler.node_type = AggregateNode.node_type
    return handler
