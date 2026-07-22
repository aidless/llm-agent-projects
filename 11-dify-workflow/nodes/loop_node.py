"""循环节点 - for / while"""

from __future__ import annotations

from typing import Any, Dict, List

from engine.errors import NodeExecutionError

from .base import BaseNode


class LoopNode(BaseNode):
    """循环节点"""

    node_type = "loop"

    MAX_ITERATIONS = 1000  # 安全上限

    def execute(
        self,
        inputs: Dict[str, Any],
        context: "ExecutionContext",
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        loop_type = inputs.get("loop_type", "for")  # "for" or "while"
        max_iterations = min(inputs.get("max_iterations", self.MAX_ITERATIONS), self.MAX_ITERATIONS)

        results: List[Any] = []

        if loop_type == "for":
            items = inputs.get("items", [])
            if not isinstance(items, (list, tuple)):
                raise NodeExecutionError("for 循环节点: items 必须是列表或元组")
            item_var = inputs.get("item_var", "item")
            for i, item in enumerate(items[:max_iterations]):
                iteration_result = self._run_iteration(i, item, item_var, inputs, context)
                results.append(iteration_result)

        elif loop_type == "while":
            condition_expr = inputs.get("condition", "True")
            iteration = 0
            while iteration < max_iterations:
                # 在沙箱中评估条件
                try:
                    cond_result = self._eval_condition(condition_expr, iteration, results, context)
                except Exception as e:
                    raise NodeExecutionError(f"while 循环条件评估失败: {e}")
                if not cond_result:
                    break
                iteration_result = self._run_iteration(iteration, None, "item", inputs, context)
                results.append(iteration_result)
                iteration += 1
        else:
            raise NodeExecutionError(f"不支持的循环类型: {loop_type}")

        return {
            "results": results,
            "iterations": len(results),
            "loop_type": loop_type,
        }

    def _run_iteration(
        self,
        index: int,
        value: Any,
        item_var: str,
        inputs: Dict[str, Any],
        context: "ExecutionContext",
    ) -> Dict[str, Any]:
        """执行一次迭代"""
        iteration_body = inputs.get("body", {})
        # body 可以是一个简化的操作描述
        return {
            "index": index,
            item_var: value,
            "processed": True,
        }

    def _eval_condition(
        self,
        expr: str,
        iteration: int,
        results: list,
        context: "ExecutionContext",
    ) -> bool:
        """简单的条件表达式评估"""
        # 仅支持简单的 True/False 或比较
        safe_globals = {"__builtins__": {}}
        safe_builtins = {
            "True": True, "False": False, "None": None,
            "len": len, "int": int, "str": str,
            "abs": abs, "min": min, "max": max,
        }
        safe_globals["__builtins__"] = safe_builtins
        local_vars = {
            "iteration": iteration,
            "results": results,
            "result_count": len(results),
        }
        try:
            result = eval(expr, safe_globals, local_vars)
            return bool(result)
        except Exception:
            return False


def create_loop_handler():
    """工厂函数"""
    node = LoopNode()

    def handler(inputs, context, config):
        return node.execute(inputs, context, config)

    handler.node_type = LoopNode.node_type
    return handler
