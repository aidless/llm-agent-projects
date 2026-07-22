"""定时器节点 - 延时执行"""

from __future__ import annotations

import time
from typing import Any, Dict

from .base import BaseNode


class TimerNode(BaseNode):
    """定时器/延时节点"""

    node_type = "timer"

    def execute(
        self,
        inputs: Dict[str, Any],
        context: "ExecutionContext",
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        delay_seconds = inputs.get("delay_seconds", 0)
        # 在测试环境中跳过实际等待
        skip_wait = config.get("skip_wait", False) or inputs.get("skip_wait", False)

        start = time.time()

        if not skip_wait and delay_seconds > 0:
            time.sleep(delay_seconds)

        elapsed = time.time() - start

        return {
            "delayed": delay_seconds,
            "actual_wait": elapsed,
            "skipped": skip_wait,
            "timestamp": time.time(),
        }


def create_timer_handler():
    """工厂函数"""
    node = TimerNode()

    def handler(inputs, context, config):
        return node.execute(inputs, context, config)

    handler.node_type = TimerNode.node_type
    return handler
