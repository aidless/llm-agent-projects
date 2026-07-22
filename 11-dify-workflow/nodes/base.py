"""节点基类 - 所有节点类型的抽象基类"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional


class BaseNode(ABC):
    """工作流节点抽象基类"""

    node_type: str = "base"

    @abstractmethod
    def execute(
        self,
        inputs: Dict[str, Any],
        context: "ExecutionContext",
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        执行节点逻辑。

        Args:
            inputs: 已解析的输入参数
            context: 执行上下文 (可读取其他节点输出)
            config: 节点配置 (包含 type, retry 等)

        Returns:
            节点输出字典
        """
        ...

    def validate_inputs(self, inputs: Dict[str, Any]) -> None:
        """验证输入参数，子类可重写"""
        pass
