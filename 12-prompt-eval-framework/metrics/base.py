from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


class BaseMetric(ABC):
    """评估指标基类。"""

    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description

    @abstractmethod
    def compute(self, predictions: List[str], references: List[str], **kwargs) -> Dict[str, Any]:
        """计算评估指标。

        Args:
            predictions: 模型预测结果列表
            references: 参考答案列表
            **kwargs: 额外参数

        Returns:
            包含指标分数的字典
        """
        ...

    def compute_single(self, prediction: str, reference: str, **kwargs) -> Dict[str, Any]:
        """计算单条评估指标。"""
        return self.compute([prediction], [reference], **kwargs)
