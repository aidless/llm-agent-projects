from typing import Any, Callable, Dict, List, Optional, Type

from .base import BaseMetric
from .bleu import BLEUMetric
from .rouge import ROUGEMetric
from .bertscore import BERTScoreMetric
from .exact_match import ExactMatchMetric
from .f1_score import F1ScoreMetric


class MetricRegistry:
    """指标注册表，管理所有可用的评估指标。"""

    def __init__(self):
        self._metrics: Dict[str, BaseMetric] = {}
        self._register_defaults()

    def _register_defaults(self):
        """注册默认指标。"""
        self.register(BLEUMetric())
        self.register(ROUGEMetric())
        self.register(BERTScoreMetric())
        self.register(ExactMatchMetric())
        self.register(F1ScoreMetric())

    def register(self, metric: BaseMetric):
        """注册一个评估指标。"""
        self._metrics[metric.name] = metric

    def register_custom(
        self,
        name: str,
        description: str,
        compute_fn: Callable[[List[str], List[str]], Dict[str, Any]],
    ):
        """注册自定义指标。

        Args:
            name: 指标名称
            description: 描述
            compute_fn: 计算函数，接收 (predictions, references) 返回分数字典
        """

        class CustomMetric(BaseMetric):
            def __init__(self_inner):
                super().__init__(name=name, description=description)
                self_inner._compute_fn = compute_fn

            def compute(self_inner, predictions: List[str], references: List[str], **kwargs) -> Dict[str, Any]:
                return self_inner._compute_fn(predictions, references)

        self.register(CustomMetric())

    def get(self, name: str) -> Optional[BaseMetric]:
        """获取指定名称的指标。"""
        return self._metrics.get(name)

    def list_metrics(self) -> List[Dict[str, str]]:
        """列出所有已注册的指标。"""
        return [
            {"name": m.name, "description": m.description}
            for m in self._metrics.values()
        ]

    def get_metrics(self, names: Optional[List[str]] = None) -> List[BaseMetric]:
        """获取指标列表，如果 names 为 None 则返回全部。"""
        if names is None:
            return list(self._metrics.values())
        result = []
        for name in names:
            metric = self.get(name)
            if metric is not None:
                result.append(metric)
            else:
                raise ValueError(f"未知指标: {name}")
        return result