"""采样策略 - 决定是否对 Trace 进行采样记录。"""

import random
from abc import ABC, abstractmethod
from typing import Optional


class Sampler(ABC):
    """采样器基类。"""

    @abstractmethod
    def should_sample(self, trace_id: str, name: str) -> bool:
        """根据 trace_id 和操作名决定是否采样。"""
        ...

    @abstractmethod
    def description(self) -> str:
        ...


class AlwaysOnSampler(Sampler):
    """全量采样 - 记录所有 Trace。"""

    def should_sample(self, trace_id: str, name: str) -> bool:  # noqa: ARG002
        return True

    def description(self) -> str:
        return "AlwaysOnSampler"


class AlwaysOffSampler(Sampler):
    """关闭采样 - 不记录任何 Trace。"""

    def should_sample(self, trace_id: str, name: str) -> bool:  # noqa: ARG002
        return False

    def description(self) -> str:
        return "AlwaysOffSampler"


class RatioSampler(Sampler):
    """概率采样 - 按固定概率采样。

    Args:
        ratio: 采样率，0.0 ~ 1.0
    """

    def __init__(self, ratio: float = 0.1) -> None:
        if not 0.0 <= ratio <= 1.0:
            raise ValueError(f"采样率必须在 [0, 1] 之间，当前值: {ratio}")
        self._ratio = ratio

    def should_sample(self, trace_id: str, name: str) -> bool:  # noqa: ARG002
        return random.random() < self._ratio

    def description(self) -> str:
        return f"RatioSampler(rate={self._ratio})"


class ParentBasedSampler(Sampler):
    """父 Span 采样策略 - 如果父 Span 已采样，则子 Span 也采样。
    回退到 delegate 采样器处理 root span。
    """

    def __init__(self, delegate: Optional[Sampler] = None) -> None:
        self._delegate = delegate or AlwaysOnSampler()

    def should_sample(self, trace_id: str, name: str, *, parent_sampled: bool = True) -> bool:
        if parent_sampled:
            return True
        return self._delegate.should_sample(trace_id, name)

    def description(self) -> str:
        return f"ParentBasedSampler(delegate={self._delegate.description()})"