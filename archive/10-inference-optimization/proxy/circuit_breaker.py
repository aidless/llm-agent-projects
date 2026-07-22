"""熔断器 - 三种状态 (关闭/开启/半开)"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any, Callable


class CircuitState(Enum):
    CLOSED = "closed"      # 正常工作
    OPEN = "open"          # 熔断开启，拒绝请求
    HALF_OPEN = "half_open"  # 半开，允许少量探测请求


@dataclass
class CircuitBreakerConfig:
    """熔断器配置"""
    failure_threshold: int = 5  # 失败次数阈值
    recovery_timeout: float = 30.0  # 熔断恢复超时 (秒)
    half_open_max_calls: int = 3  # 半开状态最大探测请求数
    success_threshold: int = 2  # 半开状态下成功多少次恢复


@dataclass
class CircuitEvent:
    """熔断事件"""
    timestamp: float
    old_state: CircuitState
    new_state: CircuitState
    reason: str


class CircuitBreaker:
    """熔断器"""

    def __init__(
        self,
        name: str,
        config: Optional[CircuitBreakerConfig] = None,
    ):
        self.name = name
        self.config = config or CircuitBreakerConfig()

        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_successes = 0
        self._half_open_calls = 0
        self._last_failure_time: float = 0.0
        self._opened_at: float = 0.0

        self._events: List[CircuitEvent] = []

    @property
    def state(self) -> CircuitState:
        """获取当前状态 (考虑超时自动转半开)"""
        if self._state == CircuitState.OPEN:
            if time.time() - self._opened_at >= self.config.recovery_timeout:
                self._transition_to(CircuitState.HALF_OPEN, "recovery_timeout")
        return self._state

    def allow_request(self) -> bool:
        """判断是否允许请求通过"""
        current_state = self.state

        if current_state == CircuitState.CLOSED:
            return True
        elif current_state == CircuitState.OPEN:
            return False
        elif current_state == CircuitState.HALF_OPEN:
            if self._half_open_calls < self.config.half_open_max_calls:
                self._half_open_calls += 1
                return True
            return False
        return False

    def record_success(self) -> None:
        """记录成功"""
        current_state = self.state

        if current_state == CircuitState.HALF_OPEN:
            self._half_open_successes += 1
            if self._half_open_successes >= self.config.success_threshold:
                self._transition_to(CircuitState.CLOSED, "success_threshold_reached")
                self._failure_count = 0
                self._half_open_successes = 0
                self._half_open_calls = 0
        elif current_state == CircuitState.CLOSED:
            self._success_count += 1
            # 成功后衰减失败计数
            self._failure_count = max(0, self._failure_count - 1)

    def record_failure(self) -> None:
        """记录失败"""
        current_state = self.state

        if current_state == CircuitState.HALF_OPEN:
            self._transition_to(CircuitState.OPEN, "failure_in_half_open")
        elif current_state == CircuitState.CLOSED:
            self._failure_count += 1
            self._last_failure_time = time.time()
            if self._failure_count >= self.config.failure_threshold:
                self._transition_to(CircuitState.OPEN, "failure_threshold_reached")

    def _transition_to(self, new_state: CircuitState, reason: str) -> None:
        old_state = self._state
        self._state = new_state
        self._events.append(CircuitEvent(
            timestamp=time.time(),
            old_state=old_state,
            new_state=new_state,
            reason=reason,
        ))
        if new_state == CircuitState.OPEN:
            self._opened_at = time.time()
            self._half_open_successes = 0
            self._half_open_calls = 0
        elif new_state == CircuitState.HALF_OPEN:
            self._half_open_successes = 0
            self._half_open_calls = 0

    def get_stats(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "state": self.state.value,
            "failure_count": self._failure_count,
            "success_count": self._success_count,
            "failure_threshold": self.config.failure_threshold,
            "recovery_timeout": self.config.recovery_timeout,
            "half_open_successes": self._half_open_successes,
            "half_open_calls": self._half_open_calls,
            "events": [
                {
                    "timestamp": e.timestamp,
                    "from": e.old_state.value,
                    "to": e.new_state.value,
                    "reason": e.reason,
                }
                for e in self._events[-20:]
            ],
        }

    def reset(self) -> None:
        """重置熔断器"""
        old_state = self._state
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._success_count = 0
        self._half_open_successes = 0
        self._half_open_calls = 0
        self._opened_at = 0.0
        self._events.append(CircuitEvent(
            timestamp=time.time(),
            old_state=old_state,
            new_state=CircuitState.CLOSED,
            reason="manual_reset",
        ))
