"""限流器 - 令牌桶和滑动窗口算法"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any


class LimiterType(Enum):
    TOKEN_BUCKET = "token_bucket"
    SLIDING_WINDOW = "sliding_window"


@dataclass
class LimiterConfig:
    """限流配置"""
    max_requests: int = 100  # 最大请求数
    window_seconds: float = 1.0  # 时间窗口 (秒)
    burst_size: int = 10  # 突发大小 (令牌桶用)
    refill_rate: float = 50.0  # 令牌填充速率/秒 (令牌桶用)


class RateLimiter:
    """限流器基类"""

    def __init__(self, config: Optional[LimiterConfig] = None):
        self.config = config or LimiterConfig()

    def allow(self, key: str = "global") -> bool:
        raise NotImplementedError

    def get_stats(self, key: str = "global") -> Dict[str, Any]:
        raise NotImplementedError

    def reset(self, key: Optional[str] = None) -> None:
        raise NotImplementedError


class TokenBucketLimiter(RateLimiter):
    """令牌桶限流器"""

    def __init__(self, config: Optional[LimiterConfig] = None):
        super().__init__(config)
        # key -> {tokens, last_refill}
        self._buckets: Dict[str, Dict[str, float]] = {}

    def _ensure_bucket(self, key: str) -> Dict[str, float]:
        if key not in self._buckets:
            self._buckets[key] = {
                "tokens": float(self.config.burst_size),
                "last_refill": time.time(),
            }
        return self._buckets[key]

    def _refill(self, key: str) -> None:
        bucket = self._buckets.get(key)
        if bucket is None:
            return

        now = time.time()
        elapsed = now - bucket["last_refill"]
        bucket["tokens"] = min(
            self.config.burst_size,
            bucket["tokens"] + elapsed * self.config.refill_rate,
        )
        bucket["last_refill"] = now

    def allow(self, key: str = "global") -> bool:
        bucket = self._ensure_bucket(key)
        self._refill(key)

        if bucket["tokens"] >= 1.0:
            bucket["tokens"] -= 1.0
            return True
        return False

    def get_stats(self, key: str = "global") -> Dict[str, Any]:
        bucket = self._ensure_bucket(key)
        self._refill(key)
        return {
            "type": LimiterType.TOKEN_BUCKET.value,
            "available_tokens": bucket["tokens"],
            "burst_size": self.config.burst_size,
            "refill_rate": self.config.refill_rate,
        }

    def reset(self, key: Optional[str] = None) -> None:
        if key:
            self._buckets.pop(key, None)
        else:
            self._buckets.clear()


class SlidingWindowLimiter(RateLimiter):
    """滑动窗口限流器"""

    def __init__(self, config: Optional[LimiterConfig] = None):
        super().__init__(config)
        # key -> deque of timestamps
        self._windows: Dict[str, deque] = {}

    def _prune(self, key: str) -> None:
        """清理过期记录"""
        window = self._windows.get(key)
        if window is None:
            return

        cutoff = time.time() - self.config.window_seconds
        while window and window[0] <= cutoff:
            window.popleft()

    def allow(self, key: str = "global") -> bool:
        if key not in self._windows:
            self._windows[key] = deque()

        self._prune(key)
        window = self._windows[key]

        if len(window) < self.config.max_requests:
            window.append(time.time())
            return True
        return False

    def get_stats(self, key: str = "global") -> Dict[str, Any]:
        if key not in self._windows:
            self._windows[key] = deque()

        self._prune(key)
        window = self._windows[key]

        return {
            "type": LimiterType.SLIDING_WINDOW.value,
            "current_count": len(window),
            "max_requests": self.config.max_requests,
            "window_seconds": self.config.window_seconds,
            "remaining": self.config.max_requests - len(window),
        }

    def reset(self, key: Optional[str] = None) -> None:
        if key:
            self._windows.pop(key, None)
        else:
            self._windows.clear()


def create_limiter(
    limiter_type: LimiterType,
    config: Optional[LimiterConfig] = None,
) -> RateLimiter:
    """工厂方法创建限流器"""
    if limiter_type == LimiterType.TOKEN_BUCKET:
        return TokenBucketLimiter(config)
    elif limiter_type == LimiterType.SLIDING_WINDOW:
        return SlidingWindowLimiter(config)
    else:
        raise ValueError(f"Unknown limiter type: {limiter_type}")
