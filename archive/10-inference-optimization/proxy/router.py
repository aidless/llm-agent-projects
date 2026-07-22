"""请求路由器 - 负载均衡和请求路由"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Any


class RoutingStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    RANDOM = "random"
    HASH_BASED = "hash_based"
    WEIGHTED = "weighted"


@dataclass
class Backend:
    """后端服务"""
    name: str
    url: str
    weight: int = 1
    healthy: bool = True
    active_connections: int = 0
    total_requests: int = 0
    total_errors: int = 0
    avg_latency_ms: float = 0.0
    last_check_time: float = 0.0

    @property
    def is_available(self) -> bool:
        return self.healthy


@dataclass
class RouterConfig:
    """路由器配置"""
    strategy: RoutingStrategy = RoutingStrategy.ROUND_ROBIN
    health_check_interval: float = 30.0
    max_retries: int = 3
    timeout_seconds: float = 30.0
    retry_delay_ms: float = 100.0


class RequestRouter:
    """请求路由器 - 支持多种负载均衡策略"""

    def __init__(self, config: Optional[RouterConfig] = None):
        self.config = config or RouterConfig()
        self._backends: List[Backend] = []
        self._current_index = 0
        self._request_log: List[Dict] = []

    def add_backend(self, backend: Backend) -> None:
        """添加后端"""
        self._backends.append(backend)

    def remove_backend(self, name: str) -> bool:
        """移除后端"""
        before = len(self._backends)
        self._backends = [b for b in self._backends if b.name != name]
        return len(self._backends) < before

    def get_next_backend(
        self, request_id: Optional[str] = None
    ) -> Optional[Backend]:
        """获取下一个后端"""
        available = [b for b in self._backends if b.is_available]
        if not available:
            return None

        strategy = self.config.strategy

        if strategy == RoutingStrategy.ROUND_ROBIN:
            backend = available[self._current_index % len(available)]
            self._current_index += 1
        elif strategy == RoutingStrategy.LEAST_CONNECTIONS:
            backend = min(available, key=lambda b: b.active_connections)
        elif strategy == RoutingStrategy.RANDOM:
            import random
            backend = random.choice(available)
        elif strategy == RoutingStrategy.HASH_BASED and request_id:
            idx = int(hashlib.md5(request_id.encode()).hexdigest(), 16) % len(available)
            backend = available[idx]
        elif strategy == RoutingStrategy.WEIGHTED:
            backend = self._weighted_select(available)
        else:
            backend = available[0]

        backend.active_connections += 1
        backend.total_requests += 1
        return backend

    def _weighted_select(self, available: List[Backend]) -> Backend:
        """加权选择"""
        total_weight = sum(b.weight for b in available)
        if total_weight == 0:
            return available[0]

        import random
        r = random.random() * total_weight
        cumulative = 0
        for b in available:
            cumulative += b.weight
            if r <= cumulative:
                return b
        return available[-1]

    def release_connection(self, backend_name: str) -> None:
        """释放连接"""
        for b in self._backends:
            if b.name == backend_name:
                b.active_connections = max(0, b.active_connections - 1)
                break

    def report_result(
        self, backend_name: str, latency_ms: float, error: bool = False
    ) -> None:
        """报告请求结果"""
        for b in self._backends:
            if b.name == backend_name:
                # EMA (指数移动平均) 延迟
                alpha = 0.3
                b.avg_latency_ms = (
                    alpha * latency_ms + (1 - alpha) * b.avg_latency_ms
                )
                if error:
                    b.total_errors += 1
                b.last_check_time = time.time()
                break

        self._request_log.append({
            "backend": backend_name,
            "latency_ms": latency_ms,
            "error": error,
            "timestamp": time.time(),
        })
        # 保留最近 1000 条日志
        if len(self._request_log) > 1000:
            self._request_log = self._request_log[-1000:]

    def health_check(self) -> Dict[str, bool]:
        """健康检查"""
        results = {}
        for b in self._backends:
            # 简单模拟: 如果错误率过高则标记不健康
            if b.total_requests > 10:
                error_rate = b.total_errors / b.total_requests
                b.healthy = error_rate < 0.5
            results[b.name] = b.healthy
        return results

    def get_backends_status(self) -> List[Dict]:
        """获取所有后端状态"""
        return [
            {
                "name": b.name,
                "url": b.url,
                "healthy": b.healthy,
                "weight": b.weight,
                "active_connections": b.active_connections,
                "total_requests": b.total_requests,
                "total_errors": b.total_errors,
                "avg_latency_ms": round(b.avg_latency_ms, 2),
            }
            for b in self._backends
        ]

    def get_stats(self) -> Dict:
        """获取路由统计"""
        total = sum(b.total_requests for b in self._backends)
        errors = sum(b.total_errors for b in self._backends)
        return {
            "strategy": self.config.strategy.value,
            "total_backends": len(self._backends),
            "available_backends": sum(1 for b in self._backends if b.is_available),
            "total_requests": total,
            "total_errors": errors,
            "error_rate": errors / total if total > 0 else 0,
        }
