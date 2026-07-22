"""
监控统计收集器 - 记录请求指标、Token 用量、成本、延迟等
使用内存存储，线程安全
"""
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RequestRecord:
    """单次请求记录"""
    request_id: str
    model: str                  # 实际使用的模型
    strategy: str               # 路由策略
    task_type: str              # 识别到的任务类型
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    cost_usd: float
    success: bool
    error_msg: str = ""
    timestamp: float = field(default_factory=time.time)


@dataclass
class ModelStats:
    """单个模型的统计信息"""
    total_requests: int = 0
    success_requests: int = 0
    failed_requests: int = 0
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    total_latency_ms: float = 0.0
    min_latency_ms: float = float("inf")
    max_latency_ms: float = 0.0
    # 最近请求的延迟，用于计算 P50/P99
    latencies: list[float] = field(default_factory=list)

    @property
    def avg_latency_ms(self) -> float:
        """平均延迟"""
        if self.success_requests == 0:
            return 0.0
        return self.total_latency_ms / self.success_requests

    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_requests == 0:
            return 0.0
        return self.success_requests / self.total_requests

    def p50_latency(self) -> float:
        """P50 延迟"""
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        idx = len(sorted_lat) // 2
        return sorted_lat[idx]

    def p99_latency(self) -> float:
        """P99 延迟"""
        if not self.latencies:
            return 0.0
        sorted_lat = sorted(self.latencies)
        idx = max(0, len(sorted_lat) - 1)
        return sorted_lat[idx]

    def to_dict(self) -> dict:
        return {
            "total_requests": self.total_requests,
            "success_requests": self.success_requests,
            "failed_requests": self.failed_requests,
            "success_rate": round(self.success_rate, 4),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "avg_latency_ms": round(self.avg_latency_ms, 2),
            "min_latency_ms": round(self.min_latency_ms, 2) if self.min_latency_ms != float("inf") else 0.0,
            "max_latency_ms": round(self.max_latency_ms, 2),
            "p50_latency_ms": round(self.p50_latency(), 2),
            "p99_latency_ms": round(self.p99_latency(), 2),
        }


class StatsCollector:
    """
    全局统计收集器（单例）
    记录每个模型的请求量、Token 用量、成本和延迟
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._model_stats: dict[str, ModelStats] = defaultdict(ModelStats)
        self._recent_records: list[RequestRecord] = []   # 保留最近 1000 条记录
        self._max_records = 1000
        self._start_time = time.time()

    def record_request(self, record: RequestRecord):
        """记录一次请求的结果"""
        with self._lock:
            stats = self._model_stats[record.model]
            stats.total_requests += 1
            if record.success:
                stats.success_requests += 1
                stats.total_prompt_tokens += record.prompt_tokens
                stats.total_completion_tokens += record.completion_tokens
                stats.total_tokens += record.total_tokens
                stats.total_cost_usd += record.cost_usd
                stats.total_latency_ms += record.latency_ms
                stats.min_latency_ms = min(stats.min_latency_ms, record.latency_ms)
                stats.max_latency_ms = max(stats.max_latency_ms, record.latency_ms)
                # 保留最近 200 条延迟用于 P50/P99 计算
                stats.latencies.append(record.latency_ms)
                if len(stats.latencies) > 200:
                    stats.latencies = stats.latencies[-200:]
            else:
                stats.failed_requests += 1

            # 保留最近记录
            self._recent_records.append(record)
            if len(self._recent_records) > self._max_records:
                self._recent_records = self._recent_records[-self._max_records:]

    def get_model_stats(self, model: str) -> ModelStats:
        """获取指定模型的统计数据"""
        with self._lock:
            return self._model_stats.get(model, ModelStats())

    def get_all_stats(self) -> dict:
        """获取所有模型的统计数据汇总"""
        with self._lock:
            total = ModelStats()
            models_dict = {}
            for model_name, stats in self._model_stats.items():
                models_dict[model_name] = stats.to_dict()
                total.total_requests += stats.total_requests
                total.success_requests += stats.success_requests
                total.failed_requests += stats.failed_requests
                total.total_prompt_tokens += stats.total_prompt_tokens
                total.total_completion_tokens += stats.total_completion_tokens
                total.total_tokens += stats.total_tokens
                total.total_cost_usd += stats.total_cost_usd
                total.total_latency_ms += stats.total_latency_ms

            return {
                "uptime_seconds": round(time.time() - self._start_time, 1),
                "total": total.to_dict(),
                "models": models_dict,
            }

    def get_recent_records(self, limit: int = 20) -> list[dict]:
        """获取最近的请求记录"""
        with self._lock:
            records = self._recent_records[-limit:]
            return [
                {
                    "request_id": r.request_id,
                    "model": r.model,
                    "strategy": r.strategy,
                    "task_type": r.task_type,
                    "tokens": r.total_tokens,
                    "latency_ms": round(r.latency_ms, 2),
                    "cost_usd": round(r.cost_usd, 6),
                    "success": r.success,
                    "error_msg": r.error_msg,
                    "timestamp": r.timestamp,
                }
                for r in records
            ]

    def reset(self):
        """重置所有统计数据"""
        with self._lock:
            self._model_stats.clear()
            self._recent_records.clear()
            self._start_time = time.time()


# 全局统计收集器单例
stats_collector = StatsCollector()