"""
请求队列和并发控制模块
使用 asyncio.Semaphore 实现并发控制
提供请求排队、限流、超时管理等功能
"""
import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class RequestPriority(int, Enum):
    """请求优先级"""
    LOW = 0
    NORMAL = 1
    HIGH = 2


@dataclass
class QueuedRequest:
    """排队中的请求"""
    request_id: str
    priority: RequestPriority
    create_time: float = field(default_factory=time.time)
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())


class RequestQueue:
    """
    异步请求队列
    使用 asyncio.Semaphore 控制最大并发数
    请求按 FIFO 顺序执行
    """

    def __init__(self, max_concurrent: int = 10, request_timeout: float = 60.0):
        """
        参数:
            max_concurrent: 最大并发请求数
            request_timeout: 单个请求超时时间（秒）
        """
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._max_concurrent = max_concurrent
        self._request_timeout = request_timeout
        self._active_count = 0           # 当前活跃请求数
        self._total_queued = 0           # 总排队请求数
        self._total_completed = 0        # 总完成请求数
        self._total_timeout = 0          # 总超时请求数
        self._total_rejected = 0         # 总拒绝请求数（队列满时）
        self._lock = asyncio.Lock()      # 用于计数器的异步锁

    @property
    def active_count(self) -> int:
        """当前活跃的请求数"""
        return self._active_count

    @property
    def available_slots(self) -> int:
        """当前可用的并发槽位数"""
        return self._max_concurrent - self._active_count

    async def acquire(self) -> bool:
        """
        获取一个执行槽位
        如果已满则等待，超时返回 False
        """
        try:
            await asyncio.wait_for(
                self._semaphore.acquire(),
                timeout=self._request_timeout,
            )
            async with self._lock:
                self._active_count += 1
                self._total_queued += 1
            return True
        except asyncio.TimeoutError:
            async with self._lock:
                self._total_timeout += 1
            logger.warning("请求等待队列超时，已拒绝")
            return False

    async def release(self):
        """释放一个执行槽位"""
        self._semaphore.release()
        async with self._lock:
            self._active_count = max(0, self._active_count - 1)
            self._total_completed += 1

    async def execute(
        self,
        func: Callable,
        *args,
        **kwargs,
    ) -> Any:
        """
        在并发控制下执行异步函数

        参数:
            func: 异步函数
            *args, **kwargs: 传递给函数的参数

        返回:
            函数的返回值

        异常:
            asyncio.TimeoutError: 等待超时
        """
        # 获取执行槽位
        acquired = await self.acquire()
        if not acquired:
            raise asyncio.TimeoutError("请求队列已满，等待超时")

        try:
            # 带超时执行
            result = await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=self._request_timeout,
            )
            return result
        finally:
            await self.release()

    def get_stats(self) -> dict:
        """获取队列统计信息"""
        return {
            "max_concurrent": self._max_concurrent,
            "active_count": self._active_count,
            "available_slots": self.available_slots,
            "total_queued": self._total_queued,
            "total_completed": self._total_completed,
            "total_timeout": self._total_timeout,
            "total_rejected": self._total_rejected,
        }


class FailoverManager:
    """
    失败自动降级管理器
    当主模型调用失败时，自动尝试备用模型
    支持配置最大降级次数
    """

    def __init__(self, max_fallbacks: int = 3):
        """
        参数:
            max_fallbacks: 最大降级尝试次数
        """
        self._max_fallbacks = max_fallbacks

    async def execute_with_fallback(
        self,
        call_func: Callable[[str], Any],
        primary_model: str,
        fallback_models: list[str],
    ) -> tuple[Any, str]:
        """
        带降级的执行

        参数:
            call_func: 模型调用函数，接受 model_key 参数
            primary_model: 主模型 key
            fallback_models: 备用模型 key 列表

        返回:
            (result, actual_model) - 结果和实际使用的模型
        """
        errors = []
        models_to_try = [primary_model] + fallback_models[:self._max_fallbacks]

        for i, model_key in enumerate(models_to_try):
            try:
                logger.info(
                    f"尝试调用模型: {model_key}"
                    f" (第 {i + 1}/{len(models_to_try)} 次)"
                )
                result = await call_func(model_key)
                if i > 0:
                    logger.info(f"降级成功: 使用 {model_key} 替代 {primary_model}")
                return result, model_key
            except Exception as e:
                error_msg = f"模型 {model_key} 调用失败: {str(e)[:200]}"
                logger.warning(error_msg)
                errors.append(error_msg)
                continue

        # 所有模型都失败了
        all_errors = "; ".join(errors)
        raise RuntimeError(
            f"所有模型均调用失败（共 {len(models_to_try)} 个）。"
            f"错误: {all_errors}"
        )


# 全局请求队列单例
_request_queue: Optional[RequestQueue] = None
_failover_manager: Optional[FailoverManager] = None


def get_request_queue(max_concurrent: int = 10, timeout: float = 60.0) -> RequestQueue:
    """获取全局请求队列实例"""
    global _request_queue
    if _request_queue is None:
        _request_queue = RequestQueue(max_concurrent=max_concurrent, request_timeout=timeout)
    return _request_queue


def get_failover_manager(max_fallbacks: int = 3) -> FailoverManager:
    """获取全局降级管理器实例"""
    global _failover_manager
    if _failover_manager is None:
        _failover_manager = FailoverManager(max_fallbacks=max_fallbacks)
    return _failover_manager


def reset_queue():
    """重置全局队列（主要用于测试）"""
    global _request_queue, _failover_manager
    _request_queue = None
    _failover_manager = None