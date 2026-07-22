"""Proxy 模块测试"""

import pytest
import time

from proxy.router import (
    RequestRouter, Backend, RouterConfig, RoutingStrategy,
)
from proxy.rate_limiter import (
    TokenBucketLimiter, SlidingWindowLimiter,
    LimiterConfig, LimiterType, create_limiter,
)
from proxy.circuit_breaker import (
    CircuitBreaker, CircuitBreakerConfig, CircuitState,
)


class TestRequestRouter:
    """测试请求路由器"""

    def test_create(self):
        router = RequestRouter()
        stats = router.get_stats()
        assert stats["total_backends"] == 0

    def test_add_remove_backend(self):
        router = RequestRouter()
        b = Backend(name="test", url="http://localhost:8000")
        router.add_backend(b)
        assert router.get_stats()["total_backends"] == 1
        router.remove_backend("test")
        assert router.get_stats()["total_backends"] == 0

    def test_round_robin(self):
        router = RequestRouter(RouterConfig(strategy=RoutingStrategy.ROUND_ROBIN))
        router.add_backend(Backend(name="b1", url="http://b1"))
        router.add_backend(Backend(name="b2", url="http://b2"))

        b1 = router.get_next_backend("req_1")
        b2 = router.get_next_backend("req_2")
        assert b1.name != b2.name

    def test_no_available_backend(self):
        router = RequestRouter()
        assert router.get_next_backend() is None

    def test_report_result(self):
        router = RequestRouter()
        router.add_backend(Backend(name="b1", url="http://b1"))
        backend = router.get_next_backend()
        assert backend is not None
        router.report_result("b1", 50.0, error=False)
        assert backend.avg_latency_ms > 0

    def test_health_check(self):
        router = RequestRouter()
        router.add_backend(Backend(name="b1", url="http://b1"))
        results = router.health_check()
        assert "b1" in results
        assert results["b1"] is True

    def test_get_backends_status(self):
        router = RequestRouter()
        router.add_backend(Backend(name="b1", url="http://b1"))
        status = router.get_backends_status()
        assert len(status) == 1
        assert status[0]["name"] == "b1"


class TestTokenBucketLimiter:
    """测试令牌桶限流器"""

    def test_create(self):
        limiter = TokenBucketLimiter()
        stats = limiter.get_stats()
        assert stats["available_tokens"] > 0

    def test_allow_within_limit(self):
        limiter = TokenBucketLimiter(LimiterConfig(burst_size=5, refill_rate=100))
        for _ in range(5):
            assert limiter.allow() is True
        # 第6个应该被限流
        assert limiter.allow() is False

    def test_refill(self):
        limiter = TokenBucketLimiter(LimiterConfig(burst_size=2, refill_rate=1000))
        limiter.allow()
        limiter.allow()
        assert limiter.allow() is False
        time.sleep(0.01)  # 等待填充
        assert limiter.allow() is True

    def test_per_key(self):
        limiter = TokenBucketLimiter(LimiterConfig(burst_size=2))
        limiter.allow("key_a")
        limiter.allow("key_a")
        assert limiter.allow("key_a") is False
        assert limiter.allow("key_b") is True  # 不同 key

    def test_reset(self):
        limiter = TokenBucketLimiter(LimiterConfig(burst_size=2))
        limiter.allow()
        limiter.allow()
        limiter.reset()
        assert limiter.allow() is True

    def test_stats(self):
        limiter = TokenBucketLimiter(LimiterConfig(burst_size=10))
        stats = limiter.get_stats()
        assert stats["type"] == "token_bucket"
        assert stats["burst_size"] == 10


class TestSlidingWindowLimiter:
    """测试滑动窗口限流器"""

    def test_create(self):
        limiter = SlidingWindowLimiter()
        stats = limiter.get_stats()
        assert stats["current_count"] == 0

    def test_allow_within_limit(self):
        limiter = SlidingWindowLimiter(
            LimiterConfig(max_requests=5, window_seconds=1.0)
        )
        for _ in range(5):
            assert limiter.allow() is True
        assert limiter.allow() is False

    def test_window_expiry(self):
        limiter = SlidingWindowLimiter(
            LimiterConfig(max_requests=2, window_seconds=0.05)
        )
        limiter.allow()
        limiter.allow()
        assert limiter.allow() is False
        time.sleep(0.06)  # 等窗口过期
        assert limiter.allow() is True

    def test_per_key(self):
        limiter = SlidingWindowLimiter(
            LimiterConfig(max_requests=2, window_seconds=1.0)
        )
        limiter.allow("key_a")
        limiter.allow("key_a")
        assert limiter.allow("key_a") is False
        assert limiter.allow("key_b") is True

    def test_stats(self):
        limiter = SlidingWindowLimiter(
            LimiterConfig(max_requests=10, window_seconds=1.0)
        )
        limiter.allow()
        stats = limiter.get_stats()
        assert stats["type"] == "sliding_window"
        assert stats["remaining"] == 9

    def test_reset(self):
        limiter = SlidingWindowLimiter(
            LimiterConfig(max_requests=2, window_seconds=1.0)
        )
        limiter.allow()
        limiter.reset()
        assert limiter.get_stats()["current_count"] == 0


class TestCreateLimiter:
    """测试工厂方法"""

    def test_create_token_bucket(self):
        limiter = create_limiter(LimiterType.TOKEN_BUCKET)
        assert isinstance(limiter, TokenBucketLimiter)

    def test_create_sliding_window(self):
        limiter = create_limiter(LimiterType.SLIDING_WINDOW)
        assert isinstance(limiter, SlidingWindowLimiter)

    def test_create_unknown(self):
        with pytest.raises(ValueError):
            create_limiter("unknown_type")


class TestCircuitBreaker:
    """测试熔断器"""

    def test_initial_state_closed(self):
        cb = CircuitBreaker("test", CircuitBreakerConfig())
        assert cb.state == CircuitState.CLOSED

    def test_allow_request_when_closed(self):
        cb = CircuitBreaker("test")
        assert cb.allow_request() is True

    def test_open_after_failures(self):
        config = CircuitBreakerConfig(failure_threshold=3)
        cb = CircuitBreaker("test", config)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.allow_request() is False

    def test_half_open_success(self):
        config = CircuitBreakerConfig(
            failure_threshold=2,
            success_threshold=2,
            half_open_max_calls=5,
        )
        cb = CircuitBreaker("test", config)
        # 触发熔断
        for _ in range(2):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # 手动设置 opened_at 使其可转半开
        cb._opened_at = time.time() - 100
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.allow_request() is True

        cb.record_success()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_half_open_failure_reopens(self):
        config = CircuitBreakerConfig(
            failure_threshold=2,
            recovery_timeout=100.0,
        )
        cb = CircuitBreaker("test", config)
        for _ in range(2):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN

        # 手动设置 opened_at 使其可转半开
        cb._opened_at = time.time() - 200
        assert cb.state == CircuitState.HALF_OPEN
        cb.allow_request()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_record_success_in_closed(self):
        cb = CircuitBreaker("test")
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_get_stats(self):
        cb = CircuitBreaker("test")
        stats = cb.get_stats()
        assert stats["name"] == "test"
        assert stats["state"] == "closed"
        assert "events" in stats

    def test_reset(self):
        config = CircuitBreakerConfig(failure_threshold=2)
        cb = CircuitBreaker("test", config)
        for _ in range(2):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.reset()
        assert cb.state == CircuitState.CLOSED
        assert cb.allow_request() is True

    def test_events_recorded(self):
        cb = CircuitBreaker("test", CircuitBreakerConfig(failure_threshold=1))
        cb.record_failure()
        stats = cb.get_stats()
        assert len(stats["events"]) >= 1
        assert stats["events"][0]["to"] == "open"
