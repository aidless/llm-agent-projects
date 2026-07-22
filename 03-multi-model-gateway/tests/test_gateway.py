"""
测试多模型智能路由网关
使用 httpx.AsyncClient 的 ASGITransport 进行异步测试
不依赖真实的 LLM API，通过 mock 适配器进行测试
"""
import asyncio
import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

# 在导入 app 之前重置所有单例，确保测试隔离
from routers.manager import reset_adapter_manager
from strategy.router import reset_smart_router
from reqqueue.request_queue import reset_queue
from monitor.stats_collector import StatsCollector

reset_adapter_manager()
reset_smart_router()
reset_queue()

# 统计收集器在每次测试中独立创建，无需重置全局单例

from app.main import app
from app.models import ChatCompletionRequest, ChatMessage
from strategy.task_classifier import TaskClassifier, TaskType
from strategy.router import SmartRouter, TaskTypeRouter, CostOptimizeRouter
from reqqueue.request_queue import RequestQueue, FailoverManager


# ==================== 任务分类器测试 ====================


class TestTaskClassifier:
    """任务类型分类器测试"""

    @pytest.fixture
    def classifier(self):
        return TaskClassifier()

    def test_classify_code_generation(self, classifier):
        """测试代码生成任务分类"""
        assert classifier.classify("写一个 Python 快速排序函数") == TaskType.CODE_GENERATION
        assert classifier.classify("implement a binary search in Java") == TaskType.CODE_GENERATION
        assert classifier.classify("debug this error: TypeError") == TaskType.CODE_GENERATION
        assert classifier.classify("帮我写一段 React 组件代码") == TaskType.CODE_GENERATION

    def test_classify_creative_writing(self, classifier):
        """测试创意写作任务分类"""
        assert classifier.classify("写一首关于春天的诗") == TaskType.CREATIVE_WRITING
        assert classifier.classify("帮我写一个品牌宣传文案") == TaskType.CREATIVE_WRITING
        assert classifier.classify("write a short story about space") == TaskType.CREATIVE_WRITING

    def test_classify_data_analysis(self, classifier):
        """测试数据分析任务分类"""
        assert classifier.classify("分析这份数据的均值和方差") == TaskType.DATA_ANALYSIS
        assert classifier.classify("用 SQL 查询今天的销售额") == TaskType.DATA_ANALYSIS

    def test_classify_translation(self, classifier):
        """测试翻译任务分类"""
        assert classifier.classify("把这段话翻译成英文") == TaskType.TRANSLATION
        assert classifier.classify("translate to Chinese") == TaskType.TRANSLATION
        assert classifier.classify("中译英：今天天气很好") == TaskType.TRANSLATION

    def test_classify_summary(self, classifier):
        """测试摘要任务分类"""
        assert classifier.classify("总结一下这篇文章的要点") == TaskType.SUMMARY
        assert classifier.classify("summarize this document") == TaskType.SUMMARY

    def test_classify_general(self, classifier):
        """测试通用任务分类"""
        assert classifier.classify("你好") == TaskType.GENERAL
        assert classifier.classify("什么是机器学习？") == TaskType.GENERAL

    def test_user_hint_override(self, classifier):
        """测试用户提示覆盖自动分类"""
        assert classifier.classify("你好", user_hint="code") == TaskType.CODE_GENERATION
        assert classifier.classify("写代码", user_hint="creative") == TaskType.CREATIVE_WRITING

    def test_estimate_complexity(self, classifier):
        """测试复杂度估算"""
        # 短文本，低复杂度
        low = classifier.estimate_complexity("你好")
        assert low < 0.2

        # 长文本 + 代码块，高复杂度
        long_text = "```python\ndef hello():\n    print('hello')\n```\n" * 50
        high = classifier.estimate_complexity(long_text)
        assert high > 0.3


# ==================== 请求队列测试 ====================


class TestRequestQueue:
    """请求队列测试"""

    def test_queue_stats(self):
        """测试队列统计信息"""
        queue = RequestQueue(max_concurrent=5, request_timeout=10.0)
        stats = queue.get_stats()
        assert stats["max_concurrent"] == 5
        assert stats["active_count"] == 0
        assert stats["available_slots"] == 5

    @pytest.mark.asyncio
    async def test_concurrent_execution(self):
        """测试并发控制"""
        queue = RequestQueue(max_concurrent=2, request_timeout=5.0)

        execution_count = 0
        max_concurrent = 0
        lock = asyncio.Lock()

        async def track_concurrency():
            nonlocal execution_count, max_concurrent
            async with lock:
                execution_count += 1
                max_concurrent = max(max_concurrent, execution_count)
            await asyncio.sleep(0.1)
            async with lock:
                execution_count -= 1

        # 同时启动 4 个任务，但最大并发为 2
        tasks = [queue.execute(track_concurrency) for _ in range(4)]
        await asyncio.gather(*tasks)

        assert max_concurrent <= 2

    @pytest.mark.asyncio
    async def test_timeout(self):
        """测试超时处理"""
        queue = RequestQueue(max_concurrent=1, request_timeout=0.5)

        async def slow_task():
            await asyncio.sleep(10)  # 远超超时时间

        with pytest.raises(asyncio.TimeoutError):
            await queue.execute(slow_task)


class TestFailoverManager:
    """失败降级测试"""

    @pytest.mark.asyncio
    async def test_fallback_on_failure(self):
        """测试主模型失败时降级到备用模型"""
        failover = FailoverManager(max_fallbacks=3)
        call_log = []

        async def call_func(model_key: str) -> str:
            call_log.append(model_key)
            if model_key == "primary":
                raise RuntimeError("primary failed")
            return f"result from {model_key}"

        result, actual_model = await failover.execute_with_fallback(
            call_func=call_func,
            primary_model="primary",
            fallback_models=["fallback1", "fallback2"],
        )

        assert actual_model == "fallback1"
        assert result == "result from fallback1"
        assert call_log == ["primary", "fallback1"]

    @pytest.mark.asyncio
    async def test_all_fail(self):
        """测试所有模型都失败"""
        failover = FailoverManager(max_fallbacks=2)

        async def call_func(model_key: str) -> str:
            raise RuntimeError(f"{model_key} failed")

        with pytest.raises(RuntimeError, match="所有模型均调用失败"):
            await failover.execute_with_fallback(
                call_func=call_func,
                primary_model="m1",
                fallback_models=["m2", "m3"],
            )


# ==================== 路由策略测试 ====================


class TestRoutingStrategies:
    """路由策略测试"""

    def _make_request(self, content: str, strategy: str = None, model: str = None):
        """创建测试请求"""
        return ChatCompletionRequest(
            messages=[ChatMessage(role="user", content=content)],
            strategy=strategy,
            model=model,
        )

    def test_task_type_router_code(self):
        """测试任务类型路由 - 代码生成"""
        router = TaskTypeRouter()
        request = self._make_request("写一个 Python 排序算法")
        result = router.route(request)
        assert result.model_key == "deepseek"
        assert result.strategy == "task_type"
        assert result.task_type == "code_generation"

    def test_task_type_router_creative(self):
        """测试任务类型路由 - 创意写作"""
        router = TaskTypeRouter()
        request = self._make_request("写一首关于秋天的诗")
        result = router.route(request)
        assert result.model_key == "claude"
        assert result.task_type == "creative_writing"

    def test_task_type_router_general(self):
        """测试任务类型路由 - 通用"""
        router = TaskTypeRouter()
        request = self._make_request("你好，请问你能做什么？")
        result = router.route(request)
        assert result.model_key == "gpt-4o"

    def test_cost_router_simple(self):
        """测试成本路由 - 简单请求"""
        router = CostOptimizeRouter()
        request = self._make_request("你好")
        result = router.route(request)
        # 简单请求应该选择便宜的模型
        assert result.strategy == "cost"
        assert result.model_key in ("deepseek", "qwen")

    def test_direct_model_specified(self):
        """测试用户直接指定模型"""
        router = SmartRouter()
        request = self._make_request("写代码", model="deepseek")
        result = router.route(request)
        assert result.model_key == "deepseek"
        assert result.strategy == "direct"


# ==================== 监控统计测试 ====================


class TestStatsCollector:
    """统计收集器测试"""

    def test_record_and_retrieve(self):
        """测试记录和查询统计数据"""
        collector = StatsCollector()

        from monitor.stats_collector import RequestRecord
        collector.record_request(RequestRecord(
            request_id="test-001",
            model="gpt-4o",
            strategy="task_type",
            task_type="general",
            prompt_tokens=100,
            completion_tokens=50,
            total_tokens=150,
            latency_ms=1200.0,
            cost_usd=0.001,
            success=True,
        ))

        stats = collector.get_model_stats("gpt-4o")
        assert stats.total_requests == 1
        assert stats.success_requests == 1
        assert stats.total_tokens == 150
        assert stats.total_cost_usd == 0.001

    def test_recent_records(self):
        """测试最近记录查询"""
        collector = StatsCollector()

        from monitor.stats_collector import RequestRecord
        for i in range(5):
            collector.record_request(RequestRecord(
                request_id=f"test-{i:03d}",
                model="deepseek",
                strategy="cost",
                task_type="code_generation",
                prompt_tokens=10,
                completion_tokens=5,
                total_tokens=15,
                latency_ms=500.0 + i * 100,
                cost_usd=0.0001,
                success=True,
            ))

        recent = collector.get_recent_records(limit=3)
        assert len(recent) == 3
        # 最近 3 条应该是 test-002, test-003, test-004
        assert recent[-1]["request_id"] == "test-004"

    def test_reset(self):
        """测试重置统计"""
        collector = StatsCollector()

        from monitor.stats_collector import RequestRecord
        collector.record_request(RequestRecord(
            request_id="test-reset",
            model="qwen",
            strategy="latency",
            task_type="translation",
            prompt_tokens=20,
            completion_tokens=30,
            total_tokens=50,
            latency_ms=800.0,
            cost_usd=0.0005,
            success=True,
        ))

        collector.reset()
        stats = collector.get_all_stats()
        assert stats["total"]["total_requests"] == 0


# ==================== API 端点测试 ====================


@pytest_asyncio.fixture
async def client():
    """创建测试用的异步 HTTP 客户端"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


class TestAPIEndpoints:
    """API 端点测试"""

    @pytest.mark.asyncio
    async def test_health_check(self, client):
        """测试健康检查接口"""
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "available_models" in data
        assert "queue" in data

    @pytest.mark.asyncio
    async def test_list_models(self, client):
        """测试列出模型接口"""
        response = await client.get("/v1/models")
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert isinstance(data["data"], list)
        # 应该至少有 4 个模型
        assert len(data["data"]) >= 4

    @pytest.mark.asyncio
    async def test_list_strategies(self, client):
        """测试列出路由策略接口"""
        response = await client.get("/strategies")
        assert response.status_code == 200
        data = response.json()
        assert len(data["strategies"]) >= 4
        strategy_names = [s["name"] for s in data["strategies"]]
        assert "task_type" in strategy_names
        assert "cost" in strategy_names
        assert "latency" in strategy_names
        assert "load_balance" in strategy_names

    @pytest.mark.asyncio
    async def test_stats_endpoint(self, client):
        """测试统计接口"""
        response = await client.get("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "stats" in data
        assert "queue" in data

    @pytest.mark.asyncio
    async def test_stats_recent(self, client):
        """测试最近记录接口"""
        response = await client.get("/stats/recent?limit=5")
        assert response.status_code == 200
        data = response.json()
        assert "recent_requests" in data

    @pytest.mark.asyncio
    async def test_chat_completions_missing_messages(self, client):
        """测试缺少 messages 的请求"""
        response = await client.post("/v1/chat/completions", json={})
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_chat_completions_invalid_json(self, client):
        """测试无效 JSON"""
        response = await client.post(
            "/v1/chat/completions",
            content="not json",
            headers={"Content-Type": "application/json"},
        )
        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_chat_completions_no_api_key(self, client):
        """测试未配置 API Key 时的请求（会失败，但验证流程正确）"""
        response = await client.post("/v1/chat/completions", json={
            "messages": [{"role": "user", "content": "你好"}],
        })
        # 由于没有真实 API Key，应该返回 500 或 502
        assert response.status_code in (500, 502)

    @pytest.mark.asyncio
    async def test_reset_stats(self, client):
        """测试重置统计接口"""
        response = await client.delete("/stats")
        assert response.status_code == 200
        data = response.json()
        assert "已重置" in data["message"]