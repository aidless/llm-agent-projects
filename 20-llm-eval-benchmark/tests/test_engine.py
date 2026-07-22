"""
tests/test_engine.py - 评测引擎测试

测试并发执行、进度追踪、超时、重试和缓存。
"""

import asyncio
import pytest

from benchmarks import MMLUBenchmark, GSM8KBenchmark
from benchmarks.base import BenchmarkResult
from engine import Evaluator, EvalConfig, EvalProgress, MockLLMClient


class TestMockLLMClient:
    """Mock LLM 客户端测试."""

    @pytest.mark.asyncio
    async def test_fixed_response(self):
        """测试固定响应."""
        client = MockLLMClient(fixed_response="Hello")
        result = await client.generate("Any prompt")
        assert result == "Hello"

    @pytest.mark.asyncio
    async def test_default_response_not_empty(self):
        """测试默认响应非空."""
        client = MockLLMClient()
        result = await client.generate("Some prompt")
        assert len(result) > 0

    @pytest.mark.asyncio
    async def test_set_fixed_response(self):
        """测试动态设置固定响应."""
        client = MockLLMClient()
        client.set_fixed_response("New response")
        result = await client.generate("Any prompt")
        assert result == "New response"


class TestEvalConfig:
    """评测配置测试."""

    def test_default_config(self):
        """测试默认配置."""
        config = EvalConfig()
        assert config.max_concurrent == 5
        assert config.timeout_seconds == 30.0
        assert config.max_retries == 2
        assert config.enable_cache is True

    def test_custom_config(self):
        """测试自定义配置."""
        config = EvalConfig(max_concurrent=10, timeout_seconds=60.0)
        assert config.max_concurrent == 10
        assert config.timeout_seconds == 60.0


class TestEvalProgress:
    """评测进度测试."""

    def test_progress_initial(self):
        """测试初始进度."""
        progress = EvalProgress()
        assert progress.total == 0
        assert progress.accuracy == 0.0
        assert progress.progress_pct == 0.0

    def test_progress_accuracy(self):
        """测试准确率计算."""
        progress = EvalProgress(total=10, completed=8, correct=6)
        assert progress.accuracy == 0.75
        assert progress.progress_pct == 80.0

    def test_progress_to_dict(self):
        """测试进度序列化."""
        progress = EvalProgress(total=100, completed=50, correct=25)
        d = progress.to_dict()
        assert d["accuracy"] == 0.5
        assert d["progress_pct"] == 50.0
        assert "total" in d


class TestEvaluator:
    """评测引擎测试."""

    @pytest.mark.asyncio
    async def test_evaluate_mmlu(self):
        """测试 MMLU 评测执行."""
        client = MockLLMClient(fixed_response="A")
        config = EvalConfig(
            max_concurrent=5,
            timeout_seconds=10.0,
            question_limit=5,
            enable_cache=False,
        )
        evaluator = Evaluator(llm_client=client, config=config)
        benchmark = MMLUBenchmark()

        results = await evaluator.evaluate(benchmark, "test-model")

        assert len(results) == 5
        assert all(isinstance(r, BenchmarkResult) for r in results)

    @pytest.mark.asyncio
    async def test_evaluate_with_progress(self):
        """测试带进度回调的评测."""
        client = MockLLMClient(fixed_response="A")
        config = EvalConfig(question_limit=3, enable_cache=False)
        evaluator = Evaluator(llm_client=client, config=config)
        benchmark = MMLUBenchmark()

        progress_updates = []

        def on_progress(p: EvalProgress):
            progress_updates.append(p.progress_pct)

        await evaluator.evaluate(benchmark, "test-model", on_progress=on_progress)

        assert len(progress_updates) > 0
        assert progress_updates[-1] == 100.0

    @pytest.mark.asyncio
    async def test_evaluate_results_sorted(self):
        """测试结果按题目 ID 排序."""
        client = MockLLMClient(fixed_response="A")
        config = EvalConfig(question_limit=10, enable_cache=False)
        evaluator = Evaluator(llm_client=client, config=config)
        benchmark = MMLUBenchmark()

        results = await evaluator.evaluate(benchmark, "test-model")
        ids = [r.question_id for r in results]
        assert ids == sorted(ids)

    @pytest.mark.asyncio
    async def test_evaluate_all_completed(self):
        """测试所有题目都完成."""
        client = MockLLMClient(fixed_response="A")
        config = EvalConfig(question_limit=10, enable_cache=False)
        evaluator = Evaluator(llm_client=client, config=config)
        benchmark = MMLUBenchmark()

        results = await evaluator.evaluate(benchmark, "test-model")
        assert evaluator.progress.completed == 10
        assert evaluator.progress.total == 10

    @pytest.mark.asyncio
    async def test_evaluate_gsm8k(self):
        """测试 GSM8K 评测执行."""
        client = MockLLMClient(fixed_response="The answer is 42.\n#### 42")
        config = EvalConfig(question_limit=5, enable_cache=False)
        evaluator = Evaluator(llm_client=client, config=config)
        benchmark = GSM8KBenchmark()

        results = await evaluator.evaluate(benchmark, "test-model")
        assert len(results) == 5