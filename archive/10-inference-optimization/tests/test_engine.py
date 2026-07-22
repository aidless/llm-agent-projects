"""Engine 模块测试"""

import asyncio
import time
import pytest
from unittest.mock import patch

from engine.tokenizer_mock import MockTokenizer, TokenizerConfig, EncodeResult
from engine.batcher import DynamicBatcher, InferenceRequest, BatchResult, BatchState
from engine.scheduler import RequestScheduler, SchedulingPolicy, QueueStats, SchedulerConfig
from engine.simulator import InferenceSimulator, SimulatorConfig, ModelStats


class TestMockTokenizer:
    """测试模拟分词器"""

    def test_default_config(self):
        tokenizer = MockTokenizer()
        assert tokenizer.vocab_size == 32000
        assert tokenizer.config.bos_token_id == 1
        assert tokenizer.config.eos_token_id == 2

    def test_custom_config(self):
        config = TokenizerConfig(vocab_size=1000, max_seq_length=512)
        tokenizer = MockTokenizer(config)
        assert tokenizer.vocab_size == 1000

    def test_encode_empty_string(self):
        tokenizer = MockTokenizer()
        result = tokenizer.encode("")
        assert result.num_tokens == 2  # BOS + EOS
        assert result.token_ids[0] == tokenizer.config.bos_token_id
        assert result.token_ids[-1] == tokenizer.config.eos_token_id

    def test_encode_text(self):
        tokenizer = MockTokenizer()
        result = tokenizer.encode("Hello world")
        assert isinstance(result, EncodeResult)
        assert result.num_tokens > 2  # BOS + tokens + EOS
        assert result.encode_latency_ms >= 0

    def test_decode(self):
        tokenizer = MockTokenizer()
        token_ids = [1, 100, 200, 2]  # BOS, some tokens, EOS
        text = tokenizer.decode(token_ids)
        assert "<s>" in text
        assert "</s>" in text

    def test_decode_empty(self):
        tokenizer = MockTokenizer()
        assert tokenizer.decode([]) == ""

    def test_decode_pad_skipped(self):
        tokenizer = MockTokenizer()
        text = tokenizer.decode([0, 1, 0, 2])  # PAD tokens should be skipped
        assert "0" not in text

    def test_truncation(self):
        config = TokenizerConfig(max_seq_length=5)
        tokenizer = MockTokenizer(config)
        result = tokenizer.encode("a" * 100)
        assert len(result.token_ids) <= 5


class TestDynamicBatcher:
    """测试动态批处理"""

    def test_create_batcher(self):
        batcher = DynamicBatcher(max_batch_size=8)
        assert batcher.max_batch_size == 8
        assert batcher.pending_count == 0

    @pytest.mark.asyncio
    async def test_submit_and_get_batch(self):
        batcher = DynamicBatcher(max_batch_size=4)
        req = InferenceRequest(
            request_id="test_1",
            prompt="Hello",
            max_tokens=10,
        )
        await batcher.submit(req)
        assert batcher.pending_count == 1

        batch = await batcher.get_next_batch()
        assert batch is not None
        assert len(batch) == 1
        assert batch[0].request_id == "test_1"

    @pytest.mark.asyncio
    async def test_batch_processing(self):
        batcher = DynamicBatcher(max_batch_size=2)
        requests = [
            InferenceRequest(request_id=f"req_{i}", prompt=f"Test {i}", max_tokens=5)
            for i in range(2)
        ]
        for req in requests:
            await batcher.submit(req)

        batch = await batcher.get_next_batch()
        assert batch is not None
        results = await batcher.process_batch(batch)
        assert len(results) == 2
        for r in results:
            assert isinstance(r, BatchResult)
            assert r.completion_tokens > 0
            assert r.ttft_ms > 0
            assert r.tokens_per_second > 0

    @pytest.mark.asyncio
    async def test_empty_queue(self):
        batcher = DynamicBatcher()
        batch = await batcher.get_next_batch()
        assert batch is None

    @pytest.mark.asyncio
    async def test_priority_sorting(self):
        batcher = DynamicBatcher()
        low = InferenceRequest(request_id="low", prompt="a", priority=1)
        high = InferenceRequest(request_id="high", prompt="b", priority=10)
        await batcher.submit(low)
        await batcher.submit(high)

        batch = await batcher.get_next_batch()
        # 两个都应该出来
        assert len(batch) == 2

    @pytest.mark.asyncio
    async def test_stats_updated(self):
        batcher = DynamicBatcher()
        req = InferenceRequest(request_id="s1", prompt="test", max_tokens=3)
        await batcher.submit(req)
        batch = await batcher.get_next_batch()
        await batcher.process_batch(batch)

        assert batcher.total_batches_processed == 1
        assert batcher.total_requests_processed == 1
        assert batcher.total_tokens_generated > 0


class TestRequestScheduler:
    """测试请求调度器"""

    @pytest.mark.asyncio
    async def test_enqueue_and_dequeue(self):
        batcher = DynamicBatcher(max_batch_size=4)
        scheduler = RequestScheduler(batcher)
        req = InferenceRequest(request_id="s1", prompt="test", max_tokens=5,
                                start_time=time.perf_counter())
        result = await scheduler.enqueue(req)
        assert result is True

        batch = await scheduler.dequeue_batch()
        assert batch is not None
        assert batch[0].request_id == "s1"

    @pytest.mark.asyncio
    async def test_queue_full_rejection(self):
        config = SchedulerConfig(max_queue_size=2)
        batcher = DynamicBatcher()
        scheduler = RequestScheduler(batcher, config)

        for i in range(3):
            req = InferenceRequest(request_id=f"req_{i}", prompt="test")
            await scheduler.enqueue(req)

        stats = scheduler.get_stats()
        assert stats.total_rejected == 1

    @pytest.mark.asyncio
    async def test_set_policy(self):
        batcher = DynamicBatcher()
        scheduler = RequestScheduler(batcher)
        scheduler.set_policy(SchedulingPolicy.SHORTEST_JOB_FIRST)
        assert scheduler.config.policy == SchedulingPolicy.SHORTEST_JOB_FIRST


class TestInferenceSimulator:
    """测试推理模拟器"""

    def test_create(self):
        sim = InferenceSimulator()
        assert sim.config.model_name == "mock-llm-7b"

    def test_model_stats(self):
        sim = InferenceSimulator()
        stats = sim.get_model_stats()
        assert isinstance(stats, ModelStats)
        assert stats.num_parameters > 0
        assert stats.memory_fp32_gb > stats.memory_fp16_gb
        assert stats.memory_fp16_gb > stats.memory_int8_gb
        assert stats.memory_int8_gb > stats.memory_int4_gb

    @pytest.mark.asyncio
    async def test_single_generate(self):
        sim = InferenceSimulator()
        result = await sim.generate("Hello world", max_tokens=5)
        assert result.request_id.startswith("req_")
        assert result.completion_tokens > 0
        assert result.ttft_ms > 0

    @pytest.mark.asyncio
    async def test_batch_generate(self):
        sim = InferenceSimulator()
        prompts = ["Hello", "World", "Test"]
        results = await sim.generate_batch(prompts, max_tokens=5)
        assert len(results) == 3

    def test_get_summary(self):
        sim = InferenceSimulator()
        summary = sim.get_summary()
        assert summary["model_name"] == "mock-llm-7b"
        assert "total_requests_served" in summary
