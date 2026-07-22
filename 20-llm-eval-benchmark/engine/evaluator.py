"""
engine/evaluator.py - 评测引擎

支持并发评测执行、进度追踪、超时控制、重试机制和结果缓存。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import time
import logging
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from benchmarks.base import BaseBenchmark, BenchmarkQuestion, BenchmarkResult

logger = logging.getLogger(__name__)


@dataclass
class EvalProgress:
    """评测进度."""
    total: int = 0
    completed: int = 0
    correct: int = 0
    errors: int = 0
    current_question_id: Optional[int] = None
    start_time: float = 0.0
    elapsed_seconds: float = 0.0

    @property
    def accuracy(self) -> float:
        return self.correct / self.completed if self.completed > 0 else 0.0

    @property
    def progress_pct(self) -> float:
        return (self.completed / self.total * 100) if self.total > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "completed": self.completed,
            "correct": self.correct,
            "errors": self.errors,
            "accuracy": round(self.accuracy, 4),
            "progress_pct": round(self.progress_pct, 1),
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


@dataclass
class EvalConfig:
    """评测配置."""
    max_concurrent: int = 5
    timeout_seconds: float = 30.0
    max_retries: int = 2
    retry_delay: float = 1.0
    enable_cache: bool = True
    cache_dir: Optional[str] = None
    question_limit: Optional[int] = None


class LLMClient:
    """LLM API 调用客户端 (可被 mock 替换)."""

    async def generate(self, prompt: str, **kwargs) -> str:
        """调用 LLM 生成响应.

        默认实现返回模拟响应。子类或 mock 可覆盖此方法。
        """
        raise NotImplementedError("Use MockLLMClient for testing or implement a real client")


class MockLLMClient(LLMClient):
    """Mock LLM 客户端，用于测试.

    支持固定回答、按基准名称返回不同模式。
    """

    def __init__(
        self,
        fixed_response: Optional[str] = None,
        accuracy_rate: float = 0.8,
        response_delay: float = 0.01,
    ):
        self._fixed_response = fixed_response
        self._accuracy_rate = accuracy_rate
        self._response_delay = response_delay

    async def generate(self, prompt: str, **kwargs) -> str:
        """返回模拟响应."""
        await asyncio.sleep(self._response_delay)

        if self._fixed_response:
            return self._fixed_response

        # 模拟不同场景
        if "Answer:" in prompt and any(c in prompt for c in "ABCD"):
            # 多选题: 模拟正确答案
            import random
            if random.random() < self._accuracy_rate:
                # 尝试返回 "A" (MMLU 的正确答案模式)
                return "A"
            return "B"

        if "####" in prompt:
            # GSM8K: 模拟数值答案
            import random
            if random.random() < self._accuracy_rate:
                return "The answer is 42.\n#### 42"
            return "The answer is 99.\n#### 99"

        if "def " in prompt or "function" in prompt:
            # 代码生成: 模拟正确代码
            import re
            func_match = re.search(r"def\s+(\w+)\s*\(", prompt)
            if func_match and random.random() < self._accuracy_rate:
                func_name = func_match.group(1)
                return f"def {func_name}(*args, **kwargs):\n    pass"
            return "I cannot generate this code."

        # 默认返回
        return "This is a simulated response."

    def set_fixed_response(self, response: str) -> None:
        """设置固定响应."""
        self._fixed_response = response


class Evaluator:
    """评测引擎.

    协调基准加载、LLM 调用、答案提取和评分的完整流程。
    """

    def __init__(
        self,
        llm_client: LLMClient,
        config: Optional[EvalConfig] = None,
    ):
        self._llm_client = llm_client
        self._config = config or EvalConfig()
        self._progress = EvalProgress()
        self._results: List[BenchmarkResult] = []
        self._cache: Dict[str, Any] = {}

    @property
    def progress(self) -> EvalProgress:
        return self._progress

    @property
    def results(self) -> List[BenchmarkResult]:
        return list(self._results)

    def _get_cache_key(self, model_name: str, benchmark_name: str, question_id: int, prompt: str) -> str:
        """生成缓存键."""
        content = f"{model_name}:{benchmark_name}:{question_id}:{prompt}"
        return hashlib.md5(content.encode()).hexdigest()

    def _get_cache_dir(self) -> str:
        """获取缓存目录."""
        if self._config.cache_dir:
            return self._config.cache_dir
        cache_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".cache")
        os.makedirs(cache_dir, exist_ok=True)
        return cache_dir

    def _load_cache(self) -> None:
        """从磁盘加载缓存."""
        if not self._config.enable_cache:
            return
        cache_file = os.path.join(self._get_cache_dir(), "eval_cache.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "r") as f:
                    self._cache = json.load(f)
            except (json.JSONDecodeError, IOError):
                self._cache = {}

    def _save_cache(self) -> None:
        """保存缓存到磁盘."""
        if not self._config.enable_cache:
            return
        cache_file = os.path.join(self._get_cache_dir(), "eval_cache.json")
        try:
            with open(cache_file, "w") as f:
                json.dump(self._cache, f)
        except IOError:
            logger.warning("Failed to save evaluation cache")

    async def _evaluate_single(
        self,
        question: BenchmarkQuestion,
        benchmark: BaseBenchmark,
        model_name: str,
        semaphore: asyncio.Semaphore,
    ) -> BenchmarkResult:
        """评测单个题目."""
        prompt = benchmark.build_prompt(question)
        cache_key = self._get_cache_key(model_name, benchmark.name, question.id, prompt)

        # 检查缓存
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            return BenchmarkResult(
                question_id=question.id,
                question=question.question,
                expected=cached["expected"],
                predicted=cached["predicted"],
                correct=cached["correct"],
                score=cached.get("score", 1.0 if cached["correct"] else 0.0),
                metadata=cached.get("metadata", {}),
            )

        async with semaphore:
            last_error = None
            for attempt in range(self._config.max_retries + 1):
                try:
                    async with asyncio.timeout(self._config.timeout_seconds):
                        response = await self._llm_client.generate(prompt)

                    extracted = benchmark.extract_answer(response, question)
                    correct = benchmark.check_answer(extracted, question)
                    score = 1.0 if correct else 0.0

                    # 确定期望值
                    if "answer" in question.metadata:
                        expected = question.metadata["answer"]
                    elif "choices" in question.metadata and "answer" in question.metadata:
                        idx = question.metadata["answer"]
                        expected = question.metadata["choices"][idx]
                    else:
                        expected = None

                    result = BenchmarkResult(
                        question_id=question.id,
                        question=question.question,
                        expected=expected,
                        predicted=extracted,
                        correct=correct,
                        score=score,
                        metadata={
                            "model": model_name,
                            "benchmark": benchmark.name,
                            "attempt": attempt + 1,
                            "response_preview": response[:200] if response else "",
                        },
                    )

                    # 写入缓存
                    if self._config.enable_cache:
                        self._cache[cache_key] = {
                            "expected": result.expected,
                            "predicted": result.predicted,
                            "correct": result.correct,
                            "score": result.score,
                            "metadata": result.metadata,
                        }

                    return result

                except asyncio.TimeoutError:
                    last_error = f"Timeout after {self._config.timeout_seconds}s"
                except Exception as e:
                    last_error = str(e)

                if attempt < self._config.max_retries:
                    await asyncio.sleep(self._config.retry_delay * (attempt + 1))

        # 所有重试都失败
        return BenchmarkResult(
            question_id=question.id,
            question=question.question,
            expected=question.metadata.get("answer"),
            predicted=None,
            correct=False,
            score=0.0,
            error=last_error,
            metadata={"model": model_name, "benchmark": benchmark.name},
        )

    async def evaluate(
        self,
        benchmark: BaseBenchmark,
        model_name: str,
        on_progress: Optional[Callable[[EvalProgress], None]] = None,
    ) -> List[BenchmarkResult]:
        """执行完整评测.

        Args:
            benchmark: 评测基准实例
            model_name: 模型名称
            on_progress: 进度回调函数

        Returns:
            评测结果列表
        """
        self._load_cache()
        self._results = []
        self._progress = EvalProgress(
            total=len(benchmark.questions),
            start_time=time.time(),
        )

        questions = benchmark.questions
        if self._config.question_limit:
            questions = questions[: self._config.question_limit]
            self._progress.total = len(questions)

        semaphore = asyncio.Semaphore(self._config.max_concurrent)

        tasks = [
            self._evaluate_single(q, benchmark, model_name, semaphore)
            for q in questions
        ]

        for coro in asyncio.as_completed(tasks):
            result = await coro
            self._results.append(result)
            self._progress.completed += 1
            if result.correct:
                self._progress.correct += 1
            if result.error:
                self._progress.errors += 1
            self._progress.elapsed_seconds = time.time() - self._progress.start_time

            if on_progress:
                on_progress(self._progress)

        # 保存缓存
        self._save_cache()

        # 按题目 ID 排序
        self._results.sort(key=lambda r: r.question_id)

        return self._results

    def clear_cache(self) -> None:
        """清除缓存."""
        self._cache = {}
        cache_dir = self._get_cache_dir()
        cache_file = os.path.join(cache_dir, "eval_cache.json")
        if os.path.exists(cache_file):
            os.remove(cache_file)