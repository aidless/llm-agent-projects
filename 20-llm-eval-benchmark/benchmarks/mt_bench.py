"""
benchmarks/mt_bench.py - MT-Bench 多轮对话评测

支持多轮对话评分，涵盖 writing、roleplay、reasoning 等类别。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .base import BaseBenchmark, BenchmarkQuestion


class MTBenchBenchmark(BaseBenchmark):
    """MT-Bench 多轮对话评测基准."""

    name = "mt_bench"
    description = "Multi-Turn Benchmark for chat model evaluation"

    def __init__(self, data_path: Optional[str] = None):
        if data_path is None:
            data_path = f"{self.get_default_data_dir()}/mt_bench_sample.json"
        super().__init__(data_path)

    def load_data(self) -> List[BenchmarkQuestion]:
        """加载 MT-Bench 对话数据."""
        raw = self._load_json_data(self._data_path)
        self._raw_data = raw
        questions = []
        for item in raw.get("conversations", []):
            # 构建对话摘要作为问题
            turns = item.get("turns", [])
            user_turns = [t["content"] for t in turns if t["role"] == "user"]
            question_text = " | ".join(user_turns)

            questions.append(BenchmarkQuestion(
                id=item["id"],
                question=question_text,
                metadata={
                    "category": item.get("category", "unknown"),
                    "turns": turns,
                    "reference_answers": item.get("reference_answers", []),
                    "evaluation_dimensions": item.get(
                        "evaluation_dimensions", ["relevance", "coherence"]
                    ),
                },
            ))
        self._questions = questions
        return questions

    def build_prompt(self, question: BenchmarkQuestion) -> str:
        """构建 MT-Bench 对话提示."""
        turns = question.metadata.get("turns", [])
        if not turns:
            return question.question

        # 构建多轮对话历史
        conversation = []
        for turn in turns:
            role_label = "User" if turn["role"] == "user" else "Assistant"
            conversation.append(f"{role_label}: {turn['content']}")

        return "\n".join(conversation)

    def extract_answer(self, response: str, question: BenchmarkQuestion) -> Any:
        """MT-Bench 不提取具体答案，返回完整响应用于评分."""
        return response.strip()

    def check_answer(self, extracted: Any, question: BenchmarkQuestion) -> bool:
        """MT-Bench 不使用布尔判断，评分由外部评分器完成.

        此处返回 True 表示成功获得响应（非空即可）。
        """
        if isinstance(extracted, str):
            return len(extracted.strip()) > 0
        return False

    def get_categories(self) -> List[str]:
        """获取所有对话类别."""
        if not self._questions:
            self.load_data()
        categories = set()
        for q in self._questions:
            categories.add(q.metadata.get("category", "unknown"))
        return sorted(categories)

    def get_reference_answer(self, question: BenchmarkQuestion, turn_index: int = 0) -> str:
        """获取指定轮次的参考答案."""
        refs = question.metadata.get("reference_answers", [])
        if 0 <= turn_index < len(refs):
            return refs[turn_index]
        return ""

    def get_evaluation_dimensions(self, question: BenchmarkQuestion) -> List[str]:
        """获取评估维度."""
        return question.metadata.get("evaluation_dimensions", [])