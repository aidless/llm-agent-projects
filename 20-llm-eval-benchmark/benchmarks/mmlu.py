"""
benchmarks/mmlu.py - MMLU (Massive Multitask Language Understanding) 评测

覆盖 STEM / Humanities / Social Sciences 等多选题评测。
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .base import BaseBenchmark, BenchmarkQuestion


class MMLUBenchmark(BaseBenchmark):
    """MMLU 多选题评测基准."""

    name = "mmlu"
    description = "Massive Multitask Language Understanding - multiple choice questions"

    def __init__(self, data_path: Optional[str] = None):
        if data_path is None:
            data_path = f"{self.get_default_data_dir()}/mmlu_sample.json"
        super().__init__(data_path)

    def load_data(self) -> List[BenchmarkQuestion]:
        """加载 MMLU 题目数据."""
        raw = self._load_json_data(self._data_path)
        self._raw_data = raw
        questions = []
        for item in raw.get("questions", []):
            questions.append(BenchmarkQuestion(
                id=item["id"],
                question=item["question"],
                metadata={
                    "choices": item["choices"],
                    "answer": item["answer"],
                    "category": item.get("category", "unknown"),
                    "subcategory": item.get("subcategory", "unknown"),
                },
            ))
        self._questions = questions
        return questions

    def build_prompt(self, question: BenchmarkQuestion) -> str:
        """构建 MMLU 多选题提示.

        格式: 给出问题 + 四个选项 (A/B/C/D)，要求模型只回答字母。
        """
        choices = question.metadata["choices"]
        labels = ["A", "B", "C", "D"]
        choice_lines = "\n".join(f"{label}. {text}" for label, text in zip(labels, choices))
        prompt = (
            f"Answer the following multiple choice question. "
            f"Respond with ONLY the letter of the correct answer (A, B, C, or D).\n\n"
            f"Question: {question.question}\n\n"
            f"{choice_lines}\n\n"
            f"Answer:"
        )
        return prompt

    def extract_answer(self, response: str, question: BenchmarkQuestion) -> Any:
        """从模型响应中提取选项字母."""
        response = response.strip().upper()

        # 直接匹配单个字母
        if len(response) == 1 and response in "ABCD":
            return response

        # 匹配 "The answer is X" 等模式
        patterns = [
            r"(?:answer|选择|答案)\s*(?:is|:|：|为)\s*([ABCD])",
            r"\b([ABCD])\b",
        ]
        for pattern in patterns:
            match = re.search(pattern, response)
            if match:
                return match.group(1)

        # 默认取第一个出现的 A/B/C/D
        for char in response:
            if char in "ABCD":
                return char

        return response

    def check_answer(self, extracted: Any, question: BenchmarkQuestion) -> bool:
        """检查提取的答案是否正确."""
        correct_idx = question.metadata["answer"]
        labels = ["A", "B", "C", "D"]
        correct_label = labels[correct_idx] if correct_idx < len(labels) else None

        if isinstance(extracted, str) and len(extracted) == 1:
            return extracted == correct_label

        # 如果提取的是索引
        try:
            return int(extracted) == correct_idx
        except (ValueError, TypeError):
            return False

    def get_categories(self) -> List[str]:
        """获取所有评测类别."""
        if not self._questions:
            self.load_data()
        categories = set()
        for q in self._questions:
            categories.add(q.metadata.get("category", "unknown"))
        return sorted(categories)

    def get_questions_by_category(self, category: str) -> List[BenchmarkQuestion]:
        """按类别获取题目."""
        return [q for q in self.questions if q.metadata.get("category") == category]