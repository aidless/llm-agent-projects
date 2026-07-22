"""
benchmarks/custom.py - 自定义评测基准

支持通过 JSON 格式导入自定义评测题目。
"""

from __future__ import annotations

import re
from typing import Any, List, Optional

from .base import BaseBenchmark, BenchmarkQuestion


class CustomBenchmark(BaseBenchmark):
    """自定义评测基准.

    支持从 JSON 文件加载自定义题目，格式如下:

    {
        "name": "my_benchmark",
        "description": "My custom benchmark",
        "questions": [
            {
                "id": 1,
                "question": "...",
                "answer": "...",
                "type": "multiple_choice" | "free_form" | "exact_match",
                "choices": ["A", "B", "C", "D"]  // 仅 multiple_choice
            }
        ]
    }
    """

    name = "custom"
    description = "Custom benchmark loaded from JSON"

    def __init__(self, data_path: str, benchmark_name: Optional[str] = None):
        super().__init__(data_path)
        self._custom_name = benchmark_name

    def load_data(self) -> List[BenchmarkQuestion]:
        """加载自定义题目数据."""
        raw = self._load_json_data(self._data_path)
        self._raw_data = raw

        if self._custom_name:
            self.name = self._custom_name
        elif "name" in raw:
            self.name = raw["name"]
        if "description" in raw:
            self.description = raw["description"]

        questions = []
        for item in raw.get("questions", []):
            q_type = item.get("type", "exact_match")
            questions.append(BenchmarkQuestion(
                id=item["id"],
                question=item["question"],
                metadata={
                    "answer": item.get("answer", ""),
                    "type": q_type,
                    "choices": item.get("choices", []),
                    "category": item.get("category", "custom"),
                },
            ))
        self._questions = questions
        return questions

    def build_prompt(self, question: BenchmarkQuestion) -> str:
        """根据题目类型构建提示."""
        q_type = question.metadata.get("type", "exact_match")
        choices = question.metadata.get("choices", [])

        if q_type == "multiple_choice" and choices:
            labels = ["A", "B", "C", "D", "E", "F"]
            choice_lines = "\n".join(
                f"{labels[i]}. {c}" for i, c in enumerate(choices)
                if i < len(labels)
            )
            return (
                f"Answer the following question. "
                f"Respond with ONLY the letter of the correct answer.\n\n"
                f"Question: {question.question}\n\n"
                f"{choice_lines}\n\nAnswer:"
            )
        else:
            return f"{question.question}\n\nAnswer:"

    def extract_answer(self, response: str, question: BenchmarkQuestion) -> Any:
        """根据题目类型提取答案."""
        q_type = question.metadata.get("type", "exact_match")

        if q_type == "multiple_choice":
            response = response.strip().upper()
            if len(response) == 1 and response in "ABCDEF":
                return response
            match = re.search(r"\b([ABCDEF])\b", response)
            return match.group(1) if match else response.strip()
        else:
            return response.strip()

    def check_answer(self, extracted: Any, question: BenchmarkQuestion) -> bool:
        """根据题目类型检查答案."""
        q_type = question.metadata.get("type", "exact_match")
        expected = question.metadata.get("answer", "")

        if q_type == "exact_match":
            return str(extracted).strip().lower() == str(expected).strip().lower()
        elif q_type == "multiple_choice":
            if isinstance(extracted, str) and len(extracted) == 1:
                try:
                    expected_idx = int(expected)
                    labels = ["A", "B", "C", "D", "E", "F"]
                    return extracted == labels[expected_idx]
                except (ValueError, IndexError):
                    return extracted.upper() == str(expected).upper()
            return str(extracted).upper() == str(expected).upper()
        elif q_type == "contains":
            return str(expected).lower() in str(extracted).lower()

        return str(extracted).strip().lower() == str(expected).strip().lower()