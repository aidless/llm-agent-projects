"""
benchmarks/gsm8k.py - GSM8K (Grade School Math 8K) 评测

数学推理评测，支持步骤验证和数值答案提取。
"""

from __future__ import annotations

import re
from typing import Any, List, Optional

from .base import BaseBenchmark, BenchmarkQuestion


class GSM8KBenchmark(BaseBenchmark):
    """GSM8K 数学推理评测基准."""

    name = "gsm8k"
    description = "Grade School Math 8K - mathematical reasoning"

    def __init__(self, data_path: Optional[str] = None):
        if data_path is None:
            data_path = f"{self.get_default_data_dir()}/gsm8k_sample.json"
        super().__init__(data_path)

    def load_data(self) -> List[BenchmarkQuestion]:
        """加载 GSM8K 题目数据."""
        raw = self._load_json_data(self._data_path)
        self._raw_data = raw
        questions = []
        for item in raw.get("problems", []):
            questions.append(BenchmarkQuestion(
                id=item["id"],
                question=item["question"],
                metadata={
                    "answer": item["answer"],
                    "solution": item.get("solution", ""),
                    "difficulty": item.get("difficulty", "unknown"),
                },
            ))
        self._questions = questions
        return questions

    def build_prompt(self, question: BenchmarkQuestion) -> str:
        """构建 GSM8K 数学推理提示."""
        return (
            f"Solve the following math problem step by step. "
            f"Put your final numerical answer at the end after '#### '.\n\n"
            f"Problem: {question.question}\n\n"
            f"Solution:"
        )

    def extract_answer(self, response: str, question: BenchmarkQuestion) -> Any:
        """从模型响应中提取数值答案.

        支持 '#### 42' 格式或从最终行提取数字。
        """
        # 尝试匹配 #### 格式
        hash_match = re.search(r"####\s*([-+]?\d*\.?\d+)", response)
        if hash_match:
            try:
                return float(hash_match.group(1))
            except ValueError:
                pass

        # 提取所有数字，取最后一个（通常是最终答案）
        numbers = re.findall(r"[-+]?\d*\.?\d+", response)
        if numbers:
            try:
                val = float(numbers[-1])
                if val == int(val):
                    return int(val)
                return val
            except ValueError:
                pass

        return response.strip()

    def check_answer(self, extracted: Any, question: BenchmarkQuestion) -> bool:
        """检查数值答案是否正确（容差 1e-6）."""
        expected = question.metadata["answer"]

        try:
            extracted_num = float(extracted)
            expected_num = float(expected)
            return abs(extracted_num - expected_num) < 1e-6
        except (ValueError, TypeError):
            return str(extracted).strip() == str(expected).strip()

    def get_difficulty_distribution(self) -> dict:
        """获取难度分布."""
        distribution: dict = {}
        for q in self.questions:
            diff = q.metadata.get("difficulty", "unknown")
            distribution[diff] = distribution.get(diff, 0) + 1
        return distribution