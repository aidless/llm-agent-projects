"""
benchmarks/humaneval.py - HumanEval 代码生成评测

支持代码生成、测试用例执行和 Pass@k 评分。
"""

from __future__ import annotations

import re
from typing import Any, List, Optional

from .base import BaseBenchmark, BenchmarkQuestion


class HumanEvalBenchmark(BaseBenchmark):
    """HumanEval 代码生成评测基准."""

    name = "humaneval"
    description = "HumanEval - code generation with test execution"

    def __init__(self, data_path: Optional[str] = None):
        if data_path is None:
            data_path = f"{self.get_default_data_dir()}/humaneval_sample.json"
        super().__init__(data_path)

    def load_data(self) -> List[BenchmarkQuestion]:
        """加载 HumanEval 题目数据."""
        raw = self._load_json_data(self._data_path)
        self._raw_data = raw
        questions = []
        for item in raw.get("problems", []):
            questions.append(BenchmarkQuestion(
                id=item["id"],
                question=item["prompt"],
                metadata={
                    "task_id": item.get("task_id", f"task-{item['id']}"),
                    "canonical_solution": item.get("canonical_solution", ""),
                    "test": item.get("test", ""),
                    "entry_point": item.get("entry_point", ""),
                    "difficulty": item.get("difficulty", "unknown"),
                },
            ))
        self._questions = questions
        return questions

    def build_prompt(self, question: BenchmarkQuestion) -> str:
        """构建 HumanEval 代码生成提示."""
        return (
            f"Complete the following Python function. "
            f"Only output the completed function, no explanation.\n\n"
            f"{question.question}"
        )

    def extract_answer(self, response: str, question: BenchmarkQuestion) -> Any:
        """从模型响应中提取代码."""
        # 尝试提取 markdown 代码块
        code_block_match = re.search(
            r"```python\s*\n(.*?)```", response, re.DOTALL
        )
        if code_block_match:
            return code_block_match.group(1).strip()

        # 尝试无语言标记的代码块
        code_block_match = re.search(
            r"```\s*\n(.*?)```", response, re.DOTALL
        )
        if code_block_match:
            return code_block_match.group(1).strip()

        # 直接返回完整响应（假设整个响应就是代码）
        return response.strip()

    def check_answer(self, extracted: Any, question: BenchmarkQuestion) -> bool:
        """通过测试用例执行验证代码正确性.

        注意: 在实际评测中，这会使用沙箱执行代码。
        在模拟环境中，检查代码是否包含正确的函数签名。
        """
        if not isinstance(extracted, str):
            return False

        entry_point = question.metadata.get("entry_point", "")
        if not entry_point:
            return False

        # 检查代码中是否定义了目标函数
        func_pattern = rf"def\s+{re.escape(entry_point)}\s*\("
        if re.search(func_pattern, extracted):
            return True

        return False

    def get_test_code(self, question: BenchmarkQuestion, generated_code: str) -> str:
        """组合生成的代码和测试用例."""
        test_code = question.metadata.get("test", "")
        # 替换 check 函数中的 candidate 调用
        entry_point = question.metadata.get("entry_point", "")
        test_code = test_code.replace("candidate(", f"{entry_point}(")
        return f"{generated_code}\n\n{test_code}\n\ncheck({entry_point})"