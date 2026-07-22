"""
benchmarks/base.py - 评测基准抽象基类

定义所有评测基准的统一接口，包括数据加载、提示构建和答案提取。
"""

from __future__ import annotations

import json
import os
import abc
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class BenchmarkQuestion:
    """单个评测题目."""
    id: int
    question: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkResult:
    """单个题目的评测结果."""
    question_id: int
    question: str
    expected: Any
    predicted: Any
    correct: bool
    score: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class BaseBenchmark(abc.ABC):
    """评测基准抽象基类.

    所有具体的评测基准（MMLU、GSM8K、HumanEval、MT-Bench 等）都应
    继承此类并实现抽象方法。
    """

    # 子类应覆盖的类属性
    name: str = "base"
    description: str = "Base benchmark"

    def __init__(self, data_path: Optional[str] = None):
        self._data_path = data_path
        self._questions: List[BenchmarkQuestion] = []
        self._raw_data: Optional[Dict[str, Any]] = None

    @abc.abstractmethod
    def load_data(self) -> List[BenchmarkQuestion]:
        """从数据源加载题目."""
        ...

    @abc.abstractmethod
    def build_prompt(self, question: BenchmarkQuestion) -> str:
        """为给定题目构建 LLM 输入提示."""
        ...

    @abc.abstractmethod
    def extract_answer(self, response: str, question: BenchmarkQuestion) -> Any:
        """从 LLM 响应中提取答案."""
        ...

    @abc.abstractmethod
    def check_answer(self, extracted: Any, question: BenchmarkQuestion) -> bool:
        """检查提取的答案是否正确."""
        ...

    @property
    def questions(self) -> List[BenchmarkQuestion]:
        """获取已加载的题目列表，惰性加载."""
        if not self._questions:
            self._questions = self.load_data()
        return self._questions

    @property
    def total_questions(self) -> int:
        """获取题目总数."""
        return len(self.questions)

    def get_question_by_id(self, question_id: int) -> Optional[BenchmarkQuestion]:
        """根据 ID 获取题目."""
        for q in self.questions:
            if q.id == question_id:
                return q
        return None

    def _load_json_data(self, path: str) -> Dict[str, Any]:
        """从 JSON 文件加载数据."""
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def get_default_data_dir() -> str:
        """获取默认数据目录路径."""
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")