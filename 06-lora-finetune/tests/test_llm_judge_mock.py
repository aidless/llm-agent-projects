#!/usr/bin/env python3
"""
LLM-judge 路径的 mock 测试（不需要真 OpenAI）。

测试：
  - judge_client 返回 None → 跳过
  - judge_client 返回预设 JSON → 解析正确
  - judge_client 返回 markdown 包裹 JSON → 容错解析
  - judge_client 抛错 → 错误信息捕获
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

# 只导入需要的部分（避免 transformers 依赖）
from eval_base_vs_lora import call_llm_judge, get_judge_client, JUDGE_SYSTEM


class MockJudgeClient:
    """Mock OpenAI 客户端，可预设返回。"""

    def __init__(self, response_content: str = '', raise_exc=None):
        self.response_content = response_content
        self.raise_exc = raise_exc
        self.calls = 0

    class _Msg:
        def __init__(self, content):
            self.content = content

    class _Choice:
        def __init__(self, msg):
            self.message = msg

    class _Resp:
        def __init__(self, content):
            self.choices = [MockJudgeClient._Choice(MockJudgeClient._Msg(content))]

    class chat:
        class completions:
            @staticmethod
            def create(**kwargs):
                raise NotImplementedError


class MockJudgeClientFunc(MockJudgeClient):
    """改用 __getattr__ 让 chat.completions.create 可调用。"""

    def __init__(self, response_content: str = '', raise_exc=None):
        super().__init__(response_content, raise_exc)
        self.kwargs_received = None

    def create(self, **kwargs):
        self.calls += 1
        self.kwargs_received = kwargs
        if self.raise_exc:
            raise self.raise_exc
        return MockJudgeClient._Resp(self.response_content)


def _judge_client_with_func(parent: MockJudgeClient):
    """Wrap mock to support client.chat.completions.create()."""

    class _Chat:
        class completions:
            create = parent.create

    class _Client:
        chat = _Chat

    return _Client()


# 由于 OpenAI API 形态较复杂，我们改为直接 patch
def test_call_llm_judge_returns_none_for_none_client():
    """无 client → 返回 None。"""
    result = call_llm_judge(None, "gpt-4", "Q", "G", "A")
    assert result is None


def test_call_llm_judge_parses_clean_json():
    client = MockJudgeClientFunc(response_content='{"accuracy": 5, "relevance": 4, "completeness": 3, "conciseness": 4}')
    result = call_llm_judge(_judge_client_with_func(client), "gpt-4o-mini",
                            "What is X?", "X is foo.", "X is bar.")
    assert result is not None
    assert result["accuracy"] == 5
    assert result["relevance"] == 4


def test_call_llm_judge_handles_markdown_fence():
    """容忍 markdown ```json 包裹。"""
    client = MockJudgeClientFunc(
        response_content='```json\n{"accuracy": 4, "relevance": 5, "completeness": 4, "conciseness": 3}\n```'
    )
    result = call_llm_judge(_judge_client_with_func(client), "gpt-4o-mini",
                            "Q", "G", "A")
    assert result["accuracy"] == 4
    assert result["relevance"] == 5


def test_call_llm_judge_handles_missing_fields():
    """部分字段缺失时返回默认值 3。"""
    client = MockJudgeClientFunc(response_content='{"accuracy": 5}')
    result = call_llm_judge(_judge_client_with_func(client), "gpt-4o-mini",
                            "Q", "G", "A")
    assert result["accuracy"] == 5
    assert result["relevance"] == 3  # default
    assert result["conciseness"] == 3


def test_call_llm_judge_captures_errors():
    """API 调用失败时，错误信息存入 error 字段。"""
    client = MockJudgeClientFunc(raise_exc=RuntimeError("rate limit"))
    result = call_llm_judge(_judge_client_with_func(client), "gpt-4o-mini",
                            "Q", "G", "A")
    assert "error" in result
    assert "RuntimeError" in result["error"]
    assert result["accuracy"] == 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
