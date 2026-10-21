#!/usr/bin/env python3
"""
CPU-only 单元测试：审计脚本 + 评分函数 + 数据加载路径。

依赖：仅 Python 标准库（pytest + 本项目脚本）。
不需要 GPU，不需要 transformers，不需要联网。

运行：
  pytest tests/test_audit_split.py -v

退出码：
  0 = 所有测试通过
  1 = 有失败
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

# 把脚本和 src 加入 path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

# 导入待测模块
from audit_dataset import (
    read_jsonl, write_jsonl, audit, render_markdown, split_indices,
    extract_code_blocks, try_parse_code, normalize_for_dup, ngrams,
    CODE_FENCE,
)


# ----------------------------- 测试：JSONL 读写 -----------------------------

class TestJsonlIO:
    def test_read_dict_format(self, tmp_path: Path):
        p = tmp_path / "in.jsonl"
        p.write_text(
            json.dumps({"instruction": "q1", "output": "a1"}, ensure_ascii=False) + "\n"
            + json.dumps({"instruction": "q2", "output": "a2"}, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        rows = read_jsonl(p)
        assert len(rows) == 2
        assert rows[0]["instruction"] == "q1"

    def test_read_list_format_compat(self, tmp_path: Path):
        """兼容 [q, a] 列表写法"""
        p = tmp_path / "in.jsonl"
        p.write_text(
            json.dumps(["q1", "a1"], ensure_ascii=False) + "\n"
            + json.dumps(["q2", "a2"], ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        rows = read_jsonl(p)
        assert len(rows) == 2
        assert rows[0] == {"instruction": "q1", "output": "a1"}

    def test_read_skip_blank(self, tmp_path: Path):
        p = tmp_path / "in.jsonl"
        p.write_text(
            "\n"
            + json.dumps({"instruction": "q", "output": "a"}, ensure_ascii=False) + "\n"
            + "\n",
            encoding="utf-8",
        )
        rows = read_jsonl(p)
        assert len(rows) == 1

    def test_read_invalid_json_quarantines(self, tmp_path: Path):
        """当前实现对单行 JSON 解析失败是 soft-fail（计入 _bad_json），不让审计崩溃。"""
        p = tmp_path / "in.jsonl"
        p.write_text("not json\n", encoding="utf-8")
        rows = read_jsonl(p)
        assert len(rows) == 1
        assert rows[0].get("_bad_json") is True

    def test_write_roundtrip(self, tmp_path: Path):
        p = tmp_path / "out.jsonl"
        rows = [{"a": 1, "b": "中"}, {"a": 2, "b": "文"}]
        write_jsonl(p, rows)
        loaded = read_jsonl(p)
        assert loaded == rows


# ----------------------------- 测试：代码块抽取 -----------------------------

class TestCodeBlocks:
    def test_python_block(self):
        text = "前置文本\n```python\nprint('hi')\n```\n后置文本"
        blocks = extract_code_blocks(text)
        assert len(blocks) == 1
        lang, code = blocks[0]
        assert lang == "python"
        assert "print('hi')" in code

    def test_no_block(self):
        assert extract_code_blocks("plain text") == []

    def test_multiple_blocks(self):
        text = "```python\nx=1\n```\n中间\n```python\ny=2\n```"
        blocks = extract_code_blocks(text)
        assert len(blocks) == 2


class TestCodeParse:
    def test_valid_python_parses(self):
        assert try_parse_code("python", "x = 1\nprint(x)") is None

    def test_invalid_python_returns_err(self):
        assert try_parse_code("python", "x = \n") is not None

    def test_other_lang_skipped(self):
        # 不支持的 lang 返回 None（跳过）
        assert try_parse_code("ruby", "x =") is None


# ----------------------------- 测试：归一化与 n-gram -----------------------------

class TestNormalize:
    def test_removes_whitespace(self):
        assert normalize_for_dup("a b\tc\nd") == "abcd"

    def test_removes_punct(self):
        assert normalize_for_dup("a，b。c") == "abc"

    def test_chinese_preserved(self):
        assert normalize_for_dup("你好，世界") == "你好世界"

    def test_ngrams_basic(self):
        grams = ngrams("abcdef", 3)
        assert grams == {"abc", "bcd", "cde", "def"}

    def test_ngrams_short_input(self):
        assert ngrams("ab", 3) == {"ab"}


# ----------------------------- 测试：审计主流程 -----------------------------

def _make_row(instruction: str, output: str) -> dict:
    return {"instruction": instruction, "output": output}


class TestAudit:
    def test_clean_rows_kept(self):
        rows = [
            _make_row("如何实现 RAG？", "RAG 包括检索和生成两个阶段，结合向量化与 LLM。"),
            _make_row("什么是 embedding？", "Embedding 把文本映射为高维向量，用于检索和聚类。"),
        ]
        stats = audit(
            rows,
            max_inst_chars=1000, max_out_chars=5000,
            min_inst_chars=2, min_out_chars=10,
            ngram_size=12,
        )
        assert stats["schema_ok"] == 2
        assert stats["keep"] == 2
        assert stats["duplicates_exact"] == 0

    def test_exact_duplicate_detected(self):
        rows = [
            _make_row("什么是 X？", "X 是一个示例。"),
            _make_row("什么是 X？", "X 是一个示例。"),
        ]
        stats = audit(
            rows, max_inst_chars=1000, max_out_chars=5000,
            min_inst_chars=2, min_out_chars=10, ngram_size=12,
        )
        assert stats["duplicates_exact"] == 1

    def test_schema_fail_quarantined(self):
        rows = [_make_row("好", "正常"), {"foo": "bar"}]  # 第二条 schema 缺
        stats = audit(
            rows, max_inst_chars=1000, max_out_chars=5000,
            min_inst_chars=2, min_out_chars=10, ngram_size=12,
        )
        assert stats["schema_ok"] == 1
        assert stats["schema_fail"] == 1

    def test_oversize_quarantined(self):
        rows = [_make_row("x" * 30, "a")]  # 过短
        rows[0]["instruction"] = "短"
        rows[0]["output"] = "短"
        stats = audit(
            rows, max_inst_chars=2000, max_out_chars=5000,
            min_inst_chars=10, min_out_chars=20,  # 输出不足 min
            ngram_size=12,
        )
        assert stats["too_short"] == 1

    def test_code_parse_fail_quarantined(self):
        rows = [_make_row(
            "示例？",
            "代码：\n```python\ndef x(:\n  pass\n```\n结束",
        )]
        stats = audit(
            rows, max_inst_chars=2000, max_out_chars=5000,
            min_inst_chars=2, min_out_chars=10, ngram_size=12,
        )
        assert stats["code_parse_fail"] >= 1

    def test_hard_metric_warns(self):
        rows = [_make_row("Q？", "性能提升了 50%。")]
        stats = audit(
            rows, max_inst_chars=2000, max_out_chars=5000,
            min_inst_chars=2, min_out_chars=10, ngram_size=12,
        )
        assert stats["hard_metric_count"] >= 1
        # 但保留（不剔除）
        assert stats["keep"] == 1

    def test_absolutist_warns(self):
        rows = [_make_row("Q？", "这个方案 100% 正确，永远不会失败。")]
        stats = audit(
            rows, max_inst_chars=2000, max_out_chars=5000,
            min_inst_chars=2, min_out_chars=10, ngram_size=12,
        )
        assert stats["absolutist_count"] >= 1


# ----------------------------- 测试：划分 -----------------------------

class TestSplit:
    def test_ratios_sum_to_one_validation(self):
        with pytest.raises(ValueError):
            split_indices(100, train=0.5, val=0.3, test=0.1, seed=42)

    def test_total_count_matches(self):
        tr, va, te = split_indices(100, train=0.8, val=0.1, test=0.1, seed=42)
        assert len(tr) + len(va) + len(te) == 100

    def test_no_overlap(self):
        tr, va, te = split_indices(100, train=0.8, val=0.1, test=0.1, seed=42)
        all_idx = set(tr) | set(va) | set(te)
        assert len(all_idx) == 100  # 每条只出现一次

    def test_reproducibility(self):
        # 同一 seed → 同一划分
        tr1, va1, te1 = split_indices(50, train=0.8, val=0.1, test=0.1, seed=42)
        tr2, va2, te2 = split_indices(50, train=0.8, val=0.1, test=0.1, seed=42)
        assert tr1 == tr2 and va1 == va2 and te1 == te2

    def test_different_seed_differs(self):
        tr1, _, _ = split_indices(50, train=0.8, val=0.1, test=0.1, seed=42)
        tr2, _, _ = split_indices(50, train=0.8, val=0.1, test=0.1, seed=7)
        # 高概率不同（不是绝对保证，但 seed=42 vs 7 应不同）
        assert tr1 != tr2


# ----------------------------- 测试：实际数据冒烟 -----------------------------

class TestRealData:
    """对实际生成的数据 v2 做端到端冒烟。前提：数据已生成。"""

    @pytest.fixture(scope="class")
    def v2_audit_dir(self) -> Path:
        d = ROOT / "data" / "audit" / "2026-07-22-v2"
        if not d.exists():
            pytest.skip(f"audit dir not found: {d}")
        return d

    def test_train_split_loadable(self, v2_audit_dir: Path):
        rows = read_jsonl(v2_audit_dir / "train.jsonl")
        assert len(rows) > 100
        for r in rows[:10]:
            assert "instruction" in r and "output" in r

    def test_test_split_isolated(self, v2_audit_dir: Path):
        train = read_jsonl(v2_audit_dir / "train.jsonl")
        test = read_jsonl(v2_audit_dir / "test.jsonl")
        val = read_jsonl(v2_audit_dir / "val.jsonl")
        train_set = {(r["instruction"], r["output"]) for r in train}
        test_set = {(r["instruction"], r["output"]) for r in test}
        val_set = {(r["instruction"], r["output"]) for r in val}
        assert train_set & test_set == set(), "train/test contamination"
        assert train_set & val_set == set(), "train/val contamination"

    def test_audit_json_schema(self, v2_audit_dir: Path):
        audit_json = json.loads((v2_audit_dir / "dataset_audit.json").read_text(encoding="utf-8"))
        required = {"total", "schema_ok", "keep", "quarantine", "duplicates_exact", "duplicates_near",
                    "code_parse_fail", "hard_metric_count", "absolutist_count"}
        missing = required - set(audit_json.keys())
        assert not missing, f"Missing keys: {missing}"

    def test_splits_json_has_seed(self, v2_audit_dir: Path):
        s = json.loads((v2_audit_dir / "splits.json").read_text(encoding="utf-8"))
        assert "seed" in s and s["seed"] == 42
