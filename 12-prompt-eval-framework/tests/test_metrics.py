"""评估指标测试。"""

import pytest

from metrics.bleu import BLEUMetric
from metrics.rouge import ROUGEMetric
from metrics.bertscore import BERTScoreMetric
from metrics.exact_match import ExactMatchMetric
from metrics.f1_score import F1ScoreMetric
from metrics.registry import MetricRegistry


# ---- 测试数据 ----
PREDICTIONS = [
    "The cat is on the mat.",
    "A quick brown fox jumps over the lazy dog.",
    "Hello world!",
]

REFERENCES = [
    "The cat is sitting on the mat.",
    "The quick brown fox jumped over the lazy dog.",
    "Hello world!",
]


# ---- BLEU 测试 ----

class TestBLEU:
    def test_bleu_compute_returns_dict(self):
        metric = BLEUMetric()
        result = metric.compute(PREDICTIONS, REFERENCES)
        assert isinstance(result, dict)

    def test_bleu_has_all_ngram_scores(self):
        metric = BLEUMetric(max_n=4)
        result = metric.compute(PREDICTIONS, REFERENCES)
        assert "bleu-1" in result
        assert "bleu-2" in result
        assert "bleu-3" in result
        assert "bleu-4" in result

    def test_bleu_perfect_match_high_score(self):
        metric = BLEUMetric()
        result = metric.compute(["hello world"], ["hello world"])
        assert result["bleu-1"] > 0.9

    def test_bleu_length_mismatch_raises(self):
        metric = BLEUMetric()
        with pytest.raises(ValueError, match="长度"):
            metric.compute(["a"], ["b", "c"])

    def test_bleu_max_n_limits_output(self):
        metric = BLEUMetric(max_n=2)
        result = metric.compute(PREDICTIONS, REFERENCES)
        assert "bleu-1" in result
        assert "bleu-2" in result
        assert "bleu-3" not in result
        assert "bleu-4" not in result


# ---- ROUGE 测试 ----

class TestROUGE:
    def test_rouge_compute_returns_dict(self):
        metric = ROUGEMetric()
        result = metric.compute(PREDICTIONS, REFERENCES)
        assert isinstance(result, dict)

    def test_rouge_has_all_types(self):
        metric = ROUGEMetric()
        result = metric.compute(PREDICTIONS, REFERENCES)
        assert "rouge1" in result
        assert "rouge2" in result
        assert "rougeL" in result
        assert "rougeLsum" in result

    def test_rouge_perfect_match(self):
        metric = ROUGEMetric()
        result = metric.compute(["hello world"], ["hello world"])
        assert result["rouge1"] > 0.9

    def test_rouge_length_mismatch_raises(self):
        metric = ROUGEMetric()
        with pytest.raises(ValueError, match="长度"):
            metric.compute(["a"], ["b", "c"])


# ---- BERTScore 测试 ----

# ⚠️ 2026-07-22: sentence-transformers 需要联网下载 all-MiniLM-L6-v2 (~80 MB)。
# 默认情况下需要主动网络（huggingface.co）；离线环境设 SKIP_HF_DOWNLOAD=1
# 跳过；或设 HF_HUB_OFFLINE=1 让 sentence-transformers 走本地 cache。
import os as _os
_HF_AVAILABLE = (
    _os.getenv("HF_HUB_OFFLINE") == "1"
    or _os.getenv("SKIP_HF_DOWNLOAD") != "1"  # 默认行为：跑；显式 SKIP=1 才跳过
    and _os.path.isdir(_os.path.expanduser("~/.cache/huggingface/hub"))
) and _os.getenv("SKIP_HF_DOWNLOAD") != "1"
requires_hf_model = pytest.mark.skipif(
    not _HF_AVAILABLE,
    reason="BERTScore requires network for huggingface.co download. "
    "Set SKIP_HF_DOWNLOAD=1 to skip; pre-download with HF_HUB_OFFLINE=1.",
)


class TestBERTScore:
    @requires_hf_model
    def test_bertscore_compute_returns_dict(self):
        metric = BERTScoreMetric()
        result = metric.compute(PREDICTIONS, REFERENCES)
        assert isinstance(result, dict)

    @requires_hf_model
    def test_bertscore_has_all_metrics(self):
        metric = BERTScoreMetric()
        result = metric.compute(PREDICTIONS, REFERENCES)
        assert "bertscore_precision" in result
        assert "bertscore_recall" in result
        assert "bertscore_f1" in result
        assert "bertscore_cosine" in result

    @requires_hf_model
    def test_bertscore_perfect_match_high_score(self):
        metric = BERTScoreMetric()
        result = metric.compute(["the cat sat on the mat"], ["the cat sat on the mat"])
        assert result["bertscore_f1"] > 0.9

    def test_bertscore_length_mismatch_raises(self):
        """不需要模型，只测参数校验。"""
        metric = BERTScoreMetric()
        with pytest.raises(ValueError, match="长度"):
            metric.compute(["a"], ["b", "c"])


# ---- Exact Match 测试 ----

class TestExactMatch:
    def test_exact_match_perfect(self):
        metric = ExactMatchMetric()
        result = metric.compute(["hello world"], ["hello world"])
        assert result["exact_match"] == 1.0
        assert result["exact_match_count"] == 1

    def test_exact_match_none(self):
        metric = ExactMatchMetric()
        result = metric.compute(["hello world"], ["goodbye world"])
        assert result["exact_match"] == 0.0

    def test_exact_match_ignore_case(self):
        metric = ExactMatchMetric(ignore_case=True)
        result = metric.compute(["Hello World"], ["hello world"])
        assert result["exact_match"] == 1.0

    def test_exact_match_ignore_punctuation(self):
        metric = ExactMatchMetric()
        result = metric.compute(["hello, world!"], ["hello world"])
        assert result["exact_match"] == 1.0


# ---- F1 Score 测试 ----

class TestF1Score:
    def test_f1_perfect(self):
        metric = F1ScoreMetric()
        result = metric.compute(["the cat sat"], ["the cat sat"])
        assert result["f1"] == 1.0

    def test_f1_partial(self):
        metric = F1ScoreMetric()
        result = metric.compute(["the cat"], ["the cat sat on the mat"])
        assert 0 < result["f1"] < 1

    def test_f1_no_overlap(self):
        metric = F1ScoreMetric()
        result = metric.compute(["abc"], ["xyz"])
        assert result["f1"] == 0.0


# ---- MetricRegistry 测试 ----

class TestMetricRegistry:
    def test_registry_has_defaults(self):
        registry = MetricRegistry()
        metrics = registry.list_metrics()
        assert len(metrics) >= 5

    def test_registry_get_by_name(self):
        registry = MetricRegistry()
        metric = registry.get("bleu")
        assert metric is not None
        assert metric.name == "bleu"

    def test_registry_get_unknown_raises(self):
        registry = MetricRegistry()
        with pytest.raises(ValueError, match="未知指标"):
            registry.get_metrics(["nonexistent"])

    def test_registry_custom_metric(self):
        registry = MetricRegistry()
        registry.register_custom(
            name="custom_test",
            description="A test custom metric",
            compute_fn=lambda preds, refs: {"custom_score": 0.5},
        )
        metric = registry.get("custom_test")
        assert metric is not None
        result = metric.compute(["a"], ["b"])
        assert result["custom_score"] == 0.5

    def test_registry_list_metrics_format(self):
        registry = MetricRegistry()
        metrics = registry.list_metrics()
        for m in metrics:
            assert "name" in m
            assert "description" in m
