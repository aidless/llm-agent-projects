"""
tests/test_scorer.py - 评分系统测试

测试所有评分指标: 准确率、Pass@k、F1、置信区间、显著性检验。
"""

import pytest
from benchmarks.base import BenchmarkResult
from engine.scorer import Scorer, ScoreReport


def _make_results(correct_count: int, total: int) -> list:
    """创建测试用的评测结果列表."""
    results = []
    for i in range(total):
        results.append(BenchmarkResult(
            question_id=i + 1,
            question=f"Question {i+1}",
            expected="A",
            predicted="A" if i < correct_count else "B",
            correct=(i < correct_count),
            score=1.0 if i < correct_count else 0.0,
        ))
    return results


class TestAccuracy:
    """准确率测试."""

    def test_perfect_accuracy(self):
        """测试完美准确率."""
        results = _make_results(10, 10)
        report = Scorer.accuracy(results)
        assert report.value == 1.0
        assert report.details["correct"] == 10

    def test_zero_accuracy(self):
        """测试零准确率."""
        results = _make_results(0, 10)
        report = Scorer.accuracy(results)
        assert report.value == 0.0

    def test_partial_accuracy(self):
        """测试部分准确率."""
        results = _make_results(7, 10)
        report = Scorer.accuracy(results)
        assert report.value == 0.7

    def test_empty_results(self):
        """测试空结果."""
        report = Scorer.accuracy([])
        assert report.value == 0.0

    def test_confidence_interval(self):
        """测试置信区间存在."""
        results = _make_results(7, 10)
        report = Scorer.accuracy(results)
        assert report.confidence_low is not None
        assert report.confidence_high is not None
        assert report.confidence_low <= report.value <= report.confidence_high

    def test_confidence_interval_bounds(self):
        """测试置信区间在 [0, 1] 范围内."""
        results = _make_results(10, 10)
        report = Scorer.accuracy(results)
        assert 0 <= report.confidence_low <= 1
        assert 0 <= report.confidence_high <= 1

    def test_to_dict(self):
        """测试序列化."""
        results = _make_results(5, 10)
        report = Scorer.accuracy(results)
        d = report.to_dict()
        assert d["metric"] == "accuracy"
        assert "value" in d
        assert "confidence_interval" in d


class TestPassAtK:
    """Pass@k 测试."""

    def test_pass_at_1(self):
        """测试 Pass@1."""
        results = _make_results(8, 10)
        report = Scorer.pass_at_k(results, k=1)
        assert report.metric_name == "pass@1"
        assert report.value == 0.8

    def test_pass_at_k_empty(self):
        """测试空结果."""
        report = Scorer.pass_at_k([], k=1)
        assert report.value == 0.0


class TestF1Score:
    """F1 分数测试."""

    def test_perfect_f1(self):
        """测试完美 F1."""
        results = _make_results(10, 10)
        report = Scorer.f1_score(results)
        assert report.value == 1.0

    def test_zero_f1(self):
        """测试零 F1."""
        results = _make_results(0, 10)
        report = Scorer.f1_score(results)
        assert report.value == 0.0

    def test_partial_f1(self):
        """测试部分 F1."""
        results = _make_results(5, 10)
        report = Scorer.f1_score(results)
        assert 0 < report.value < 1
        assert "precision" in report.details
        assert "recall" in report.details


class TestPrecisionRecall:
    """精确率和召回率测试."""

    def test_precision(self):
        """测试精确率."""
        results = _make_results(6, 10)
        report = Scorer.precision_score(results)
        assert report.value == 0.6
        assert report.details["true_positives"] == 6

    def test_recall(self):
        """测试召回率."""
        results = _make_results(6, 10)
        report = Scorer.recall_score(results)
        assert report.value == 0.6


class TestCategoryBreakdown:
    """分类别评分测试."""

    def test_category_breakdown(self):
        """测试按类别统计准确率."""
        results = []
        for i in range(5):
            results.append(BenchmarkResult(
                question_id=i + 1,
                question=f"Q{i+1}",
                expected="A",
                predicted="A" if i < 3 else "B",
                correct=(i < 3),
                score=1.0 if i < 3 else 0.0,
                metadata={"category": "math" if i < 3 else "physics"},
            ))

        breakdown = Scorer.category_breakdown(
            results, lambda r: r.metadata.get("category", "unknown")
        )

        assert "math" in breakdown
        assert "physics" in breakdown
        assert breakdown["math"].value == 1.0
        assert breakdown["physics"].value == 0.0


class TestStatisticalSignificance:
    """统计显著性检验测试."""

    def test_same_results(self):
        """测试相同结果."""
        results_a = _make_results(7, 10)
        results_b = _make_results(7, 10)
        report = Scorer.statistical_significance(results_a, results_b)
        assert report.details["n_01"] == 0
        assert report.details["n_10"] == 0

    def test_different_lengths(self):
        """测试不同长度的结果."""
        results_a = _make_results(7, 10)
        results_b = _make_results(5, 8)
        report = Scorer.statistical_significance(results_a, results_b)
        assert report.value == -1
        assert "error" in report.details


class TestFullReport:
    """完整报告测试."""

    def test_full_report_keys(self):
        """测试完整报告包含所有指标."""
        results = _make_results(7, 10)
        report = Scorer.full_report(results)
        assert "accuracy" in report
        assert "pass@1" in report
        assert "f1" in report
        assert "precision" in report
        assert "recall" in report

    def test_full_report_all_score_reports(self):
        """测试完整报告中的所有值都是 ScoreReport."""
        results = _make_results(7, 10)
        report = Scorer.full_report(results)
        for key, value in report.items():
            assert isinstance(value, ScoreReport)