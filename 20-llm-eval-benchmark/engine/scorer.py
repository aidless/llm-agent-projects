"""
engine/scorer.py - 评分系统

支持多种评分指标: 准确率、Pass@k、分数评分、F1、精确率、召回率、
置信区间和统计显著性检验。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from benchmarks.base import BenchmarkResult


@dataclass
class ScoreReport:
    """评分报告."""
    metric_name: str
    value: float
    confidence_low: Optional[float] = None
    confidence_high: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        result = {
            "metric": self.metric_name,
            "value": round(self.value, 4),
        }
        if self.confidence_low is not None:
            result["confidence_interval"] = {
                "low": round(self.confidence_low, 4),
                "high": round(self.confidence_high, 4),
            }
        if self.details:
            result["details"] = self.details
        return result


class Scorer:
    """评分系统.

    提供多种评分指标的计算方法。
    """

    @staticmethod
    def accuracy(results: List[BenchmarkResult], confidence: float = 0.95) -> ScoreReport:
        """计算准确率及其置信区间.

        使用 Wilson score interval 计算二项分布的置信区间。
        """
        if not results:
            return ScoreReport(metric_name="accuracy", value=0.0)

        correct = sum(1 for r in results if r.correct)
        total = len(results)
        acc = correct / total

        ci_low, ci_high = Scorer._wilson_interval(correct, total, confidence)

        return ScoreReport(
            metric_name="accuracy",
            value=acc,
            confidence_low=ci_low,
            confidence_high=ci_high,
            details={"correct": correct, "total": total},
        )

    @staticmethod
    def pass_at_k(
        results: List[BenchmarkResult],
        k: int = 1,
        n: int = 1,
    ) -> ScoreReport:
        """计算 Pass@k.

        Pass@k = 1 - C(n, k) / C(n+c, k), 其中 c 是正确数, n 是总尝试数。
        对于 k=1, n=1 的情况，退化为准确率。

        Args:
            results: 评测结果列表
            k: 至少通过 k 次才算通过
            n: 每题总尝试次数 (模拟场景)
        """
        if not results:
            return ScoreReport(metric_name=f"pass@{k}", value=0.0)

        if n <= 1:
            correct = sum(1 for r in results if r.correct)
            value = correct / len(results)
        else:
            # Pass@k 公式 (简化版): E[Pass@k] = 1 - (1 - p)^k
            # 其中 p 是单次通过概率
            correct = sum(1 for r in results if r.correct)
            p = correct / len(results) if results else 0
            value = 1 - (1 - p) ** k

        return ScoreReport(
            metric_name=f"pass@{k}",
            value=value,
            details={"k": k, "n": n},
        )

    @staticmethod
    def score_mean(results: List[BenchmarkResult]) -> ScoreReport:
        """计算平均分数评分 (1-10 分制).

        对于有 score 字段的结果，直接计算平均值并归一化到 10 分制。
        """
        if not results:
            return ScoreReport(metric_name="score_mean", value=0.0)

        scores = [r.score for r in results]
        mean_score = sum(scores) / len(scores)

        # 归一化到 10 分制
        max_possible = max(scores) if scores else 1
        if max_possible > 0:
            normalized = (mean_score / max_possible) * 10
        else:
            normalized = 0

        ci_low, ci_high = Scorer._mean_confidence_interval(scores, 0.95)

        return ScoreReport(
            metric_name="score_mean",
            value=round(normalized, 2),
            confidence_low=ci_low,
            confidence_high=ci_high,
            details={
                "raw_mean": round(mean_score, 4),
                "min": min(scores),
                "max": max(scores),
                "count": len(scores),
            },
        )

    @staticmethod
    def f1_score(
        results: List[BenchmarkResult],
        get_predicted_tokens: Optional[Any] = None,
        get_expected_tokens: Optional[Any] = None,
    ) -> ScoreReport:
        """计算 F1 分数.

        对于分类结果，使用正确/错误计算 precision/recall/F1。
        """
        if not results:
            return ScoreReport(metric_name="f1", value=0.0)

        tp = sum(1 for r in results if r.correct)
        fp = sum(1 for r in results if not r.correct)
        fn = fp  # 简化: FP == FN 对于二分类

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        return ScoreReport(
            metric_name="f1",
            value=f1,
            details={
                "precision": round(precision, 4),
                "recall": round(recall, 4),
                "true_positives": tp,
                "false_positives": fp,
            },
        )

    @staticmethod
    def precision_score(results: List[BenchmarkResult]) -> ScoreReport:
        """计算精确率."""
        if not results:
            return ScoreReport(metric_name="precision", value=0.0)
        tp = sum(1 for r in results if r.correct)
        total_predicted = len(results)
        precision = tp / total_predicted if total_predicted > 0 else 0
        return ScoreReport(
            metric_name="precision",
            value=precision,
            details={"true_positives": tp, "total_predicted": total_predicted},
        )

    @staticmethod
    def recall_score(results: List[BenchmarkResult]) -> ScoreReport:
        """计算召回率."""
        if not results:
            return ScoreReport(metric_name="recall", value=0.0)
        tp = sum(1 for r in results if r.correct)
        total_actual = len(results)
        recall = tp / total_actual if total_actual > 0 else 0
        return ScoreReport(
            metric_name="recall",
            value=recall,
            details={"true_positives": tp, "total_actual": total_actual},
        )

    @staticmethod
    def category_breakdown(
        results: List[BenchmarkResult],
        get_category: callable,
    ) -> Dict[str, ScoreReport]:
        """按类别计算准确率."""
        from collections import defaultdict

        by_category: Dict[str, List[BenchmarkResult]] = defaultdict(list)
        for r in results:
            cat = get_category(r)
            by_category[cat].append(r)

        breakdown = {}
        for cat, cat_results in by_category.items():
            breakdown[cat] = Scorer.accuracy(cat_results)

        return breakdown

    @staticmethod
    def statistical_significance(
        results_a: List[BenchmarkResult],
        results_b: List[BenchmarkResult],
    ) -> ScoreReport:
        """两组结果的统计显著性检验.

        使用 McNemar 检验 (配对二分类):
        - n_01: A 错误 B 正确的次数
        - n_10: A 正确 B 错误的次数
        """
        if len(results_a) != len(results_b):
            return ScoreReport(
                metric_name="mcnemar_p_value",
                value=-1,
                details={"error": "Result lists must have the same length"},
            )

        n_01 = 0  # A wrong, B correct
        n_10 = 0  # A correct, B wrong

        for ra, rb in zip(results_a, results_b):
            a_correct = ra.correct
            b_correct = rb.correct
            if not a_correct and b_correct:
                n_01 += 1
            elif a_correct and not b_correct:
                n_10 += 1

        # McNemar 检验 (连续性校正)
        statistic = (abs(n_10 - n_01) - 1) ** 2 / (n_10 + n_01) if (n_10 + n_01) > 0 else 0

        # 近似 p 值 (chi-squared df=1)
        # 对于 df=1, p < 0.05 当 statistic > 3.841
        if statistic > 3.841:
            p_value = 0.03  # 简化
        elif statistic > 2.706:
            p_value = 0.10  # 简化
        else:
            p_value = 0.50  # 简化

        significant = p_value < 0.05

        return ScoreReport(
            metric_name="mcnemar_p_value",
            value=p_value,
            details={
                "statistic": round(statistic, 4),
                "significant": significant,
                "n_01": n_01,
                "n_10": n_10,
                "alpha": 0.05,
            },
        )

    @staticmethod
    def _wilson_interval(
        correct: int, total: int, confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Wilson score interval."""
        if total == 0:
            return 0.0, 0.0

        p_hat = correct / total
        z = 1.96  # 95% confidence
        if confidence == 0.99:
            z = 2.576
        elif confidence == 0.90:
            z = 1.645

        denominator = 1 + z ** 2 / total
        center = (p_hat + z ** 2 / (2 * total)) / denominator
        spread = z * math.sqrt(
            (p_hat * (1 - p_hat) + z ** 2 / (4 * total)) / total
        ) / denominator

        return max(0, center - spread), min(1, center + spread)

    @staticmethod
    def _mean_confidence_interval(
        values: List[float], confidence: float = 0.95
    ) -> Tuple[Optional[float], Optional[float]]:
        """计算均值的置信区间."""
        if len(values) < 2:
            return None, None

        n = len(values)
        mean = sum(values) / n
        variance = sum((v - mean) ** 2 for v in values) / (n - 1)
        std = math.sqrt(variance)
        se = std / math.sqrt(n)

        z = 1.96  # 95% confidence
        if confidence == 0.99:
            z = 2.576
        elif confidence == 0.90:
            z = 1.645

        return mean - z * se, mean + z * se

    @classmethod
    def full_report(cls, results: List[BenchmarkResult]) -> Dict[str, ScoreReport]:
        """生成完整的评分报告."""
        return {
            "accuracy": cls.accuracy(results),
            "pass@1": cls.pass_at_k(results, k=1),
            "f1": cls.f1_score(results),
            "precision": cls.precision_score(results),
            "recall": cls.recall_score(results),
        }