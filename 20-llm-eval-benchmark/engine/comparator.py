"""
engine/comparator.py - 模型对比分析

支持多模型评测结果的对比、差异分析和排名。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from benchmarks.base import BenchmarkResult
from .scorer import Scorer, ScoreReport


@dataclass
class ModelComparison:
    """两个模型的对比结果."""
    model_a: str
    model_b: str
    benchmark_name: str
    metric_a: float
    metric_b: float
    difference: float
    significant: bool
    p_value: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "model_a": self.model_a,
            "model_b": self.model_b,
            "benchmark": self.benchmark_name,
            "metric_a": round(self.metric_a, 4),
            "metric_b": round(self.metric_b, 4),
            "difference": round(self.difference, 4),
            "significant": self.significant,
            "p_value": round(self.p_value, 4) if self.p_value is not None else None,
            "better_model": self.model_a if self.difference > 0 else (
                self.model_b if self.difference < 0 else "tie"
            ),
            "details": self.details,
        }


@dataclass
class ModelEvalResult:
    """单个模型的完整评测结果."""
    model_name: str
    benchmark_name: str
    results: List[BenchmarkResult]
    scores: Dict[str, ScoreReport] = field(default_factory=dict)
    eval_time_seconds: float = 0.0

    def compute_scores(self) -> Dict[str, ScoreReport]:
        """计算所有评分指标."""
        self.scores = Scorer.full_report(self.results)
        return self.scores

    @property
    def accuracy(self) -> float:
        if "accuracy" not in self.scores:
            self.compute_scores()
        return self.scores["accuracy"].value

    def to_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "benchmark_name": self.benchmark_name,
            "total_questions": len(self.results),
            "correct": sum(1 for r in self.results if r.correct),
            "accuracy": round(self.accuracy, 4),
            "eval_time_seconds": round(self.eval_time_seconds, 2),
            "scores": {k: v.to_dict() for k, v in self.scores.items()},
        }


class Comparator:
    """模型对比器.

    支持多模型评测结果的对比和排名。
    """

    def __init__(self):
        self._model_results: Dict[str, Dict[str, ModelEvalResult]] = {}
        # {model_name: {benchmark_name: ModelEvalResult}}

    def add_result(self, result: ModelEvalResult) -> None:
        """添加模型评测结果."""
        if result.model_name not in self._model_results:
            self._model_results[result.model_name] = {}
        result.compute_scores()
        self._model_results[result.model_name][result.benchmark_name] = result

    def compare_models(
        self,
        model_a: str,
        model_b: str,
        benchmark_name: Optional[str] = None,
    ) -> List[ModelComparison]:
        """对比两个模型."""
        comparisons = []

        if model_a not in self._model_results:
            raise ValueError(f"Model '{model_a}' not found")
        if model_b not in self._model_results:
            raise ValueError(f"Model '{model_b}' not found")

        benchmarks_a = self._model_results[model_a]
        benchmarks_b = self._model_results[model_b]

        # 确定要对比的基准
        if benchmark_name:
            bench_names = [benchmark_name]
        else:
            bench_names = set(benchmarks_a.keys()) & set(benchmarks_b.keys())

        for bname in bench_names:
            result_a = benchmarks_a.get(bname)
            result_b = benchmarks_b.get(bname)

            if not result_a or not result_b:
                continue

            metric_a = result_a.accuracy
            metric_b = result_b.accuracy

            # 统计显著性检验
            sig_report = Scorer.statistical_significance(
                result_a.results, result_b.results
            )

            comparisons.append(ModelComparison(
                model_a=model_a,
                model_b=model_b,
                benchmark_name=bname,
                metric_a=metric_a,
                metric_b=metric_b,
                difference=metric_a - metric_b,
                significant=sig_report.details.get("significant", False),
                p_value=sig_report.value if sig_report.value >= 0 else None,
                details={
                    "model_a_scores": {k: v.to_dict() for k, v in result_a.scores.items()},
                    "model_b_scores": {k: v.to_dict() for k, v in result_b.scores.items()},
                },
            ))

        return comparisons

    def get_ranking(self, benchmark_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """获取模型排名.

        如果指定了 benchmark_name，返回该基准的排名。
        否则返回综合排名（所有基准的平均准确率）。
        """
        ranking = []

        if benchmark_name:
            for model_name, benchmarks in self._model_results.items():
                if benchmark_name in benchmarks:
                    result = benchmarks[benchmark_name]
                    ranking.append({
                        "model_name": model_name,
                        "benchmark": benchmark_name,
                        "score": round(result.accuracy, 4),
                        "total": len(result.results),
                        "correct": sum(1 for r in result.results if r.correct),
                    })
        else:
            # 综合排名
            for model_name, benchmarks in self._model_results.items():
                if benchmarks:
                    avg_acc = sum(r.accuracy for r in benchmarks.values()) / len(benchmarks)
                    ranking.append({
                        "model_name": model_name,
                        "benchmark": "overall",
                        "score": round(avg_acc, 4),
                        "benchmarks_evaluated": len(benchmarks),
                    })

        ranking.sort(key=lambda x: x["score"], reverse=True)
        for i, entry in enumerate(ranking, 1):
            entry["rank"] = i

        return ranking

    def get_all_results(self) -> Dict[str, Any]:
        """获取所有评测结果的汇总."""
        summary = {
            "models": list(self._model_results.keys()),
            "rankings": self.get_ranking(),
            "per_benchmark": {},
        }

        for model_name, benchmarks in self._model_results.items():
            summary["per_benchmark"][model_name] = {
                bname: result.to_dict()
                for bname, result in benchmarks.items()
            }

        return summary

    def generate_comparison_report(self) -> str:
        """生成文本格式的对比报告."""
        lines = ["=" * 60, "MODEL COMPARISON REPORT", "=" * 60, ""]

        rankings = self.get_ranking()
        lines.append("OVERALL RANKING:")
        for entry in rankings:
            lines.append(
                f"  #{entry['rank']} {entry['model_name']}: {entry['score']:.2%}"
            )
        lines.append("")

        # 逐基准对比
        all_benchmarks = set()
        for benchmarks in self._model_results.values():
            all_benchmarks.update(benchmarks.keys())

        for bname in sorted(all_benchmarks):
            lines.append(f"- {bname.upper()} -")
            bench_ranking = self.get_ranking(benchmark_name=bname)
            for entry in bench_ranking:
                lines.append(
                    f"  #{entry['rank']} {entry['model_name']}: "
                    f"{entry['score']:.2%} ({entry['correct']}/{entry['total']})"
                )
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)