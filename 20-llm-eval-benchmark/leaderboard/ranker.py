"""
leaderboard/ranker.py - 排行榜系统

支持综合排名、分基准排名和模型卡片生成。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from engine.comparator import ModelEvalResult


@dataclass
class LeaderboardEntry:
    """排行榜条目."""
    rank: int
    model_name: str
    overall_score: float
    benchmark_scores: Dict[str, float] = field(default_factory=dict)
    total_questions: int = 0
    total_correct: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "model_name": self.model_name,
            "overall_score": round(self.overall_score, 4),
            "overall_score_pct": f"{self.overall_score:.2%}",
            "benchmark_scores": {
                k: round(v, 4) for k, v in self.benchmark_scores.items()
            },
            "total_questions": self.total_questions,
            "total_correct": self.total_correct,
        }


class Leaderboard:
    """排行榜.

    管理模型评测结果的排名和展示。
    """

    def __init__(self):
        self._entries: Dict[str, LeaderboardEntry] = {}
        self._benchmarks: Dict[str, Dict[str, float]] = {}
        # {model_name: {benchmark_name: accuracy}}

    def add_model_result(self, result: ModelEvalResult) -> None:
        """添加模型评测结果到排行榜."""
        result.compute_scores()
        accuracy = result.accuracy

        if result.model_name not in self._benchmarks:
            self._benchmarks[result.model_name] = {}
        self._benchmarks[result.model_name][result.benchmark_name] = accuracy

        self._update_entry(result.model_name)

    def _update_entry(self, model_name: str) -> None:
        """更新模型在排行榜中的条目."""
        bench_scores = self._benchmarks.get(model_name, {})
        if not bench_scores:
            return

        overall = sum(bench_scores.values()) / len(bench_scores)

        entry = LeaderboardEntry(
            rank=0,
            model_name=model_name,
            overall_score=overall,
            benchmark_scores=bench_scores,
        )
        self._entries[model_name] = entry

    def get_overall_ranking(self) -> List[LeaderboardEntry]:
        """获取综合排行榜."""
        entries = sorted(
            self._entries.values(),
            key=lambda e: e.overall_score,
            reverse=True,
        )
        for i, entry in enumerate(entries, 1):
            entry.rank = i
        return entries

    def get_benchmark_ranking(self, benchmark_name: str) -> List[Dict[str, Any]]:
        """获取指定基准的排行榜."""
        scores = []
        for model_name, benchmarks in self._benchmarks.items():
            if benchmark_name in benchmarks:
                scores.append({
                    "model_name": model_name,
                    "score": round(benchmarks[benchmark_name], 4),
                })

        scores.sort(key=lambda x: x["score"], reverse=True)
        for i, entry in enumerate(scores, 1):
            entry["rank"] = i

        return scores

    def get_model_card(self, model_name: str) -> Optional[Dict[str, Any]]:
        """获取模型卡片信息."""
        entry = self._entries.get(model_name)
        if not entry:
            return None

        return {
            "model_name": model_name,
            "overall_rank": entry.rank,
            "overall_score": round(entry.overall_score, 4),
            "benchmark_breakdown": entry.benchmark_scores,
            "strengths": sorted(
                entry.benchmark_scores.items(),
                key=lambda x: x[1],
                reverse=True,
            )[:3],
            "weaknesses": sorted(
                entry.benchmark_scores.items(),
                key=lambda x: x[1],
            )[:3],
        }

    def get_available_benchmarks(self) -> List[str]:
        """获取所有已评测的基准."""
        benchmarks = set()
        for model_benchmarks in self._benchmarks.values():
            benchmarks.update(model_benchmarks.keys())
        return sorted(benchmarks)

    def to_dict(self) -> Dict[str, Any]:
        """导出排行榜数据."""
        return {
            "total_models": len(self._entries),
            "available_benchmarks": self.get_available_benchmarks(),
            "overall_ranking": [e.to_dict() for e in self.get_overall_ranking()],
        }