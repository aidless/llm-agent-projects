"""
tests/test_leaderboard.py - 排行榜和可视化测试
"""

import os
import pytest

from benchmarks.base import BenchmarkResult
from engine.comparator import Comparator, ModelEvalResult
from leaderboard import Leaderboard, LeaderboardEntry, Visualizer


def _make_model_eval(model_name: str, benchmark_name: str, accuracy: float, n: int = 20) -> ModelEvalResult:
    """创建模型评测结果."""
    correct = int(n * accuracy)
    results = []
    for i in range(n):
        results.append(BenchmarkResult(
            question_id=i + 1,
            question=f"Q{i+1}",
            expected="A",
            predicted="A" if i < correct else "B",
            correct=(i < correct),
            score=1.0 if i < correct else 0.0,
        ))
    return ModelEvalResult(
        model_name=model_name,
        benchmark_name=benchmark_name,
        results=results,
    )


class TestComparator:
    """模型对比器测试."""

    def test_add_and_rank(self):
        """测试添加结果和排名."""
        comp = Comparator()
        comp.add_result(_make_model_eval("model-a", "mmlu", 0.8))
        comp.add_result(_make_model_eval("model-b", "mmlu", 0.6))

        ranking = comp.get_ranking("mmlu")
        assert len(ranking) == 2
        assert ranking[0]["model_name"] == "model-a"
        assert ranking[0]["rank"] == 1

    def test_overall_ranking(self):
        """测试综合排名."""
        comp = Comparator()
        comp.add_result(_make_model_eval("model-a", "mmlu", 0.9))
        comp.add_result(_make_model_eval("model-a", "gsm8k", 0.7))
        comp.add_result(_make_model_eval("model-b", "mmlu", 0.8))
        comp.add_result(_make_model_eval("model-b", "gsm8k", 0.8))

        ranking = comp.get_ranking()
        # model-b avg = 0.8, model-a avg = 0.8 -> same
        assert len(ranking) == 2

    def test_compare_models(self):
        """测试模型对比."""
        comp = Comparator()
        comp.add_result(_make_model_eval("model-a", "mmlu", 0.9, 20))
        comp.add_result(_make_model_eval("model-b", "mmlu", 0.5, 20))

        comparisons = comp.compare_models("model-a", "model-b", "mmlu")
        assert len(comparisons) == 1
        assert comparisons[0].difference > 0

    def test_compare_nonexistent_model(self):
        """测试对比不存在的模型."""
        comp = Comparator()
        comp.add_result(_make_model_eval("model-a", "mmlu", 0.9))
        with pytest.raises(ValueError):
            comp.compare_models("model-a", "nonexistent")

    def test_comparison_report(self):
        """测试对比报告生成."""
        comp = Comparator()
        comp.add_result(_make_model_eval("model-a", "mmlu", 0.9, 10))
        comp.add_result(_make_model_eval("model-b", "mmlu", 0.6, 10))

        report = comp.generate_comparison_report()
        assert "model-a" in report
        assert "model-b" in report
        assert "RANKING" in report


class TestLeaderboard:
    """排行榜测试."""

    def test_add_and_rank(self):
        """测试添加结果和排名."""
        lb = Leaderboard()
        lb.add_model_result(_make_model_eval("model-a", "mmlu", 0.9))
        lb.add_model_result(_make_model_eval("model-b", "mmlu", 0.7))
        lb.add_model_result(_make_model_eval("model-c", "mmlu", 0.8))

        ranking = lb.get_overall_ranking()
        assert ranking[0].model_name == "model-a"
        assert ranking[0].rank == 1
        assert ranking[1].model_name == "model-c"
        assert ranking[2].model_name == "model-b"

    def test_benchmark_ranking(self):
        """测试分基准排名."""
        lb = Leaderboard()
        lb.add_model_result(_make_model_eval("model-a", "mmlu", 0.9))
        lb.add_model_result(_make_model_eval("model-b", "mmlu", 0.95))

        ranking = lb.get_benchmark_ranking("mmlu")
        assert ranking[0]["model_name"] == "model-b"

    def test_model_card(self):
        """测试模型卡片."""
        lb = Leaderboard()
        lb.add_model_result(_make_model_eval("model-a", "mmlu", 0.9))
        lb.add_model_result(_make_model_eval("model-a", "gsm8k", 0.7))

        card = lb.get_model_card("model-a")
        assert card is not None
        assert card["model_name"] == "model-a"
        assert "strengths" in card
        assert "weaknesses" in card

    def test_model_card_not_found(self):
        """测试不存在的模型卡片."""
        lb = Leaderboard()
        assert lb.get_model_card("nonexistent") is None

    def test_available_benchmarks(self):
        """测试获取可用基准列表."""
        lb = Leaderboard()
        lb.add_model_result(_make_model_eval("model-a", "mmlu", 0.9))
        lb.add_model_result(_make_model_eval("model-a", "gsm8k", 0.8))

        benchmarks = lb.get_available_benchmarks()
        assert "mmlu" in benchmarks
        assert "gsm8k" in benchmarks

    def test_to_dict(self):
        """测试序列化."""
        lb = Leaderboard()
        lb.add_model_result(_make_model_eval("model-a", "mmlu", 0.9))
        d = lb.to_dict()
        assert d["total_models"] == 1
        assert len(d["overall_ranking"]) == 1


class TestLeaderboardEntry:
    """排行榜条目测试."""

    def test_to_dict(self):
        """测试序列化."""
        entry = LeaderboardEntry(
            rank=1, model_name="test", overall_score=0.85,
            benchmark_scores={"mmlu": 0.9, "gsm8k": 0.8},
        )
        d = entry.to_dict()
        assert d["rank"] == 1
        assert d["overall_score_pct"] == "85.00%"

    def test_empty_benchmark_scores(self):
        """测试空基准分数."""
        entry = LeaderboardEntry(rank=1, model_name="test", overall_score=0.5)
        d = entry.to_dict()
        assert d["benchmark_scores"] == {}


class TestVisualizer:
    """可视化测试."""

    def test_generate_bar_chart(self, tmp_path):
        """测试生成柱状图."""
        lb = Leaderboard()
        lb.add_model_result(_make_model_eval("model-a", "mmlu", 0.9))
        lb.add_model_result(_make_model_eval("model-b", "mmlu", 0.7))

        viz = Visualizer(output_dir=str(tmp_path))
        path = viz.plot_leaderboard_bar(lb, "test_bar.png")
        assert os.path.exists(path)

    def test_generate_all_charts(self, tmp_path):
        """测试生成所有图表."""
        lb = Leaderboard()
        lb.add_model_result(_make_model_eval("model-a", "mmlu", 0.9))
        lb.add_model_result(_make_model_eval("model-a", "gsm8k", 0.8))
        lb.add_model_result(_make_model_eval("model-b", "mmlu", 0.7))
        lb.add_model_result(_make_model_eval("model-b", "gsm8k", 0.75))

        viz = Visualizer(output_dir=str(tmp_path))
        charts = viz.generate_all_charts(lb)
        assert "leaderboard_bar" in charts
        assert "benchmark_comparison" in charts
        assert "radar_comparison" in charts
        for path in charts.values():
            assert os.path.exists(path)