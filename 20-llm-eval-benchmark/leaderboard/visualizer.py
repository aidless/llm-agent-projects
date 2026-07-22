"""
leaderboard/visualizer.py - 可视化模块

使用 matplotlib 生成排行榜图表: 雷达图、柱状图、趋势图等。
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # 非交互式后端
import matplotlib.pyplot as plt
import numpy as np

from .ranker import Leaderboard


class Visualizer:
    """可视化器.

    生成排行榜相关的图表 PNG 文件。
    """

    def __init__(self, output_dir: str = "reports"):
        self._output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        plt.style.use("seaborn-v0_8-whitegrid") if "seaborn-v0_8-whitegrid" in plt.style.available else plt.style.use("ggplot")

    def _save_fig(self, fig: plt.Figure, filename: str, dpi: int = 150) -> str:
        """保存图表到文件."""
        filepath = os.path.join(self._output_dir, filename)
        fig.savefig(filepath, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return filepath

    def plot_leaderboard_bar(
        self,
        leaderboard: Leaderboard,
        filename: str = "leaderboard_bar.png",
    ) -> str:
        """绘制综合排行榜柱状图."""
        ranking = leaderboard.get_overall_ranking()
        if not ranking:
            return ""

        model_names = [e.model_name for e in ranking]
        scores = [e.overall_score * 100 for e in ranking]

        fig, ax = plt.subplots(figsize=(max(8, len(ranking) * 1.5), 6))

        colors = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(ranking)))
        bars = ax.barh(
            range(len(model_names)),
            scores,
            color=colors,
            edgecolor="gray",
            linewidth=0.5,
        )

        ax.set_yticks(range(len(model_names)))
        ax.set_yticklabels(model_names, fontsize=10)
        ax.set_xlabel("Overall Score (%)", fontsize=12)
        ax.set_title("Model Leaderboard - Overall Score", fontsize=14, fontweight="bold")
        ax.invert_yaxis()

        # 添加分数标签
        for bar, score in zip(bars, scores):
            ax.text(
                bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"{score:.1f}%",
                va="center",
                fontsize=9,
            )

        ax.set_xlim(0, max(scores) * 1.15 if scores else 100)
        return self._save_fig(fig, filename)

    def plot_benchmark_comparison(
        self,
        leaderboard: Leaderboard,
        filename: str = "benchmark_comparison.png",
    ) -> str:
        """绘制多模型多基准对比柱状图."""
        ranking = leaderboard.get_overall_ranking()
        benchmarks = leaderboard.get_available_benchmarks()

        if not ranking or not benchmarks:
            return ""

        model_names = [e.model_name for e in ranking[:10]]  # 最多显示10个模型
        x = np.arange(len(benchmarks))
        width = 0.8 / max(len(model_names), 1)

        fig, ax = plt.subplots(figsize=(max(8, len(benchmarks) * 2), 6))

        for i, model_name in enumerate(model_names):
            entry = None
            for e in ranking:
                if e.model_name == model_name:
                    entry = e
                    break
            if not entry:
                continue

            scores = [
                entry.benchmark_scores.get(b, 0) * 100 for b in benchmarks
            ]
            offset = (i - len(model_names) / 2 + 0.5) * width
            ax.bar(x + offset, scores, width, label=model_name)

        ax.set_xlabel("Benchmark")
        ax.set_ylabel("Score (%)")
        ax.set_title("Multi-Benchmark Comparison", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(benchmarks, rotation=45, ha="right")
        ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)

        return self._save_fig(fig, filename)

    def plot_radar(
        self,
        leaderboard: Leaderboard,
        model_names: Optional[List[str]] = None,
        filename: str = "radar_comparison.png",
    ) -> str:
        """绘制雷达图对比多个模型."""
        ranking = leaderboard.get_overall_ranking()
        benchmarks = leaderboard.get_available_benchmarks()

        if not ranking or not benchmarks:
            return ""

        if model_names is None:
            model_names = [e.model_name for e in ranking[:5]]

        # 获取数据
        data = {}
        for mn in model_names:
            entry = None
            for e in ranking:
                if e.model_name == mn:
                    entry = e
                    break
            if entry:
                data[mn] = [
                    entry.benchmark_scores.get(b, 0) * 100 for b in benchmarks
                ]

        if not data:
            return ""

        # 雷达图
        num_vars = len(benchmarks)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

        colors = plt.cm.Set2(np.linspace(0, 1, len(data)))
        for i, (mn, scores) in enumerate(data.items()):
            values = scores + scores[:1]
            ax.plot(angles, values, "o-", linewidth=2, label=mn, color=colors[i])
            ax.fill(angles, values, alpha=0.1, color=colors[i])

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(benchmarks, fontsize=9)
        ax.set_title("Model Capability Radar", fontsize=14, fontweight="bold", y=1.08)
        ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=8)

        return self._save_fig(fig, filename)

    def plot_score_distribution(
        self,
        scores_by_model: Dict[str, List[float]],
        filename: str = "score_distribution.png",
    ) -> str:
        """绘制分数分布直方图."""
        if not scores_by_model:
            return ""

        fig, axes = plt.subplots(
            1, len(scores_by_model),
            figsize=(5 * len(scores_by_model), 4),
            squeeze=False,
        )

        for i, (model_name, scores) in enumerate(scores_by_model.items()):
            ax = axes[0][i]
            ax.hist(scores, bins=10, edgecolor="black", alpha=0.7, color=f"C{i}")
            ax.set_title(model_name, fontsize=11)
            ax.set_xlabel("Score")
            ax.set_ylabel("Count")

        fig.suptitle("Score Distribution by Model", fontsize=14, fontweight="bold")
        fig.tight_layout()

        return self._save_fig(fig, filename)

    def generate_all_charts(self, leaderboard: Leaderboard) -> Dict[str, str]:
        """生成所有图表并返回文件路径."""
        charts = {}

        bar_path = self.plot_leaderboard_bar(leaderboard)
        if bar_path:
            charts["leaderboard_bar"] = bar_path

        comp_path = self.plot_benchmark_comparison(leaderboard)
        if comp_path:
            charts["benchmark_comparison"] = comp_path

        radar_path = self.plot_radar(leaderboard)
        if radar_path:
            charts["radar_comparison"] = radar_path

        return charts