"""统计检验工具。"""

from typing import Dict, List, Tuple

import numpy as np
from scipy import stats


def paired_t_test(scores_a: List[float], scores_b: List[float], alpha: float = 0.05) -> Dict:
    """配对 t 检验。

    Args:
        scores_a: 版本 A 的分数列表
        scores_b: 版本 B 的分数列表
        alpha: 显著性水平

    Returns:
        包含 t 统计量、p 值、是否显著的字典
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("两个版本分数列表长度必须相同")
    if len(scores_a) < 2:
        raise ValueError("样本数至少为 2")

    t_stat, p_value = stats.ttest_rel(scores_a, scores_b)
    significant = bool(p_value < alpha)
    mean_diff = float(np.mean(np.array(scores_a) - np.array(scores_b)))

    return {
        "test": "paired_t_test",
        "t_statistic": round(float(t_stat), 4),
        "p_value": round(float(p_value), 6),
        "alpha": alpha,
        "significant": significant,
        "mean_difference": round(mean_diff, 4),
        "direction": "A > B" if mean_diff > 0 else "B > A" if mean_diff < 0 else "A == B",
    }


def bootstrap_confidence_interval(
    scores: List[float],
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
    seed: int = 42,
) -> Dict:
    """Bootstrap 置信区间。

    Args:
        scores: 分数列表
        n_bootstrap: bootstrap 采样次数
        alpha: 显著性水平
        seed: 随机种子

    Returns:
        包含均值置信区间的字典
    """
    rng = np.random.RandomState(seed)
    scores_arr = np.array(scores)
    n = len(scores_arr)

    bootstrap_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(scores_arr, size=n, replace=True)
        bootstrap_means.append(float(np.mean(sample)))

    lower = float(np.percentile(bootstrap_means, 100 * alpha / 2))
    upper = float(np.percentile(bootstrap_means, 100 * (1 - alpha / 2)))

    return {
        "test": "bootstrap_ci",
        "mean": round(float(np.mean(scores_arr)), 4),
        "std": round(float(np.std(scores_arr)), 4),
        "ci_lower": round(lower, 4),
        "ci_upper": round(upper, 4),
        "confidence_level": round(1 - alpha, 2),
        "n_bootstrap": n_bootstrap,
    }


def compute_win_rate(scores_a: List[float], scores_b: List[float]) -> Dict:
    """计算 A/B 胜率。

    Args:
        scores_a: 版本 A 分数列表
        scores_b: 版本 B 分数列表

    Returns:
        胜率统计字典
    """
    if len(scores_a) != len(scores_b):
        raise ValueError("两个版本分数列表长度必须相同")

    wins_a = 0
    wins_b = 0
    ties = 0

    for a, b in zip(scores_a, scores_b):
        if a > b:
            wins_a += 1
        elif a < b:
            wins_b += 1
        else:
            ties += 1

    total = len(scores_a)
    return {
        "win_rate_a": round(wins_a / total, 4) if total else 0.0,
        "win_rate_b": round(wins_b / total, 4) if total else 0.0,
        "tie_rate": round(ties / total, 4) if total else 0.0,
        "wins_a": wins_a,
        "wins_b": wins_b,
        "ties": ties,
        "total": total,
    }
