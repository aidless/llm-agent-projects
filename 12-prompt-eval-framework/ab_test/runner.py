"""A/B 测试运行器。"""

import time
from typing import Any, Dict, List, Optional

from metrics import MetricRegistry
from .statistics import paired_t_test, bootstrap_confidence_interval, compute_win_rate


class PromptVersion:
    """Prompt 版本。"""

    def __init__(
        self,
        version_id: str,
        name: str,
        prompt_template: str,
        description: str = "",
    ):
        self.version_id = version_id
        self.name = name
        self.prompt_template = prompt_template
        self.description = description
        self.created_at = time.time()


class ABTestRunner:
    """A/B 测试运行器，支持多版本并行评估。"""

    def __init__(self, metric_registry: Optional[MetricRegistry] = None):
        self.registry = metric_registry or MetricRegistry()
        self.versions: Dict[str, PromptVersion] = {}

    def add_version(self, version: PromptVersion):
        """添加一个 Prompt 版本。"""
        self.versions[version.version_id] = version

    def remove_version(self, version_id: str):
        """移除一个 Prompt 版本。"""
        if version_id not in self.versions:
            raise ValueError(f"版本 {version_id} 不存在")
        del self.versions[version_id]

    def list_versions(self) -> List[Dict[str, Any]]:
        """列出所有版本。"""
        return [
            {
                "version_id": v.version_id,
                "name": v.name,
                "description": v.description,
                "prompt_template": v.prompt_template[:100] + "..." if len(v.prompt_template) > 100 else v.prompt_template,
            }
            for v in self.versions.values()
        ]

    def run_test(
        self,
        version_a_id: str,
        version_b_id: str,
        predictions_a: List[str],
        predictions_b: List[str],
        references: List[str],
        metric_names: Optional[List[str]] = None,
        alpha: float = 0.05,
    ) -> Dict[str, Any]:
        """运行 A/B 测试。

        Args:
            version_a_id: 版本 A ID
            version_b_id: 版本 B ID
            predictions_a: 版本 A 的预测结果
            predictions_b: 版本 B 的预测结果
            references: 参考答案
            metric_names: 要计算的指标名称列表
            alpha: 显著性水平

        Returns:
            A/B 测试结果
        """
        if version_a_id not in self.versions:
            raise ValueError(f"版本 A {version_a_id} 不存在")
        if version_b_id not in self.versions:
            raise ValueError(f"版本 B {version_b_id} 不存在")

        metrics = self.registry.get_metrics(metric_names)

        # 计算各版本各指标分数
        version_a_scores: Dict[str, List[float]] = {}
        version_b_scores: Dict[str, List[float]] = {}

        for metric in metrics:
            result_a = metric.compute(predictions_a, references)
            result_b = metric.compute(predictions_b, references)

            # 取每个指标的第一个分值作为代表分数
            for key in result_a:
                if key not in version_a_scores:
                    version_a_scores[key] = []
                    version_b_scores[key] = []
                version_a_scores[key].append(result_a[key])
                version_b_scores[key].append(result_b[key])

        # 统计检验
        comparison = {}
        for score_key in version_a_scores:
            if len(version_a_scores[score_key]) >= 2:
                # 对多指标取均值得到每个样本的综合分
                a_per_sample = self._per_sample_scores(predictions_a, references, metric_names)
                b_per_sample = self._per_sample_scores(predictions_b, references, metric_names)
                break
        else:
            a_per_sample = self._per_sample_scores(predictions_a, references, metric_names)
            b_per_sample = self._per_sample_scores(predictions_b, references, metric_names)

        t_test_result = paired_t_test(a_per_sample, b_per_sample, alpha)
        win_rate_result = compute_win_rate(a_per_sample, b_per_sample)
        ci_a = bootstrap_confidence_interval(a_per_sample)
        ci_b = bootstrap_confidence_interval(b_per_sample)

        return {
            "version_a": version_a_id,
            "version_b": version_b_id,
            "version_a_name": self.versions[version_a_id].name,
            "version_b_name": self.versions[version_b_id].name,
            "num_samples": len(references),
            "metrics": metric_names or [m.name for m in metrics],
            "version_a_scores": version_a_scores,
            "version_b_scores": version_b_scores,
            "statistical_test": t_test_result,
            "win_rate": win_rate_result,
            "confidence_interval_a": ci_a,
            "confidence_interval_b": ci_b,
            "recommendation": (
                f"版本 A ({self.versions[version_a_id].name}) 表现更好"
                if win_rate_result["win_rate_a"] > win_rate_result["win_rate_b"]
                else f"版本 B ({self.versions[version_b_id].name}) 表现更好"
                if win_rate_result["win_rate_b"] > win_rate_result["win_rate_a"]
                else "两个版本表现持平"
            ),
        }

    def _per_sample_scores(
        self,
        predictions: List[str],
        references: List[str],
        metric_names: Optional[List[str]] = None,
    ) -> List[float]:
        """计算每个样本的综合分数（取各指标的均值）。"""
        metrics = self.registry.get_metrics(metric_names)
        n = len(predictions)
        sample_scores: List[List[float]] = [[] for _ in range(n)]

        for metric in metrics:
            for i in range(n):
                result = metric.compute_single(predictions[i], references[i])
                # 取第一个数值
                for v in result.values():
                    if isinstance(v, (int, float)) and v <= 1.0:
                        sample_scores[i].append(v)
                        break

        return [
            sum(scores) / len(scores) if scores else 0.0
            for scores in sample_scores
        ]