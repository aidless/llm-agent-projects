"""评估流水线，支持批量评估、并发控制和报告生成。"""

import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional

from metrics import MetricRegistry


class EvaluationPipeline:
    """评估流水线。"""

    def __init__(
        self,
        metric_registry: Optional[MetricRegistry] = None,
        max_workers: int = 4,
    ):
        self.registry = metric_registry or MetricRegistry()
        self.max_workers = max_workers

    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
        metric_names: Optional[List[str]] = None,
        questions: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """执行评估。

        Args:
            predictions: 预测结果列表
            references: 参考答案列表
            metric_names: 指标名称列表
            questions: 问题列表（可选，用于 LLM Judge）

        Returns:
            评估结果字典
        """
        start_time = time.time()

        if len(predictions) != len(references):
            raise ValueError("predictions 和 references 长度必须相同")

        metrics = self.registry.get_metrics(metric_names)

        # 并发计算各指标
        all_results: Dict[str, Dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_metric = {
                executor.submit(m.compute, predictions, references): m
                for m in metrics
            }
            for future in as_completed(future_to_metric):
                metric = future_to_metric[future]
                try:
                    result = future.result()
                    all_results[metric.name] = result
                except Exception as e:
                    all_results[metric.name] = {"error": str(e)}

        elapsed = time.time() - start_time

        return {
            "num_samples": len(predictions),
            "metrics_used": [m.name for m in metrics],
            "results": all_results,
            "elapsed_seconds": round(elapsed, 3),
        }

    def generate_report(
        self,
        evaluation_result: Dict[str, Any],
        format: str = "json",
    ) -> str:
        """生成评估报告。

        Args:
            evaluation_result: 评估结果
            format: 报告格式 (json/markdown)

        Returns:
            报告字符串
        """
        if format == "json":
            return json.dumps(evaluation_result, ensure_ascii=False, indent=2)
        elif format == "markdown":
            return self._generate_markdown_report(evaluation_result)
        else:
            raise ValueError(f"不支持的格式: {format}")

    def _generate_markdown_report(self, result: Dict[str, Any]) -> str:
        """生成 Markdown 格式报告。"""
        lines = [
            "# Prompt 评估报告",
            "",
            f"## 概览",
            "",
            f"- **样本数量**: {result['num_samples']}",
            f"- **使用指标**: {', '.join(result['metrics_used'])}",
            f"- **耗时**: {result['elapsed_seconds']}s",
            "",
            "## 指标结果",
            "",
        ]

        for metric_name, scores in result["results"].items():
            if "error" in scores:
                lines.append(f"### {metric_name}")
                lines.append(f"**错误**: {scores['error']}")
                lines.append("")
                continue

            lines.append(f"### {metric_name}")
            lines.append("")
            lines.append("| 指标 | 分数 |")
            lines.append("|------|------|")
            for key, value in scores.items():
                if isinstance(value, (int, float)):
                    lines.append(f"| {key} | {value} |")
            lines.append("")

        return "\n".join(lines)