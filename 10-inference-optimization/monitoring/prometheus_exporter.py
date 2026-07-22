"""Prometheus 指标导出器"""

from __future__ import annotations

from typing import Dict, List, Optional

from .metrics import MetricsCollector, Metric, MetricType


class PrometheusExporter:
    """将指标导出为 Prometheus text format"""

    def __init__(self, collector: MetricsCollector):
        self.collector = collector

    def generate(self) -> str:
        """生成 Prometheus text format 输出"""
        lines: List[str] = []
        all_metrics = self.collector.get_all_metrics()

        for name, data in sorted(all_metrics.items()):
            metric_type = data["type"]
            # TYPE 注释
            lines.append(f"# TYPE {name} {metric_type}")

            if metric_type == MetricType.COUNTER.value:
                lines.append(f"{name} {data['value']}")
            elif metric_type == MetricType.GAUGE.value:
                lines.append(f"{name} {data['value']}")
            elif metric_type == MetricType.HISTOGRAM.value:
                histogram = data.get("histogram", {})
                # _count
                lines.append(f"{name}_count {histogram.get('count', 0)}")
                # _sum
                lines.append(f"{name}_sum {histogram.get('sum', 0.0)}")
                # _bucket
                for bucket in histogram.get("buckets", []):
                    le = bucket["le"]
                    count = bucket["count"]
                    lines.append(f'{name}_bucket{{le="{le}"}} {count}')
                # +Inf bucket
                total_count = histogram.get("count", 0)
                lines.append(f'{name}_bucket{{le="+Inf"}} {total_count}')

            lines.append("")  # 空行

        return "\n".join(lines)

    def generate_metrics_for_scrape(self) -> str:
        """生成用于 Prometheus 抓取的完整输出"""
        lines: List[str] = []
        lines.append("# LLM Inference Optimization - Prometheus Metrics")
        lines.append("")

        # 汇总指标
        summary = self.collector.get_summary()
        lines.append("# Summary metrics")
        lines.append(f'llm_total_requests {summary["total_requests"]}')
        lines.append(f'llm_total_tokens_generated {summary["total_tokens_generated"]}')
        lines.append(f'llm_avg_ttft_ms {summary["avg_ttft_ms"]}')
        lines.append(f'llm_p95_ttft_ms {summary["p95_ttft_ms"]}')
        lines.append(f'llm_avg_e2e_latency_ms {summary["avg_e2e_latency_ms"]}')
        lines.append(f'llm_p95_e2e_latency_ms {summary["p95_e2e_latency_ms"]}')
        lines.append(f'llm_current_tps {summary["current_tps"]}')
        lines.append(f'llm_gpu_utilization {summary["gpu_utilization"]}')
        lines.append(f'llm_gpu_memory_gb {summary["gpu_memory_gb"]}')
        lines.append(f'llm_kv_cache_utilization {summary["kv_cache_utilization"]}')
        lines.append("")

        # 所有详细指标
        lines.append("# Detailed metrics")
        lines.append(self.generate())

        return "\n".join(lines)

    def generate_help_text(self) -> Dict[str, str]:
        """生成各指标的帮助文本"""
        metrics = self.collector.get_all_metrics()
        help_texts = {}
        for name, data in metrics.items():
            help_texts[name] = {
                "type": data["type"],
                "current_value": data["value"],
            }
        return help_texts
