"""数据导出器 - 导出 Trace 数据用于微调和分析。"""

import json
import time
from typing import Any, Dict, List, Optional

from storage.memory_store import get_store


class DataExporter:
    """导出 Trace/Feedback 数据，支持多种格式。

    用法::

        exporter = DataExporter()
        # 导出为 JSONL (用于微调)
        dataset = exporter.export_for_finetuning()
        # 导出为 JSON
        traces = exporter.export_traces_json()
    """

    def __init__(self) -> None:
        self._store = get_store()

    def export_traces_json(
        self,
        span_type: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """导出 Trace 列表为 JSON。"""
        return self._store.list_traces(
            limit=limit,
            offset=offset,
            span_type=span_type,
        )

    def export_for_finetuning(
        self,
        span_type: str = "llm",
        min_rating: Optional[float] = None,
    ) -> List[Dict[str, str]]:
        """导出用于微调的数据集 (JSONL 格式)。

        只导出 LLM 类型的 Span，每条记录包含:
        - instruction: 用户输入 / prompt
        - output: LLM 输出
        - (可选) rating 来自关联的反馈

        Args:
            span_type: 要导出的 Span 类型
            min_rating: 最低评分门槛 (仅包含评分 >= 此值的记录)

        Returns:
            [{"instruction": ..., "output": ...}, ...]
        """
        traces = self._store.list_traces(limit=10000, span_type=span_type)
        dataset: List[Dict[str, str]] = []

        for trace in traces:
            for span in trace.get("spans", []):
                if span.get("span_type") != span_type:
                    continue
                if not span.get("llm_input") and not span.get("llm_output"):
                    continue

                # 检查评分门槛
                if min_rating is not None:
                    fbs = self._store.list_feedbacks(trace_id=trace["trace_id"])
                    ratings = [
                        f["value"] for f in fbs
                        if f.get("feedback_type") == "rating" and f.get("value") is not None
                    ]
                    if ratings and min(ratings) < min_rating:
                        continue

                instruction = ""
                output = ""

                inp = span.get("llm_input")
                if isinstance(inp, str):
                    instruction = inp
                elif isinstance(inp, list):
                    # 尝试从 messages 格式提取
                    for msg in inp:
                        if isinstance(msg, dict) and msg.get("role") == "user":
                            instruction = msg.get("content", "")
                            break
                    if not instruction:
                        instruction = json.dumps(inp, ensure_ascii=False)
                elif inp is not None:
                    instruction = json.dumps(inp, ensure_ascii=False)

                out = span.get("llm_output")
                if isinstance(out, str):
                    output = out
                elif out is not None:
                    output = json.dumps(out, ensure_ascii=False)

                if instruction and output:
                    dataset.append({
                        "instruction": instruction,
                        "output": output,
                    })

        return dataset

    def export_feedbacks_json(
        self,
        trace_id: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """导出反馈数据。"""
        return self._store.list_feedbacks(trace_id=trace_id, limit=limit)

    def export_to_jsonl(self, data: List[Dict[str, Any]]) -> str:
        """将数据列表转为 JSONL 字符串。"""
        return "\n".join(json.dumps(item, ensure_ascii=False, default=str) for item in data)

    def export_metrics_summary(self) -> Dict[str, Any]:
        """导出指标摘要。"""
        names = self._store.list_metric_names()
        summary: Dict[str, Any] = {}
        for name in names:
            points = self._store.get_metrics(name)
            if not points:
                continue
            values = [p["value"] for p in points]
            summary[name] = {
                "count": len(values),
                "sum": sum(values),
                "avg": sum(values) / len(values),
                "min": min(values),
                "max": max(values),
                "latest": values[-1],
            }
        return summary