from typing import Any, Dict, List

from rouge_score import rouge_scorer

from .base import BaseMetric


class ROUGEMetric(BaseMetric):
    """ROUGE 评估指标，支持 ROUGE-1/2/L/sum。"""

    def __init__(self, rouge_types: List[str] | None = None, use_stemmer: bool = True):
        self.rouge_types = rouge_types or ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        super().__init__(
            name="rouge",
            description=f"ROUGE scores: {', '.join(self.rouge_types)}"
        )
        self.scorer = rouge_scorer.RougeScorer(
            self.rouge_types, use_stemmer=use_stemmer
        )

    def compute(self, predictions: List[str], references: List[str], **kwargs) -> Dict[str, Any]:
        if len(predictions) != len(references):
            raise ValueError("predictions 和 references 长度必须相同")

        aggregate_scores: Dict[str, Dict[str, float]] = {}

        for pred, ref in zip(predictions, references):
            scores = self.scorer.score(ref, pred)
            for rouge_type in self.rouge_types:
                if rouge_type not in aggregate_scores:
                    aggregate_scores[rouge_type] = {"precision": [], "recall": [], "fmeasure": []}
                aggregate_scores[rouge_type]["precision"].append(scores[rouge_type].precision)
                aggregate_scores[rouge_type]["recall"].append(scores[rouge_type].recall)
                aggregate_scores[rouge_type]["fmeasure"].append(scores[rouge_type].fmeasure)

        results: Dict[str, Any] = {}
        for rouge_type, metrics in aggregate_scores.items():
            for metric_name, values in metrics.items():
                avg = sum(values) / len(values) if values else 0.0
                key = f"{rouge_type}_{metric_name}" if metric_name != "fmeasure" else rouge_type
                if metric_name != "fmeasure":
                    results[key] = round(avg, 4)
                else:
                    results[rouge_type] = round(avg, 4)

        return results