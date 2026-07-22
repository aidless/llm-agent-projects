from typing import Any, Dict, List

from .base import BaseMetric


class F1ScoreMetric(BaseMetric):
    """Token-level F1 Score 评估指标。"""

    def __init__(self, ignore_case: bool = True):
        super().__init__(
            name="f1",
            description="Token-level F1 Score"
        )
        self.ignore_case = ignore_case

    def _tokenize(self, text: str) -> set:
        tokens = text.lower().split() if self.ignore_case else text.split()
        return set(tokens)

    def compute(self, predictions: List[str], references: List[str], **kwargs) -> Dict[str, Any]:
        if len(predictions) != len(references):
            raise ValueError("predictions 和 references 长度必须相同")

        f1_scores = []
        precisions = []
        recalls = []

        for pred, ref in zip(predictions, references):
            pred_tokens = self._tokenize(pred)
            ref_tokens = self._tokenize(ref)

            common = pred_tokens & ref_tokens

            if not pred_tokens and not ref_tokens:
                precisions.append(1.0)
                recalls.append(1.0)
                f1_scores.append(1.0)
            elif not pred_tokens or not ref_tokens:
                precisions.append(0.0)
                recalls.append(0.0)
                f1_scores.append(0.0)
            else:
                p = len(common) / len(pred_tokens)
                r = len(common) / len(ref_tokens)
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
                precisions.append(p)
                recalls.append(r)
                f1_scores.append(f1)

        return {
            "f1": round(sum(f1_scores) / len(f1_scores) if f1_scores else 0.0, 4),
            "f1_precision": round(sum(precisions) / len(precisions) if precisions else 0.0, 4),
            "f1_recall": round(sum(recalls) / len(recalls) if recalls else 0.0, 4),
        }