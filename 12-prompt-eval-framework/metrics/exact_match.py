from typing import Any, Dict, List

from .base import BaseMetric


class ExactMatchMetric(BaseMetric):
    """精确匹配评估指标。"""

    def __init__(self, ignore_case: bool = True, ignore_punctuation: bool = True):
        super().__init__(
            name="exact_match",
            description="Exact Match score"
        )
        self.ignore_case = ignore_case
        self.ignore_punctuation = ignore_punctuation

    def _normalize(self, text: str) -> str:
        if self.ignore_case:
            text = text.lower()
        if self.ignore_punctuation:
            for ch in '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~':
                text = text.replace(ch, '')
            text = text.strip()
        return text

    def compute(self, predictions: List[str], references: List[str], **kwargs) -> Dict[str, Any]:
        if len(predictions) != len(references):
            raise ValueError("predictions 和 references 长度必须相同")

        matches = 0
        for pred, ref in zip(predictions, references):
            if self._normalize(pred) == self._normalize(ref):
                matches += 1

        score = matches / len(predictions) if predictions else 0.0
        return {
            "exact_match": round(score, 4),
            "exact_match_count": matches,
            "total_count": len(predictions),
        }
