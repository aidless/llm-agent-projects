import re
from typing import Any, Dict, List

from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction

from .base import BaseMetric


class BLEUMetric(BaseMetric):
    """BLEU 评估指标，支持 1-4 gram。"""

    def __init__(self, max_n: int = 4, smoothing: bool = True):
        super().__init__(
            name="bleu",
            description=f"BLEU score with max {max_n}-gram"
        )
        self.max_n = max_n
        self.smoothing = smoothing

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """简单分词，不依赖 nltk punkt 数据包。"""
        text = text.lower()
        # 用正则分词：匹配单词（含连字符和撇号）和标点
        tokens = re.findall(r"\w+(?:[-']\w+)*|[^\w\s]", text)
        return tokens

    def compute(self, predictions: List[str], references: List[str], **kwargs) -> Dict[str, Any]:
        if len(predictions) != len(references):
            raise ValueError("predictions 和 references 长度必须相同")

        weights_options = [
            (1, [1.0]),
            (2, [0.5, 0.5]),
            (3, [0.33, 0.33, 0.34]),
            (4, [0.25, 0.25, 0.25, 0.25]),
        ]

        results: Dict[str, Any] = {}
        tokenized_refs = [
            [self._tokenize(ref)] for ref in references
        ]
        tokenized_preds = [self._tokenize(pred) for pred in predictions]

        for n, weights in weights_options:
            if n > self.max_n:
                continue
            smooth_fn = SmoothingFunction().method1 if self.smoothing else None
            score = corpus_bleu(
                tokenized_refs,
                tokenized_preds,
                weights=weights,
                smoothing_function=smooth_fn,
            )
            results[f"bleu-{n}"] = round(score, 4)

        return results