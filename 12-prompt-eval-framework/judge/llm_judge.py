"""LLM-as-Judge 实现，使用 mock 评估结果。"""

import random
from typing import Any, Dict, List, Optional

from .templates import JUDGE_TEMPLATES, get_template


class LLMJudge:
    """LLM-as-Judge 评估器（Mock 实现）。

    支持多维度评估：准确性/相关性/完整性/连贯性/安全性/综合
    评分量表：1-5 分
    """

    VALID_DIMENSIONS = list(JUDGE_TEMPLATES.keys())

    def __init__(
        self,
        dimensions: Optional[List[str]] = None,
        judge_name: str = "default_judge",
        seed: Optional[int] = None,
    ):
        if dimensions is None:
            dimensions = ["accuracy", "relevance", "completeness", "coherence", "safety", "overall"]
        for d in dimensions:
            if d not in self.VALID_DIMENSIONS:
                raise ValueError(f"无效维度: {d}，有效维度: {self.VALID_DIMENSIONS}")
        self.dimensions = dimensions
        self.judge_name = judge_name
        self._rng = random.Random(seed)

    def judge_single(
        self,
        question: str,
        prediction: str,
        reference: str,
    ) -> Dict[str, Any]:
        """对单条样本进行 LLM-as-Judge 评估（Mock）。"""
        templates = get_template(self.dimensions)
        scores = {}
        prompts = {}

        for dim_name, dim_info in templates.items():
            prompt = dim_info["template"].format(
                question=question,
                reference=reference,
                prediction=prediction,
            )
            prompts[dim_name] = prompt
            # Mock: 基于文本重叠度生成一个合理的随机分数
            score = self._mock_score(prediction, reference)
            scores[dim_name] = {
                "score": score,
                "dimension": dim_name,
                "dimension_name": dim_info["name"],
                "judge": self.judge_name,
            }

        return {
            "question": question,
            "prediction": prediction,
            "reference": reference,
            "scores": scores,
            "prompts": prompts,
            "judge": self.judge_name,
        }

    def judge_batch(
        self,
        questions: List[str],
        predictions: List[str],
        references: List[str],
    ) -> List[Dict[str, Any]]:
        """批量评估。"""
        if not (len(questions) == len(predictions) == len(references)):
            raise ValueError("questions, predictions, references 长度必须相同")
        return [
            self.judge_single(q, p, r)
            for q, p, r in zip(questions, predictions, references)
        ]

    def aggregate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """聚合多次评估结果。"""
        dim_scores: Dict[str, List[int]] = {}
        for result in results:
            for dim, score_info in result["scores"].items():
                if dim not in dim_scores:
                    dim_scores[dim] = []
                dim_scores[dim].append(score_info["score"])

        agg = {}
        for dim, scores in dim_scores.items():
            agg[dim] = {
                "mean": round(sum(scores) / len(scores), 2),
                "min": min(scores),
                "max": max(scores),
                "std": round(self._std(scores), 2),
                "count": len(scores),
            }

        return agg

    def _mock_score(self, prediction: str, reference: str) -> int:
        """根据文本重叠生成 mock 分数 1-5。"""
        pred_tokens = set(prediction.lower().split())
        ref_tokens = set(reference.lower().split())
        if not pred_tokens or not ref_tokens:
            return self._rng.randint(2, 3)
        overlap = len(pred_tokens & ref_tokens) / len(ref_tokens)
        if overlap > 0.8:
            base = 5
        elif overlap > 0.6:
            base = 4
        elif overlap > 0.4:
            base = 3
        elif overlap > 0.2:
            base = 2
        else:
            base = 1
        # 加一点随机性
        score = base + self._rng.choice([-1, 0, 0, 0, 1])
        return max(1, min(5, score))

    @staticmethod
    def _std(values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return variance ** 0.5
