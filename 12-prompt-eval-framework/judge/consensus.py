"""多 Judge 共识机制。"""

from typing import Any, Dict, List, Optional

from .llm_judge import LLMJudge


class ConsensusJudge:
    """多 Judge 投票一致性评估。"""

    def __init__(
        self,
        num_judges: int = 3,
        dimensions: Optional[List[str]] = None,
        seed: Optional[int] = None,
    ):
        self.num_judges = num_judges
        self.dimensions = dimensions
        self.judges: List[LLMJudge] = []
        base_seed = seed if seed is not None else 42
        for i in range(num_judges):
            judge = LLMJudge(
                dimensions=dimensions,
                judge_name=f"judge_{i+1}",
                seed=base_seed + i,
            )
            self.judges.append(judge)

    def judge_single(
        self,
        question: str,
        prediction: str,
        reference: str,
    ) -> Dict[str, Any]:
        """多 Judge 评估单条样本。"""
        all_results = []
        for judge in self.judges:
            result = judge.judge_single(question, prediction, reference)
            all_results.append(result)

        return self._compute_consensus(all_results)

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

    def _compute_consensus(self, all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """计算多 Judge 共识结果。"""
        consensus: Dict[str, Any] = {
            "question": all_results[0]["question"],
            "prediction": all_results[0]["prediction"],
            "reference": all_results[0]["reference"],
            "num_judges": self.num_judges,
            "dimensions": {},
        }

        # 收集各维度各 Judge 的分数
        dim_scores: Dict[str, List[int]] = {}
        for result in all_results:
            for dim, score_info in result["scores"].items():
                if dim not in dim_scores:
                    dim_scores[dim] = []
                dim_scores[dim].append(score_info["score"])

        # 计算各维度的共识
        for dim, scores in dim_scores.items():
            mean_score = sum(scores) / len(scores)
            # Fleiss' Kappa 简化版：用标准差衡量一致性
            std = self._std(scores)
            agreement = max(0.0, 1.0 - std / 2.0)  # 标准差为0时完全一致

            # 多数投票
            from collections import Counter
            counter = Counter(scores)
            majority_score = counter.most_common(1)[0][0]

            consensus["dimensions"][dim] = {
                "mean_score": round(mean_score, 2),
                "majority_score": majority_score,
                "individual_scores": scores,
                "agreement": round(agreement, 4),
                "judge_count": len(scores),
            }

        # 总体一致性
        all_agreements = [
            d["agreement"] for d in consensus["dimensions"].values()
        ]
        consensus["overall_agreement"] = round(
            sum(all_agreements) / len(all_agreements), 4
        ) if all_agreements else 0.0

        return consensus

    @staticmethod
    def _std(values: List[float]) -> float:
        if len(values) < 2:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        return variance ** 0.5