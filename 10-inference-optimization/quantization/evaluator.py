"""量化精度评估器"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .formats import QuantFormat
from .quantizer import Quantizer, QuantizationConfig


@dataclass
class AccuracyReport:
    """精度评估报告"""
    format: str
    perplexity: float
    accuracy_percentage: float
    token_match_rate: float
    semantic_similarity: float
    details: Dict[str, float] = field(default_factory=dict)


class QuantEvaluator:
    """量化精度评估器 - 模拟评估量化对模型质量的影响"""

    def __init__(self, config: Optional[QuantizationConfig] = None):
        self.quantizer = Quantizer(config or QuantizationConfig())
        self._random = random.Random(42)  # 固定种子，确保结果可复现

    def evaluate(
        self,
        fmt: QuantFormat,
        num_samples: int = 100,
    ) -> AccuracyReport:
        """评估量化格式的精度"""
        quality = self.quantizer.estimate_quality(fmt)
        quality_factor = quality["quality_factor"]

        # 模拟 perplexity 测量
        base_perplexity = 10.0
        # 量化导致 perplexity 上升
        measured_perplexity = self._simulate_perplexity(
            base_perplexity, quality_factor, num_samples
        )

        # 模拟 token 级别匹配率
        token_match_rate = self._simulate_token_match(quality_factor, num_samples)

        # 模拟语义相似度
        semantic_sim = self._simulate_semantic_similarity(quality_factor, num_samples)

        accuracy_pct = quality_factor * 100

        report = AccuracyReport(
            format=fmt.value,
            perplexity=measured_perplexity,
            accuracy_percentage=accuracy_pct,
            token_match_rate=token_match_rate,
            semantic_similarity=semantic_sim,
            details={
                "num_samples": num_samples,
                "base_perplexity": base_perplexity,
                "quality_factor": quality_factor,
                "perplexity_delta": measured_perplexity - base_perplexity,
            },
        )

        return report

    def _simulate_perplexity(
        self, base_ppl: float, factor: float, n: int
    ) -> float:
        """模拟 perplexity 测量（加入采样噪声）"""
        ideal_ppl = base_ppl / factor
        noise = self._random.gauss(0, 0.1)
        return max(1.0, ideal_ppl + noise)

    def _simulate_token_match(self, factor: float, n: int) -> float:
        """模拟 token 精确匹配率"""
        # 精度越高，匹配率越高
        base_rate = 0.5 + factor * 0.5  # range [0.5, 1.0]
        noise = self._random.gauss(0, 0.02)
        return min(1.0, max(0.0, base_rate + noise))

    def _simulate_semantic_similarity(self, factor: float, n: int) -> float:
        """模拟语义相似度"""
        # 语义相似度对量化不太敏感
        base_sim = 0.7 + factor * 0.3  # range [0.7, 1.0]
        noise = self._random.gauss(0, 0.01)
        return min(1.0, max(0.0, base_sim + noise))

    def compare_formats(
        self, formats: Optional[List[QuantFormat]] = None
    ) -> List[AccuracyReport]:
        """对比多个量化格式"""
        if formats is None:
            formats = list(QuantFormat)
        return [self.evaluate(fmt) for fmt in formats]

    def get_recommendation(
        self,
        max_memory_gb: float = 24.0,
        min_accuracy_pct: float = 95.0,
    ) -> Dict:
        """根据约束条件推荐量化格式"""
        candidates = []
        for fmt in QuantFormat:
            mem = self.quantizer.calculate_memory(fmt)
            qual = self.quantizer.estimate_quality(fmt)

            if mem["total_memory_gb"] <= max_memory_gb:
                if qual["quality_percentage"] >= min_accuracy_pct:
                    candidates.append({
                        "format": fmt.value,
                        "memory_gb": mem["total_memory_gb"],
                        "quality_pct": qual["quality_percentage"],
                        "score": qual["quality_percentage"] * (1 - mem["total_memory_gb"] / max_memory_gb),
                    })

        candidates.sort(key=lambda x: x["score"], reverse=True)
        return {
            "recommendation": candidates[0] if candidates else None,
            "all_candidates": candidates,
            "constraints": {
                "max_memory_gb": max_memory_gb,
                "min_accuracy_pct": min_accuracy_pct,
            },
        }
