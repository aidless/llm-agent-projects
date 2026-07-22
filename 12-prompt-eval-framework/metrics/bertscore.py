from typing import Any, Dict, List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .base import BaseMetric


class BERTScoreMetric(BaseMetric):
    """BERTScore 模拟实现，使用 TfidfVectorizer + cosine_similarity。"""

    def __init__(self, ngram_range: tuple = (1, 2), analyzer: str = "word"):
        super().__init__(
            name="bertscore",
            description="BERTScore (simulated with TF-IDF + cosine similarity)"
        )
        self.ngram_range = ngram_range
        self.analyzer = analyzer

    def compute(self, predictions: List[str], references: List[str], **kwargs) -> Dict[str, Any]:
        if len(predictions) != len(references):
            raise ValueError("predictions 和 references 长度必须相同")

        all_texts = list(predictions) + list(references)
        vectorizer = TfidfVectorizer(
            ngram_range=self.ngram_range,
            analyzer=self.analyzer,
            stop_words="english",
        )
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        n = len(predictions)

        pred_vectors = tfidf_matrix[:n]
        ref_vectors = tfidf_matrix[n:]

        similarities = np.array([
            cosine_similarity(pred_vectors[i], ref_vectors[i])[0][0]
            for i in range(n)
        ])

        # 模拟 Precision/Recall/F1：以余弦相似度作为 F1 基础，
        # 用简单的 token overlap 模拟 precision 和 recall
        precisions = []
        recalls = []

        for pred, ref in zip(predictions, references):
            pred_tokens = set(pred.lower().split())
            ref_tokens = set(ref.lower().split())
            if not pred_tokens or not ref_tokens:
                precisions.append(0.0)
                recalls.append(0.0)
                continue
            common = pred_tokens & ref_tokens
            p = len(common) / len(pred_tokens)
            r = len(common) / len(ref_tokens)
            precisions.append(p)
            recalls.append(r)

        f1_scores = 2 * np.array(precisions) * np.array(recalls) / (
            np.array(precisions) + np.array(recalls) + 1e-8
        )

        return {
            "bertscore_precision": round(float(np.mean(precisions)), 4),
            "bertscore_recall": round(float(np.mean(recalls)), 4),
            "bertscore_f1": round(float(np.mean(f1_scores)), 4),
            "bertscore_cosine": round(float(np.mean(similarities)), 4),
        }