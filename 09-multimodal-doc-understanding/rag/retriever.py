"""检索器 - 语义+关键词混合检索。"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from rag.document_indexer import DocumentChunk, SimpleVectorStore

logger = logging.getLogger(__name__)


class HybridRetriever:
    """混合检索器。

    支持:
    - 语义检索 (TF-IDF)
    - 关键词检索 (精确匹配)
    - 混合排序 (加权融合)
    """

    def __init__(
        self,
        vector_store: SimpleVectorStore,
        semantic_weight: float = 0.7,
        keyword_weight: float = 0.3,
    ):
        """初始化检索器。

        Args:
            vector_store: 向量存储。
            semantic_weight: 语义检索权重。
            keyword_weight: 关键词检索权重。
        """
        self.store = vector_store
        self.semantic_weight = semantic_weight
        self.keyword_weight = keyword_weight

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        document_ids: Optional[List[str]] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """混合检索。

        Args:
            query: 查询文本。
            top_k: 返回前 k 个结果。
            document_ids: 限定文档 ID 列表。

        Returns:
            List[Tuple[DocumentChunk, float]]: (分块, 分数) 列表。
        """
        # 语义检索
        semantic_results = self.store.search(query, top_k=top_k * 2)

        # 关键词检索
        keyword_results = self._keyword_search(query, top_k=top_k * 2)

        # 融合排序
        merged = self._merge_results(semantic_results, keyword_results)

        # 限制文档范围
        if document_ids:
            merged = [
                (idx, score) for idx, score in merged
                if self.store.chunks[idx].document_id in document_ids
            ]

        # 返回 top_k
        results = []
        for idx, score in merged[:top_k]:
            results.append((self.store.chunks[idx], score))

        return results

    def _keyword_search(
        self, query: str, top_k: int = 10
    ) -> List[Tuple[int, float]]:
        """关键词检索 (BM25 简化版)。"""
        query_terms = set(re.findall(r"\S+", query.lower()))

        scores = []
        for i, chunk in enumerate(self.store.chunks):
            chunk_terms = set(re.findall(r"\S+", chunk.text.lower()))
            # 词项重叠度
            overlap = len(query_terms & chunk_terms)
            if overlap == 0:
                continue
            # 精确短语匹配加分
            phrase_bonus = 1.0
            for term in query_terms:
                if term in chunk.text.lower():
                    phrase_bonus += 0.5
            score = overlap * phrase_bonus
            scores.append((i, score))

        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    def _merge_results(
        self,
        semantic_results: List[Tuple[int, float]],
        keyword_results: List[Tuple[int, float]],
    ) -> List[Tuple[int, float]]:
        """融合两种检索结果。

        使用加权倒数排名融合 (Weighted RRF)。
        """
        k = 60  # RRF 常数
        score_map: Dict[int, float] = {}

        for rank, (idx, _) in enumerate(semantic_results):
            score_map[idx] = score_map.get(idx, 0) + self.semantic_weight / (k + rank + 1)

        for rank, (idx, _) in enumerate(keyword_results):
            score_map[idx] = score_map.get(idx, 0) + self.keyword_weight / (k + rank + 1)

        merged = sorted(score_map.items(), key=lambda x: -x[1])
        return merged