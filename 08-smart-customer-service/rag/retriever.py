"""检索器模块 - 语义检索 + 关键词检索混合"""

from __future__ import annotations

import math
import re
from typing import Optional

from app.models import KnowledgeSearchResult
from rag.knowledge_base import KnowledgeBase


def _simple_tokenize(text: str) -> list[str]:
    """简单分词（按字符和空格）"""
    # 对中文按单字，对英文按词
    tokens: list[str] = []
    current_english: list[str] = []
    for char in text.lower():
        if char.isascii() and char.isalpha():
            current_english.append(char)
        else:
            if current_english:
                tokens.append("".join(current_english))
                current_english = []
            if char.strip():
                tokens.append(char)
    if current_english:
        tokens.append("".join(current_english))
    return tokens


def _compute_similarity(text1: str, text2: str) -> float:
    """计算两段文本的简单相似度（基于字符和词的重叠）"""
    tokens1 = set(_simple_tokenize(text1))
    tokens2 = set(_simple_tokenize(text2))

    if not tokens1 or not tokens2:
        return 0.0

    intersection = tokens1 & tokens2
    union = tokens1 | tokens2

    jaccard = len(intersection) / len(union) if union else 0.0

    # 额外的子串匹配加分
    substring_score = 0.0
    for t1 in tokens1:
        if any(t1 in t2 or t2 in t1 for t2 in tokens2):
            substring_score += 0.1

    return min(jaccard + substring_score * 0.3, 1.0)


class Retriever:
    """混合检索器"""

    def __init__(self, knowledge_base: KnowledgeBase) -> None:
        self._kb = knowledge_base
        self._llm_retriever_func: Optional[callable] = None

    def set_llm_retriever(self, func: callable) -> None:
        """设置可选的 LLM 检索回调"""
        self._llm_retriever_func = func

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        category: Optional[str] = None,
    ) -> list[KnowledgeSearchResult]:
        """混合检索：关键词 + 语义"""
        # 1. 关键词检索
        keyword_results = self._kb.search_by_keyword(query, top_k=top_k * 2)

        # 2. 语义检索（简单文本相似度）
        semantic_results = self._semantic_search(query, top_k=top_k * 2, category=category)

        # 3. 合并去重，加权融合
        merged: dict[str, dict[str, float]] = {}
        for item, score in keyword_results:
            if item.id not in merged:
                merged[item.id] = {"keyword_score": 0.0, "semantic_score": 0.0}
            merged[item.id]["keyword_score"] = score
            merged[item.id]["item"] = item  # type: ignore[assignment]

        for item, score in semantic_results:
            if item.id not in merged:
                merged[item.id] = {"keyword_score": 0.0, "semantic_score": 0.0}
            merged[item.id]["semantic_score"] = score
            if "item" not in merged[item.id]:
                merged[item.id]["item"] = item  # type: ignore[assignment]

        # 加权融合分数
        final_results: list[tuple[object, float]] = []
        for entry in merged.values():
            item = entry["item"]
            kw_score = entry.get("keyword_score", 0.0)
            sem_score = entry.get("semantic_score", 0.0)
            # 归一化
            combined = 0.6 * min(kw_score / 5.0, 1.0) + 0.4 * sem_score
            final_results.append((item, combined))

        # 排序
        final_results.sort(key=lambda x: x[1], reverse=True)

        # LLM 增强（可选）
        if self._llm_retriever_func and final_results:
            try:
                llm_boosted = self._llm_retriever_func(query, [r[0] for r in final_results])
                if llm_boosted:
                    final_results = llm_boosted
            except Exception:
                pass

        # 返回 top_k 结果
        results: list[KnowledgeSearchResult] = []
        for item, score in final_results[:top_k]:
            results.append(KnowledgeSearchResult(
                id=item.id,
                question=item.question,
                answer=item.answer,
                category=item.category,
                score=round(score, 4),
            ))

        return results

    def _semantic_search(
        self,
        query: str,
        top_k: int = 5,
        category: Optional[str] = None,
    ) -> list[tuple[object, float]]:
        """语义搜索（基于文本相似度）"""
        results: list[tuple[object, float]] = []
        query_lower = query.lower()

        items = self._kb.get_all_items()
        if category:
            items = [item for item in items if item.category == category]

        for item in items:
            # 问题相似度
            question_sim = _compute_similarity(query_lower, item.question.lower())
            # 答案相似度
            answer_sim = _compute_similarity(query_lower, item.answer.lower()) * 0.5

            score = max(question_sim, answer_sim)

            if score > 0.05:
                results.append((item, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]
