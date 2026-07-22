"""知识库模块"""

from __future__ import annotations

import json
import os
from datetime import datetime
from typing import Optional

from app.models import FAQItem, KnowledgeBaseStats


class KnowledgeBase:
    """FAQ 知识库管理"""

    def __init__(self, data_path: Optional[str] = None) -> None:
        self._items: dict[str, FAQItem] = {}
        self._data_path = data_path
        self._last_updated: Optional[datetime] = None

        if data_path and os.path.exists(data_path):
            self._load_from_file(data_path)

    def _load_from_file(self, path: str) -> None:
        """从 JSON 文件加载 FAQ 数据"""
        with open(path, "r", encoding="utf-8") as f:
            items = json.load(f)
        for item_data in items:
            faq = FAQItem(**item_data)
            self._items[faq.id] = faq
        self._last_updated = datetime.utcnow()

    def add_item(
        self,
        question: str,
        answer: str,
        category: str,
        sub_category: Optional[str] = None,
        keywords: Optional[list[str]] = None,
    ) -> FAQItem:
        """添加 FAQ 条目"""
        item_id = f"faq_{len(self._items) + 1:03d}"
        faq = FAQItem(
            id=item_id,
            question=question,
            answer=answer,
            category=category,
            sub_category=sub_category,
            keywords=keywords or [],
        )
        self._items[faq.id] = faq
        self._last_updated = datetime.utcnow()
        return faq

    def remove_item(self, item_id: str) -> bool:
        """删除 FAQ 条目"""
        if item_id in self._items:
            del self._items[item_id]
            self._last_updated = datetime.utcnow()
            return True
        return False

    def get_item(self, item_id: str) -> Optional[FAQItem]:
        """获取 FAQ 条目"""
        return self._items.get(item_id)

    def get_all_items(self) -> list[FAQItem]:
        """获取所有 FAQ 条目"""
        return list(self._items.values())

    def get_by_category(self, category: str) -> list[FAQItem]:
        """按分类获取 FAQ"""
        return [item for item in self._items.values() if item.category == category]

    def search_by_keyword(self, keyword: str, top_k: int = 5) -> list[tuple[FAQItem, float]]:
        """基于关键词搜索"""
        keyword_lower = keyword.lower()
        results: list[tuple[FAQItem, float]] = []

        for item in self._items.values():
            score = 0.0
            # 问题匹配
            if keyword_lower in item.question.lower():
                score += 3.0
            # 关键词匹配
            for kw in item.keywords:
                if keyword_lower in kw.lower() or kw.lower() in keyword_lower:
                    score += 2.0
            # 分类匹配
            if keyword_lower in item.category.lower():
                score += 1.0
            # 子分类匹配
            if item.sub_category and keyword_lower in item.sub_category.lower():
                score += 1.5

            if score > 0:
                results.append((item, score))

        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

    def get_stats(self) -> KnowledgeBaseStats:
        """获取知识库统计"""
        categories: dict[str, int] = {}
        for item in self._items.values():
            categories[item.category] = categories.get(item.category, 0) + 1

        return KnowledgeBaseStats(
            total_count=len(self._items),
            categories=categories,
            last_updated=self._last_updated,
        )

    def count(self) -> int:
        """获取 FAQ 总数"""
        return len(self._items)
