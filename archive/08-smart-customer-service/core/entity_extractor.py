"""实体提取模块"""

from __future__ import annotations

import re
from typing import Optional


# 实体类型
class EntityType:
    ORDER_NUMBER = "order_number"
    TRACKING_NUMBER = "tracking_number"
    PHONE_NUMBER = "phone_number"
    EMAIL = "email"
    PRODUCT_NAME = "product_name"
    DATE = "date"
    MONEY = "money"
    ADDRESS = "address"


# 正则模式
_ENTITY_PATTERNS: dict[str, list[tuple[str, re.Pattern]]] = {
    EntityType.ORDER_NUMBER: [
        ("order", re.compile(r"(?:订单号?|order\s*(?:no|number|#)?)[：:\s]*([A-Z0-9\-]{6,20})", re.IGNORECASE)),
        ("order_direct", re.compile(r"(ORD-?\d{4,}-?\w{4,})")),
        ("order_cn", re.compile(r"(\d{10,20})")),
    ],
    EntityType.TRACKING_NUMBER: [
        ("tracking", re.compile(r"(?:快递单号?|物流单号?|tracking)[：:\s]*([A-Z0-9]{8,20})", re.IGNORECASE)),
        ("tracking_cn", re.compile(r"(?:SF|YT|JD|ZTO|YD|STO|EMS)\d{10,18}")),
    ],
    EntityType.PHONE_NUMBER: [
        ("phone_cn", re.compile(r"1[3-9]\d{9}")),
        ("phone_intl", re.compile(r"\+?\d{1,3}[-\s]?\(?\d{2,4}\)?[-\s]?\d{4,8}")),
    ],
    EntityType.EMAIL: [
        ("email", re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")),
    ],
    EntityType.MONEY: [
        ("money_cn", re.compile(r"(\d+(?:\.\d{1,2})?)\s*(?:元|块|人民币)")),
        ("money_en", re.compile(r"(?:\$|USD|CNY|RMB)\s*(\d+(?:\.\d{1,2})?)")),
        ("money_simple", re.compile(r"(\d+(?:\.\d{1,2})?)\s*(?:yuan|dollar|bucks)")),
    ],
    EntityType.DATE: [
        ("date_cn", re.compile(r"(\d{4})[年/\-](\d{1,2})[月/\-](\d{1,2})[日号]?")),
        ("date_relative", re.compile(r"(?:今天|昨天|前天|明天|后天|上周|下周|this week|last week|yesterday|tomorrow)")),
    ],
}


class EntityExtractor:
    """实体提取器"""

    def __init__(self) -> None:
        self._llm_func: Optional[callable] = None

    def set_llm_extractor(self, func: callable) -> None:
        """设置可选的 LLM 实体提取回调"""
        self._llm_func = func

    def extract(self, text: str) -> dict[str, list[str]]:
        """从文本中提取实体"""
        entities: dict[str, list[str]] = {}

        for entity_type, patterns in _ENTITY_PATTERNS.items():
            values = []
            for _name, pattern in patterns:
                matches = pattern.findall(text)
                if isinstance(matches[0], tuple) if matches else False:
                    values.extend("".join(m) for m in matches)
                else:
                    values.extend(matches)
            if values:
                entities[entity_type] = list(set(values))

        # LLM 增强
        if self._llm_func:
            try:
                llm_entities = self._llm_func(text)
                if llm_entities:
                    for etype, vals in llm_entities.items():
                        if etype not in entities:
                            entities[etype] = vals
            except Exception:
                pass

        return entities

    def extract_single(self, text: str, entity_type: str) -> Optional[str]:
        """提取单个实体类型的第一个值"""
        entities = self.extract(text)
        values = entities.get(entity_type, [])
        return values[0] if values else None
