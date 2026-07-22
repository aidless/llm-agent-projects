"""意图识别模块 - 基于 Few-shot 的意图分类"""

from __future__ import annotations

import re
from typing import Optional

from app.models import IntentCategory, IntentResult, SubIntent


# Few-shot 示例
_INTENT_EXAMPLES: dict[IntentCategory, list[tuple[str, SubIntent]]] = {
    IntentCategory.GREETING: [
        ("你好", SubIntent.UNKNOWN),
        ("hi", SubIntent.UNKNOWN),
        ("早上好", SubIntent.UNKNOWN),
        ("hello", SubIntent.UNKNOWN),
        ("在吗", SubIntent.UNKNOWN),
    ],
    IntentCategory.CONSULTATION: [
        ("这个商品多少钱", SubIntent.PRICE_INQUIRY),
        ("有蓝色的吗", SubIntent.STOCK_INQUIRY),
        ("这款产品的参数是什么", SubIntent.PRODUCT_INFO),
        ("What is the price of this item?", SubIntent.PRICE_INQUIRY),
        ("这个手机有什么颜色", SubIntent.PRODUCT_INFO),
        ("还有库存吗", SubIntent.STOCK_INQUIRY),
    ],
    IntentCategory.COMPLAINT: [
        ("你们的产品太差了", SubIntent.QUALITY_ISSUE),
        ("客服态度太差了", SubIntent.SERVICE_ISSUE),
        ("快递怎么还没到", SubIntent.DELIVERY_ISSUE),
        ("The quality is terrible", SubIntent.QUALITY_ISSUE),
        ("你们服务太差了，我要投诉", SubIntent.SERVICE_ISSUE),
        ("包装都破了", SubIntent.QUALITY_ISSUE),
    ],
    IntentCategory.AFTER_SALE: [
        ("我要退货", SubIntent.RETURN_REQUEST),
        ("可以换货吗", SubIntent.EXCHANGE_REQUEST),
        ("退款什么时候到账", SubIntent.REFUND_REQUEST),
        ("I want to return this product", SubIntent.RETURN_REQUEST),
        ("收到有破损，要求退换", SubIntent.EXCHANGE_REQUEST),
        ("怎么申请退款", SubIntent.REFUND_REQUEST),
    ],
    IntentCategory.TRANSFER_HUMAN: [
        ("转人工", SubIntent.UNKNOWN),
        ("我要找人工客服", SubIntent.UNKNOWN),
        ("让人工来处理", SubIntent.UNKNOWN),
        ("transfer to human", SubIntent.UNKNOWN),
        ("我不跟机器人说", SubIntent.UNKNOWN),
        ("你们机器人不行，人工出来", SubIntent.UNKNOWN),
    ],
    IntentCategory.FEEDBACK: [
        ("你们服务不错", SubIntent.POSITIVE_FEEDBACK),
        ("建议增加夜间配送", SubIntent.SUGGESTION),
        ("Very good service", SubIntent.POSITIVE_FEEDBACK),
        ("我觉得可以改善包装", SubIntent.SUGGESTION),
        ("好评", SubIntent.POSITIVE_FEEDBACK),
    ],
    IntentCategory.ORDER_QUERY: [
        ("我的订单到哪了", SubIntent.ORDER_STATUS),
        ("帮我查一下订单状态", SubIntent.ORDER_STATUS),
        ("我要取消订单", SubIntent.ORDER_CANCEL),
        ("修改一下收货地址", SubIntent.ORDER_MODIFY),
        ("What is my order status?", SubIntent.ORDER_STATUS),
        ("帮我取消昨天下的订单", SubIntent.ORDER_CANCEL),
    ],
    IntentCategory.LOGISTICS: [
        ("快递单号多少", SubIntent.TRACKING),
        ("几天能到", SubIntent.DELIVERY_TIME),
        ("运费多少钱", SubIntent.SHIPPING_FEE),
        ("How long does shipping take?", SubIntent.DELIVERY_TIME),
        ("帮我查一下物流信息", SubIntent.TRACKING),
        ("包邮吗", SubIntent.SHIPPING_FEE),
    ],
}

# 关键词映射
_INTENT_KEYWORDS: dict[IntentCategory, list[str]] = {
    IntentCategory.GREETING: ["你好", "早上好", "下午好", "晚上好", "hi", "hello", "hey", "在吗", "哈喽"],
    IntentCategory.CONSULTATION: ["多少钱", "价格", "什么颜色", "参数", "规格", "尺寸", "材质", "库存", "有没有", "现货", "stock", "price", "info"],
    IntentCategory.COMPLAINT: ["投诉", "太差", "不满", "质量差", "破", "烂", "骗", "terrible", "bad", "complaint", "angry"],
    IntentCategory.AFTER_SALE: ["退货", "换货", "退款", "退换", "返修", "return", "refund", "exchange"],
    IntentCategory.TRANSFER_HUMAN: ["转人工", "人工客服", "找人工", "真人", "transfer", "human agent"],
    IntentCategory.FEEDBACK: ["建议", "反馈", "好评", "不错", "很好", "建议增加", "suggestion", "feedback", "good"],
    IntentCategory.ORDER_QUERY: ["订单", "查单", "取消订单", "修改订单", "order", "cancel order"],
    IntentCategory.LOGISTICS: ["快递", "物流", "运费", "几天到", "配送", "shipping", "delivery", "tracking"],
}

_SUB_INTENT_KEYWORDS: dict[SubIntent, list[str]] = {
    SubIntent.PRICE_INQUIRY: ["多少钱", "价格", "贵不贵", "优惠", "打折", "price", "cost"],
    SubIntent.STOCK_INQUIRY: ["库存", "有没有", "现货", "缺货", "stock", "available"],
    SubIntent.PRODUCT_INFO: ["参数", "规格", "尺寸", "材质", "功能", "介绍", "detail", "spec"],
    SubIntent.QUALITY_ISSUE: ["质量", "破损", "坏了", "缺陷", "quality", "defective", "broken"],
    SubIntent.SERVICE_ISSUE: ["态度", "服务", "客服", "慢", "service", "rude"],
    SubIntent.DELIVERY_ISSUE: ["快递", "配送", "物流", "没到", "delivery", "shipping"],
    SubIntent.RETURN_REQUEST: ["退货", "退回", "return"],
    SubIntent.EXCHANGE_REQUEST: ["换货", "换一个", "exchange"],
    SubIntent.REFUND_REQUEST: ["退款", "退钱", "refund"],
    SubIntent.ORDER_STATUS: ["订单状态", "到哪了", "order status"],
    SubIntent.ORDER_CANCEL: ["取消订单", "不要了", "cancel"],
    SubIntent.ORDER_MODIFY: ["修改订单", "改地址", "modify"],
    SubIntent.TRACKING: ["快递单号", "物流信息", "tracking"],
    SubIntent.DELIVERY_TIME: ["几天到", "什么时候到", "多久", "when", "how long"],
    SubIntent.SHIPPING_FEE: ["运费", "邮费", "包邮", "shipping fee"],
    SubIntent.POSITIVE_FEEDBACK: ["好评", "不错", "很好", "满意", "good", "great"],
    SubIntent.SUGGESTION: ["建议", "希望", "改善", "suggestion", "improve"],
}


class IntentClassifier:
    """基于关键词匹配 + Few-shot 模板的意图分类器"""

    def __init__(self) -> None:
        self._llm_func: Optional[callable] = None  # 可选 LLM 回调

    def set_llm_classifier(self, func: callable) -> None:
        """设置可选的 LLM 意图分类回调"""
        self._llm_func = func

    def classify(self, text: str) -> IntentResult:
        """对文本进行意图分类"""
        text_lower = text.lower().strip()

        # 1. 关键词快速匹配
        intent_scores = self._keyword_score(text_lower)
        best_intent = max(intent_scores, key=intent_scores.get)  # type: ignore[arg-type]
        confidence = intent_scores[best_intent]

        # 2. 二级意图
        sub_intent = self._classify_sub_intent(text_lower, best_intent)

        # 3. 如果有关键词匹配且置信度不高，尝试 LLM
        if confidence < 0.6 and self._llm_func:
            try:
                llm_result = self._llm_func(text)
                if llm_result is not None:
                    best_intent = llm_result.intent
                    sub_intent = llm_result.sub_intent
                    confidence = llm_result.confidence
            except Exception:
                pass  # LLM 失败回退到关键词结果

        # 确保最低置信度
        confidence = min(max(confidence, 0.1), 1.0)

        return IntentResult(
            intent=best_intent,
            sub_intent=sub_intent,
            confidence=confidence,
            raw_scores=intent_scores,
        )

    def _keyword_score(self, text: str) -> dict[IntentCategory, float]:
        """基于关键词计算意图得分"""
        scores: dict[IntentCategory, float] = {intent: 0.0 for intent in IntentCategory}

        for intent, keywords in _INTENT_KEYWORDS.items():
            match_count = 0
            for kw in keywords:
                if kw.lower() in text:
                    match_count += 1
            if match_count > 0:
                # 加权：匹配越多得分越高
                scores[intent] = min(0.5 + 0.15 * match_count, 0.95)

        # 如果没有任何匹配，给 greeting 一个默认分
        if all(v == 0.0 for v in scores.values()):
            scores[IntentCategory.GREETING] = 0.3

        return scores

    def _classify_sub_intent(
        self, text: str, primary_intent: IntentCategory
    ) -> SubIntent:
        """分类二级意图"""
        best_sub = SubIntent.UNKNOWN
        best_score = 0.0

        for sub_intent, keywords in _SUB_INTENT_KEYWORDS.items():
            match_count = sum(1 for kw in keywords if kw.lower() in text)
            if match_count > best_score:
                best_score = match_count
                best_sub = sub_intent

        return best_sub

    def detect_intent_switch(
        self, current_intent: IntentCategory, new_text: str
    ) -> bool:
        """检测意图是否发生切换"""
        new_intent = self.classify(new_text)
        return new_intent.intent != current_intent and new_intent.confidence > 0.5
