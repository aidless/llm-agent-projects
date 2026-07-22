"""回答生成模块"""

from __future__ import annotations

import uuid
from typing import Any, Optional

from app.models import (
    IntentCategory,
    KnowledgeSearchResult,
    SentimentLabel,
    SubIntent,
)


# 意图到回复的模板
_INTENT_RESPONSE_TEMPLATES: dict[IntentCategory, dict[str, str]] = {
    IntentCategory.GREETING: {
        "default": "您好！我是智能客服小助手，很高兴为您服务。请问有什么可以帮您？\nHello! I'm the smart customer service assistant. How can I help you today?",
    },
    IntentCategory.CONSULTATION: {
        "default": "关于您咨询的问题，让我为您查询一下相关信息。\nLet me look up the information for your inquiry.",
        "has_kb": "根据我们的知识库信息：\n{answer}\n\n来源：{source}",
    },
    IntentCategory.COMPLAINT: {
        "default": "非常抱歉给您带来不好的体验，我理解您的心情。让我记录您的反馈并尽快处理。\nI'm very sorry for the unpleasant experience. Let me record your feedback and process it as soon as possible.",
        "angry": "非常抱歉让您感到如此不满，我非常重视您的问题。让我立即为您升级处理。\nI sincerely apologize for your frustration. Your issue is very important to me. Let me escalate this immediately.",
    },
    IntentCategory.AFTER_SALE: {
        "default": "关于您的售后需求，让我为您查询相关政策。\nRegarding your after-sales request, let me check the relevant policies for you.",
        "has_kb": "根据我们的售后政策：\n{answer}\n\n如果您需要进一步帮助，请告诉我。",
    },
    IntentCategory.TRANSFER_HUMAN: {
        "default": "好的，我马上为您转接人工客服，请稍等。\nAlright, I'll transfer you to a human agent right away. Please wait a moment.",
    },
    IntentCategory.FEEDBACK: {
        "positive": "非常感谢您的好评和认可！我们会继续努力为您提供更好的服务。\nThank you very much for your positive feedback! We will continue to strive to provide you with better service.",
        "suggestion": "感谢您的宝贵建议！我们已将您的建议记录下来，会认真考虑并改进。\nThank you for your valuable suggestion! We have recorded it and will carefully consider it.",
    },
    IntentCategory.ORDER_QUERY: {
        "default": "请您提供一下订单号，我来为您查询订单信息。\nPlease provide your order number so I can look up your order information.",
        "has_kb": "根据查询结果：\n{answer}",
    },
    IntentCategory.LOGISTICS: {
        "default": "让我为您查询物流相关信息。\nLet me check the logistics information for you.",
        "has_kb": "物流信息如下：\n{answer}",
    },
}

_SLOT_PROMPTS: dict[str, str] = {
    "order_number": "请提供您的订单编号（如：ORD-20240101-XXXX）。\nPlease provide your order number.",
    "product_name": "请问您咨询的是哪款商品？\nWhich product are you inquiring about?",
    "tracking_number": "请提供快递单号，我来帮您查询。\nPlease provide the tracking number.",
    "return_reason": "请描述一下退货原因，方便我们为您处理。\nPlease describe the reason for the return.",
}


class ResponseGenerator:
    """回答生成器"""

    def __init__(self) -> None:
        self._llm_func: Optional[callable] = None

    def set_llm_generator(self, func: callable) -> None:
        """设置可选的 LLM 回答生成回调"""
        self._llm_func = func

    def generate(
        self,
        intent: IntentCategory,
        sub_intent: SubIntent,
        sentiment: SentimentLabel,
        is_angry: bool = False,
        knowledge_results: Optional[list[KnowledgeSearchResult]] = None,
        pending_slot: Optional[str] = None,
        context: Optional[dict[str, Any]] = None,
    ) -> tuple[str, list[dict[str, str]]]:
        """
        生成回复。
        返回 (reply_text, sources_list)
        """
        sources: list[dict[str, str]] = []

        # 1. 检查是否有待填充的槽位
        if pending_slot and pending_slot in _SLOT_PROMPTS:
            slot_reply = _SLOT_PROMPTS[pending_slot]
            return slot_reply, sources

        # 2. 如果有知识库结果，优先使用
        if knowledge_results:
            answer = knowledge_results[0].answer
            source_ref = f"{knowledge_results[0].question}"
            sources.append(
                {
                    "id": knowledge_results[0].id,
                    "question": knowledge_results[0].question,
                    "answer": knowledge_results[0].answer,
                }
            )
            templates = _INTENT_RESPONSE_TEMPLATES.get(intent, {})
            if "has_kb" in templates:
                reply = templates["has_kb"].format(answer=answer, source=source_ref)
            else:
                reply = f"根据我们的知识库信息：\n{answer}\n\nBased on our knowledge base:\n{answer}"
            return reply, sources

        # 3. 使用意图模板
        templates = _INTENT_RESPONSE_TEMPLATES.get(intent, {})
        reply = templates.get("default", "让我为您处理这个问题。\nLet me handle this for you.")

        # 4. 情感调整
        if is_angry and "angry" in templates:
            reply = templates["angry"]
        elif sentiment == SentimentLabel.POSITIVE and intent == IntentCategory.FEEDBACK:
            reply = templates.get("positive", reply)

        # 5. LLM 优化（可选）
        if self._llm_func:
            try:
                llm_reply = self._llm_func(
                    intent=intent,
                    context=context or {},
                    base_reply=reply,
                )
                if llm_reply:
                    reply = llm_reply
            except Exception:
                pass

        return reply, sources

    def generate_transfer_summary(
        self,
        messages: list[dict[str, Any]],
        sentiment_history: list[Any],
    ) -> str:
        """生成转人工的上下文摘要"""
        if not messages:
            return "无对话历史。No conversation history."

        summary_parts = ["【对话摘要 / Conversation Summary】"]

        # 最近的对话
        recent_msgs = messages[-6:]
        for msg in recent_msgs:
            role = "用户" if msg.get("role") == "user" else "客服"
            summary_parts.append(f"  {role}: {msg.get('content', '')[:100]}")

        # 情感摘要
        if sentiment_history:
            last_sentiment = sentiment_history[-1]
            summary_parts.append(f"\n【情感状态 / Sentiment】{last_sentiment.label.value}")

        if any(getattr(s, "is_angry", False) for s in sentiment_history):
            summary_parts.append("  ⚠ 客户存在愤怒情绪，请注意沟通方式。")

        return "\n".join(summary_parts)
