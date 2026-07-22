"""对话策略/路由模块"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional

from app.models import (
    DialogState,
    IntentCategory,
    IntentResult,
    SentimentResult,
    SubIntent,
    TransferReason,
)


class DialogAction(str, Enum):
    """对话动作"""
    REPLY = "reply"
    ASK_SLOT = "ask_slot"
    TRANSFER_HUMAN = "transfer_human"
    END = "end"
    CLARIFY = "clarify"


@dataclass
class PolicyDecision:
    """策略决策"""
    action: DialogAction
    reply_type: str = "default"
    pending_slot: Optional[str] = None
    transfer_reason: Optional[TransferReason] = None
    metadata: dict[str, Any] = field(default_factory=dict)


# 意图所需的必填槽位
_INTENT_REQUIRED_SLOTS: dict[IntentCategory, list[str]] = {
    IntentCategory.ORDER_QUERY: ["order_number"],
    IntentCategory.LOGISTICS: ["tracking_number"],
    IntentCategory.AFTER_SALE: ["order_number"],
}

# 转人工条件阈值
_TRANSFER_CONFIG = {
    "low_confidence_threshold": 0.35,
    "anger_score_threshold": 0.6,
    "max_repeated_failures": 3,
}


class DialogPolicy:
    """对话策略管理器"""

    def __init__(self) -> None:
        self._failure_counts: dict[str, int] = {}

    def decide(
        self,
        intent_result: IntentResult,
        sentiment_result: SentimentResult,
        dialog_state: Optional[DialogState] = None,
    ) -> PolicyDecision:
        """根据当前状态做出策略决策"""
        # 1. 转人工意图 -> 直接转接
        if intent_result.intent == IntentCategory.TRANSFER_HUMAN:
            return PolicyDecision(
                action=DialogAction.TRANSFER_HUMAN,
                transfer_reason=TransferReason.USER_REQUEST,
            )

        # 2. 检查是否需要转人工
        transfer = self._check_transfer_conditions(
            intent_result, sentiment_result
        )
        if transfer:
            return transfer

        # 3. 检查槽位填充
        if dialog_state and intent_result.intent in _INTENT_REQUIRED_SLOTS:
            required = _INTENT_REQUIRED_SLOTS[intent_result.intent]
            missing = [
                s for s in required
                if s not in dialog_state.slots or not dialog_state.slots[s].value
            ]
            if missing:
                return PolicyDecision(
                    action=DialogAction.ASK_SLOT,
                    pending_slot=missing[0],
                    metadata={"missing_slots": missing},
                )

        # 4. 默认回复
        reply_type = "default"
        if sentiment_result.is_angry:
            reply_type = "angry"
        elif sentiment_result.label.value == "positive" and intent_result.intent == IntentCategory.FEEDBACK:
            reply_type = "positive"

        return PolicyDecision(
            action=DialogAction.REPLY,
            reply_type=reply_type,
        )

    def _check_transfer_conditions(
        self,
        intent_result: IntentResult,
        sentiment_result: SentimentResult,
    ) -> Optional[PolicyDecision]:
        """检查是否满足转人工条件"""
        # 低置信度
        if intent_result.confidence < _TRANSFER_CONFIG["low_confidence_threshold"]:
            return PolicyDecision(
                action=DialogAction.TRANSFER_HUMAN,
                transfer_reason=TransferReason.LOW_CONFIDENCE,
                metadata={"confidence": intent_result.confidence},
            )

        # 愤怒情感
        if sentiment_result.anger_score >= _TRANSFER_CONFIG["anger_score_threshold"]:
            return PolicyDecision(
                action=DialogAction.TRANSFER_HUMAN,
                transfer_reason=TransferReason.NEGATIVE_SENTIMENT,
                metadata={"anger_score": sentiment_result.anger_score},
            )

        # 重复失败
        session_key = getattr(intent_result, "session_id", "unknown")
        if self._failure_counts.get(session_key, 0) >= _TRANSFER_CONFIG["max_repeated_failures"]:
            return PolicyDecision(
                action=DialogAction.TRANSFER_HUMAN,
                transfer_reason=TransferReason.REPEATED_FAILURE,
                metadata={"failure_count": self._failure_counts[session_key]},
            )

        return None

    def record_failure(self, session_id: str) -> None:
        """记录失败次数"""
        self._failure_counts[session_id] = self._failure_counts.get(session_id, 0) + 1

    def reset_failure(self, session_id: str) -> None:
        """重置失败次数"""
        self._failure_counts.pop(session_id, None)

    def get_failure_count(self, session_id: str) -> int:
        """获取失败次数"""
        return self._failure_counts.get(session_id, 0)
