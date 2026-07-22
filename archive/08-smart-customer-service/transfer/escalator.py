"""人工转接升级模块"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Optional

from app.models import (
    SentimentResult,
    TransferReason,
    TransferRequest,
)


# 转接规则
@dataclass
class TransferRule:
    """转接规则"""
    name: str
    reason: TransferReason
    condition: Callable[[dict[str, Any]], bool]
    priority: int = 0
    enabled: bool = True


class Escalator:
    """升级转接管理器"""

    def __init__(self) -> None:
        self._rules: list[TransferRule] = []
        self._transfer_history: list[dict[str, Any]] = []
        self._setup_default_rules()

    def _setup_default_rules(self) -> None:
        """设置默认转接规则"""
        self._rules = [
            TransferRule(
                name="user_request",
                reason=TransferReason.USER_REQUEST,
                condition=lambda ctx: ctx.get("intent") == "transfer_human",
                priority=10,
            ),
            TransferRule(
                name="high_anger",
                reason=TransferReason.NEGATIVE_SENTIMENT,
                condition=lambda ctx: ctx.get("anger_score", 0) >= 0.6,
                priority=8,
            ),
            TransferRule(
                name="low_confidence",
                reason=TransferReason.LOW_CONFIDENCE,
                condition=lambda ctx: ctx.get("confidence", 1.0) < 0.35,
                priority=5,
            ),
            TransferRule(
                name="repeated_failure",
                reason=TransferReason.REPEATED_FAILURE,
                condition=lambda ctx: ctx.get("failure_count", 0) >= 3,
                priority=7,
            ),
            TransferRule(
                name="complex_query",
                reason=TransferReason.COMPLEX_QUERY,
                condition=lambda ctx: ctx.get("turn_count", 0) >= 10 and ctx.get("confidence", 1.0) < 0.5,
                priority=6,
            ),
        ]

    def add_rule(self, rule: TransferRule) -> None:
        """添加转接规则"""
        self._rules.append(rule)
        self._rules.sort(key=lambda r: r.priority, reverse=True)

    def remove_rule(self, name: str) -> bool:
        """移除转接规则"""
        for i, rule in enumerate(self._rules):
            if rule.name == name:
                self._rules.pop(i)
                return True
        return False

    def evaluate(
        self,
        context: dict[str, Any],
    ) -> Optional[TransferRule]:
        """评估是否需要转接"""
        for rule in self._rules:
            if rule.enabled:
                try:
                    if rule.condition(context):
                        return rule
                except Exception:
                    continue
        return None

    def should_transfer(
        self,
        intent: str = "",
        confidence: float = 1.0,
        anger_score: float = 0.0,
        failure_count: int = 0,
        turn_count: int = 0,
    ) -> Optional[TransferRule]:
        """简化接口：评估是否需要转接"""
        context = {
            "intent": intent,
            "confidence": confidence,
            "anger_score": anger_score,
            "failure_count": failure_count,
            "turn_count": turn_count,
        }
        return self.evaluate(context)

    def create_transfer_request(
        self,
        session_id: str,
        reason: TransferReason,
        context_summary: str,
        sentiment_history: Optional[list[SentimentResult]] = None,
        priority: int = 0,
    ) -> TransferRequest:
        """创建转接请求"""
        request = TransferRequest(
            session_id=session_id,
            reason=reason,
            context_summary=context_summary,
            sentiment_history=sentiment_history or [],
            priority=priority,
        )
        self._transfer_history.append({
            "request": request,
            "created_at": datetime.utcnow().isoformat(),
        })
        return request

    def get_transfer_history(self) -> list[dict[str, Any]]:
        """获取转接历史"""
        return self._transfer_history.copy()

    def get_rules(self) -> list[dict[str, Any]]:
        """获取所有规则"""
        return [
            {
                "name": r.name,
                "reason": r.reason.value,
                "priority": r.priority,
                "enabled": r.enabled,
            }
            for r in self._rules
        ]
