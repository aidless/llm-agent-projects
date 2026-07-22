"""上下文管理模块"""

from __future__ import annotations

from typing import Any, Optional

from app.models import DialogState, Message, SlotValue


# 意图对应的必填槽位
_REQUIRED_SLOTS: dict[str, list[str]] = {
    "order_query": ["order_number"],
    "logistics": ["tracking_number"],
    "after_sale": ["order_number", "return_reason"],
    "return_request": ["order_number"],
    "exchange_request": ["order_number"],
    "refund_request": ["order_number"],
}


class ContextManager:
    """上下文管理器 - 负责指代消解、上下文维护"""

    def __init__(self) -> None:
        self._sessions: dict[str, DialogState] = {}

    def get_or_create_state(self, session_id: str) -> DialogState:
        """获取或创建对话状态"""
        if session_id not in self._sessions:
            self._sessions[session_id] = DialogState(session_id=session_id)
        return self._sessions[session_id]

    def update_state(
        self,
        session_id: str,
        intent: Optional[str] = None,
        sub_intent: Optional[str] = None,
        entities: Optional[dict[str, list[str]]] = None,
    ) -> DialogState:
        """更新对话状态"""
        state = self.get_or_create_state(session_id)

        if intent:
            state.current_intent = intent
        if sub_intent:
            state.sub_intent = sub_intent
        state.turn_count += 1

        # 从实体中填充槽位
        if entities:
            for entity_type, values in entities.items():
                if values:
                    if entity_type not in state.slots:
                        state.slots[entity_type] = SlotValue(name=entity_type)
                    state.slots[entity_type].value = values[0]
                    state.slots[entity_type].confirmed = False

        # 检查待填充槽位
        self._check_pending_slots(state)

        return state

    def _check_pending_slots(self, state: DialogState) -> None:
        """检查是否还有未填充的槽位"""
        if not state.current_intent:
            return

        required = _REQUIRED_SLOTS.get(state.current_intent.value, [])
        if not required:
            state.is_waiting_for_slot = False
            state.pending_slot_name = None
            return

        for slot_name in required:
            if slot_name not in state.slots or not state.slots[slot_name].value:
                state.is_waiting_for_slot = True
                state.pending_slot_name = slot_name
                return

        state.is_waiting_for_slot = False
        state.pending_slot_name = None

    def resolve_reference(
        self, current_text: str, history: list[Message]
    ) -> str:
        """
        指代消解：将当前文本中的指代替换为实际内容。
        简单实现：检查常见指代词并从历史中寻找对应实体。
        """
        # 常见指代词
        pronouns = ["它", "这个", "那个", "这件", "该", "it", "this", "that", "the"]
        words = current_text.split()

        replaced = False
        for i, word in enumerate(words):
            if word in pronouns and history:
                # 从最近的消息中寻找实体
                for msg in reversed(history):
                    if msg.role.value == "agent" and msg.metadata.get("last_entity"):
                        words[i] = msg.metadata["last_entity"]
                        replaced = True
                        break

        return " ".join(words) if replaced else current_text

    def get_context_summary(
        self,
        session_id: str,
        max_turns: int = 5,
    ) -> dict[str, Any]:
        """获取上下文摘要"""
        state = self._sessions.get(session_id)
        if not state:
            return {}

        return {
            "session_id": session_id,
            "current_intent": state.current_intent.value if state.current_intent else None,
            "sub_intent": state.sub_intent.value if state.sub_intent else None,
            "turn_count": state.turn_count,
            "filled_slots": {
                name: sv.value for name, sv in state.slots.items() if sv.value
            },
            "pending_slots": state.pending_slot_name,
            "is_waiting_for_slot": state.is_waiting_for_slot,
        }

    def clear_session(self, session_id: str) -> None:
        """清除会话状态"""
        self._sessions.pop(session_id, None)

    def confirm_slot(self, session_id: str, slot_name: str) -> Optional[str]:
        """确认槽位值"""
        state = self._sessions.get(session_id)
        if state and slot_name in state.slots:
            state.slots[slot_name].confirmed = True
            return state.slots[slot_name].value
        return None

    def get_all_slots(self, session_id: str) -> dict[str, Optional[str]]:
        """获取所有槽位值"""
        state = self._sessions.get(session_id)
        if not state:
            return {}
        return {
            name: sv.value for name, sv in state.slots.items()
        }
