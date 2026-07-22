"""对话状态跟踪模块 (DST)"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from app.models import DialogState, IntentCategory, SlotValue, SubIntent


class DialogStateTracker:
    """对话状态跟踪器"""

    def __init__(self) -> None:
        self._states: dict[str, DialogState] = {}

    def get_state(self, session_id: str) -> Optional[DialogState]:
        """获取对话状态"""
        return self._states.get(session_id)

    def create_state(self, session_id: str) -> DialogState:
        """创建新的对话状态"""
        state = DialogState(session_id=session_id)
        self._states[session_id] = state
        return state

    def get_or_create_state(self, session_id: str) -> DialogState:
        """获取或创建对话状态"""
        state = self.get_state(session_id)
        if state is None:
            state = self.create_state(session_id)
        return state

    def update_intent(
        self,
        session_id: str,
        intent: IntentCategory,
        sub_intent: Optional[SubIntent] = None,
    ) -> DialogState:
        """更新意图"""
        state = self.get_or_create_state(session_id)
        state.current_intent = intent
        if sub_intent:
            state.sub_intent = sub_intent
        state.last_updated = datetime.utcnow()
        return state

    def fill_slot(
        self,
        session_id: str,
        slot_name: str,
        value: str,
        confirmed: bool = False,
    ) -> DialogState:
        """填充槽位"""
        state = self.get_or_create_state(session_id)
        state.slots[slot_name] = SlotValue(
            name=slot_name,
            value=value,
            confirmed=confirmed,
        )
        state.last_updated = datetime.utcnow()
        return state

    def get_slot(self, session_id: str, slot_name: str) -> Optional[str]:
        """获取槽位值"""
        state = self.get_state(session_id)
        if state and slot_name in state.slots:
            return state.slots[slot_name].value
        return None

    def confirm_slot(self, session_id: str, slot_name: str) -> Optional[str]:
        """确认槽位"""
        state = self.get_state(session_id)
        if state and slot_name in state.slots:
            state.slots[slot_name].confirmed = True
            state.last_updated = datetime.utcnow()
            return state.slots[slot_name].value
        return None

    def increment_turn(self, session_id: str) -> int:
        """增加轮次计数"""
        state = self.get_or_create_state(session_id)
        state.turn_count += 1
        state.last_updated = datetime.utcnow()
        return state.turn_count

    def set_waiting_for_slot(
        self,
        session_id: str,
        slot_name: Optional[str],
        waiting: bool = True,
    ) -> DialogState:
        """设置等待槽位状态"""
        state = self.get_or_create_state(session_id)
        state.is_waiting_for_slot = waiting
        state.pending_slot_name = slot_name
        state.last_updated = datetime.utcnow()
        return state

    def get_all_filled_slots(self, session_id: str) -> dict[str, str]:
        """获取所有已填充的槽位"""
        state = self.get_state(session_id)
        if not state:
            return {}
        return {
            name: sv.value
            for name, sv in state.slots.items()
            if sv.value is not None
        }

    def get_missing_slots(
        self, session_id: str, required_slots: list[str]
    ) -> list[str]:
        """获取缺失的必填槽位"""
        state = self.get_state(session_id)
        if not state:
            return list(required_slots)
        return [
            slot_name
            for slot_name in required_slots
            if slot_name not in state.slots or state.slots[slot_name].value is None
        ]

    def reset_state(self, session_id: str) -> Optional[DialogState]:
        """重置对话状态（保留session_id）"""
        state = self.get_state(session_id)
        if state:
            state.current_intent = None
            state.sub_intent = None
            state.slots = {}
            state.turn_count = 0
            state.is_waiting_for_slot = False
            state.pending_slot_name = None
            state.last_updated = datetime.utcnow()
            return state
        return None

    def delete_state(self, session_id: str) -> bool:
        """删除对话状态"""
        return self._states.pop(session_id, None) is not None

    def to_dict(self, session_id: str) -> Optional[dict[str, Any]]:
        """将状态转为字典"""
        state = self.get_state(session_id)
        if not state:
            return None
        return state.model_dump()
