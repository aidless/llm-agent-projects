"""排队管理模块"""

from __future__ import annotations

import heapq
from datetime import datetime
from typing import Optional

from app.models import QueueTicket, TransferReason


class TransferQueue:
    """人工转接排队管理"""

    def __init__(self, max_queue_size: int = 100) -> None:
        self._queue: list[tuple[int, int, QueueTicket]] = []  # (-priority, sequence, ticket)
        self._serving: dict[str, QueueTicket] = {}
        self._completed: list[QueueTicket] = []
        self._counter: int = 0
        self._max_queue_size = max_queue_size
        self._tickets_by_session: dict[str, QueueTicket] = {}

    def enqueue(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        priority: int = 0,
        transfer_reason: TransferReason = TransferReason.USER_REQUEST,
    ) -> QueueTicket:
        """加入排队"""
        if len(self._queue) >= self._max_queue_size:
            raise RuntimeError("排队已满，请稍后再试 / Queue is full, please try again later")

        ticket = QueueTicket(
            session_id=session_id,
            user_id=user_id,
            priority=min(max(priority, 0), 10),
            transfer_reason=transfer_reason,
        )
        self._counter += 1
        # 优先级队列：priority 越大越优先，使用负数实现
        heapq.heappush(self._queue, (-ticket.priority, self._counter, ticket))
        self._tickets_by_session[session_id] = ticket
        return ticket

    def dequeue(self) -> Optional[QueueTicket]:
        """取出下一个排队（最高优先级）"""
        while self._queue:
            neg_priority, seq, ticket = heapq.heappop(self._queue)
            if ticket.status == "waiting":
                ticket.status = "serving"
                self._serving[ticket.session_id] = ticket
                return ticket
        return None

    def get_queue_position(self, session_id: str) -> int:
        """获取排队位置（从1开始）"""
        waiting = [
            t for _, _, t in self._queue
            if t.session_id == session_id and t.status == "waiting"
        ]
        if not waiting:
            return 0

        # 计算前面有多少等待的
        position = 0
        for neg_priority, seq, t in sorted(self._queue):
            if t.status == "waiting":
                position += 1
            if t.session_id == session_id:
                return position
        return 0

    def get_queue_length(self) -> int:
        """获取当前排队长度"""
        return sum(1 for _, _, t in self._queue if t.status == "waiting")

    def complete_service(self, session_id: str) -> Optional[QueueTicket]:
        """完成服务"""
        ticket = self._serving.pop(session_id, None)
        if ticket:
            ticket.status = "completed"
            self._completed.append(ticket)
            self._tickets_by_session.pop(session_id, None)
            return ticket
        return None

    def cancel(self, session_id: str) -> Optional[QueueTicket]:
        """取消排队"""
        ticket = self._tickets_by_session.pop(session_id, None)
        if ticket:
            ticket.status = "cancelled"
            # 从排队中移除
            self._queue = [
                (np, s, t) for np, s, t in self._queue
                if t.session_id != session_id
            ]
            heapq.heapify(self._queue)
            return ticket
        # 也可能在服务中
        return self._serving.pop(session_id, None)

    def get_ticket(self, session_id: str) -> Optional[QueueTicket]:
        """获取会话的排队票"""
        return self._tickets_by_session.get(session_id)

    def get_waiting_list(self) -> list[QueueTicket]:
        """获取等待列表"""
        return [t for _, _, t in sorted(self._queue) if t.status == "waiting"]

    def get_stats(self) -> dict:
        """获取排队统计"""
        return {
            "waiting_count": self.get_queue_length(),
            "serving_count": len(self._serving),
            "completed_count": len(self._completed),
            "max_capacity": self._max_queue_size,
        }

    def clear_all(self) -> None:
        """清空所有排队"""
        self._queue.clear()
        self._serving.clear()
        self._completed.clear()
        self._tickets_by_session.clear()
