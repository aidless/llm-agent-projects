"""反馈收集 - 管理用户反馈、自动评估和标注。"""

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from storage.memory_store import get_store


class FeedbackType(str, Enum):
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    RATING = "rating"
    CORRECTION = "correction"
    FLAG = "flag"


@dataclass
class Feedback:
    """用户反馈记录。"""

    trace_id: str
    feedback_type: FeedbackType
    value: Optional[float] = None  # 用于评分 (1-5)
    comment: str = ""
    user_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    feedback_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feedback_id": self.feedback_id,
            "trace_id": self.trace_id,
            "feedback_type": self.feedback_type.value,
            "value": self.value,
            "comment": self.comment,
            "user_id": self.user_id,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }


class FeedbackCollector:
    """反馈收集器 - 收集和管理用户对 LLM 输出的反馈。

    用法::

        collector = FeedbackCollector()
        collector.add_thumbs_up(trace_id="abc123", user_id="user1")
        collector.add_rating(trace_id="abc123", rating=4, comment="Good answer")
    """

    def __init__(self) -> None:
        self._store = get_store()

    def add_thumbs_up(
        self, trace_id: str, user_id: Optional[str] = None, comment: str = ""
    ) -> Feedback:
        """添加点赞反馈。"""
        fb = Feedback(
            trace_id=trace_id,
            feedback_type=FeedbackType.THUMBS_UP,
            user_id=user_id,
            comment=comment,
        )
        self._store.store_feedback(fb.to_dict())
        return fb

    def add_thumbs_down(
        self,
        trace_id: str,
        user_id: Optional[str] = None,
        comment: str = "",
    ) -> Feedback:
        """添加点踩反馈。"""
        fb = Feedback(
            trace_id=trace_id,
            feedback_type=FeedbackType.THUMBS_DOWN,
            user_id=user_id,
            comment=comment,
        )
        self._store.store_feedback(fb.to_dict())
        return fb

    def add_rating(
        self,
        trace_id: str,
        rating: float,
        user_id: Optional[str] = None,
        comment: str = "",
    ) -> Feedback:
        """添加评分反馈 (1-5)。"""
        fb = Feedback(
            trace_id=trace_id,
            feedback_type=FeedbackType.RATING,
            value=rating,
            user_id=user_id,
            comment=comment,
        )
        self._store.store_feedback(fb.to_dict())
        return fb

    def add_correction(
        self,
        trace_id: str,
        correction: str,
        user_id: Optional[str] = None,
    ) -> Feedback:
        """添加纠正反馈。"""
        fb = Feedback(
            trace_id=trace_id,
            feedback_type=FeedbackType.CORRECTION,
            comment=correction,
            user_id=user_id,
        )
        self._store.store_feedback(fb.to_dict())
        return fb

    def get_feedback(self, feedback_id: str) -> Optional[Dict[str, Any]]:
        return self._store.get_feedback(feedback_id)

    def list_trace_feedbacks(self, trace_id: str) -> List[Dict[str, Any]]:
        return self._store.list_feedbacks(trace_id=trace_id)

    def get_stats(self) -> Dict[str, Any]:
        """获取反馈统计信息。"""
        all_fbs = self._store.list_feedbacks(limit=10000)
        thumbs_up = sum(1 for f in all_fbs if f["feedback_type"] == "thumbs_up")
        thumbs_down = sum(1 for f in all_fbs if f["feedback_type"] == "thumbs_down")
        ratings = [f["value"] for f in all_fbs if f["feedback_type"] == "rating" and f["value"] is not None]

        return {
            "total": len(all_fbs),
            "thumbs_up": thumbs_up,
            "thumbs_down": thumbs_down,
            "thumbs_up_ratio": thumbs_up / len(all_fbs) if all_fbs else 0,
            "avg_rating": sum(ratings) / len(ratings) if ratings else None,
            "rating_count": len(ratings),
        }