"""对话管理测试"""

import pytest
from dialog.state_tracker import DialogStateTracker
from dialog.policy import DialogPolicy
from dialog.history import DialogHistory
from app.models import (
    DialogState,
    IntentCategory,
    IntentResult,
    SentimentLabel,
    SentimentResult,
    SubIntent,
    TransferReason,
    MessageRole,
)
from transfer.escalator import Escalator
from transfer.queue import TransferQueue


class TestDialogStateTracker:
    """对话状态跟踪测试"""

    @pytest.fixture
    def tracker(self):
        return DialogStateTracker()

    def test_create_state(self, tracker):
        """测试创建状态"""
        state = tracker.create_state("session_001")
        assert state.session_id == "session_001"
        assert state.turn_count == 0

    def test_get_or_create(self, tracker):
        """测试获取或创建"""
        state = tracker.get_or_create_state("session_001")
        assert state.session_id == "session_001"
        state2 = tracker.get_or_create_state("session_001")
        assert state2.session_id == state.session_id

    def test_update_intent(self, tracker):
        """测试更新意图"""
        tracker.update_intent("session_001", IntentCategory.ORDER_QUERY)
        state = tracker.get_state("session_001")
        assert state.current_intent == IntentCategory.ORDER_QUERY

    def test_fill_slot(self, tracker):
        """测试填充槽位"""
        tracker.fill_slot("session_001", "order_number", "ORD-20240101-ABCD")
        value = tracker.get_slot("session_001", "order_number")
        assert value == "ORD-20240101-ABCD"

    def test_confirm_slot(self, tracker):
        """测试确认槽位"""
        tracker.fill_slot("session_001", "order_number", "ORD-12345")
        tracker.confirm_slot("session_001", "order_number")
        state = tracker.get_state("session_001")
        assert state.slots["order_number"].confirmed is True

    def test_increment_turn(self, tracker):
        """测试增加轮次"""
        turns = tracker.increment_turn("session_001")
        assert turns == 1
        turns = tracker.increment_turn("session_001")
        assert turns == 2

    def test_missing_slots(self, tracker):
        """测试缺失槽位检测"""
        missing = tracker.get_missing_slots("session_001", ["order_number"])
        assert "order_number" in missing

    def test_reset_state(self, tracker):
        """测试重置状态"""
        tracker.update_intent("session_001", IntentCategory.ORDER_QUERY)
        tracker.fill_slot("session_001", "order_number", "ORD-12345")
        tracker.reset_state("session_001")
        state = tracker.get_state("session_001")
        assert state.current_intent is None
        assert state.turn_count == 0

    def test_delete_state(self, tracker):
        """测试删除状态"""
        tracker.create_state("session_001")
        assert tracker.delete_state("session_001") is True
        assert tracker.get_state("session_001") is None

    def test_to_dict(self, tracker):
        """测试转为字典"""
        tracker.update_intent("session_001", IntentCategory.ORDER_QUERY)
        d = tracker.to_dict("session_001")
        assert d is not None
        assert d["session_id"] == "session_001"


class TestDialogPolicy:
    """对话策略测试"""

    @pytest.fixture
    def policy(self):
        return DialogPolicy()

    @pytest.fixture
    def normal_intent(self):
        return IntentResult(
            intent=IntentCategory.CONSULTATION,
            sub_intent=SubIntent.PRODUCT_INFO,
            confidence=0.8,
        )

    @pytest.fixture
    def neutral_sentiment(self):
        return SentimentResult(
            label=SentimentLabel.NEUTRAL,
            score=0.5,
        )

    def test_transfer_human_decision(self, policy):
        """测试转人工决策"""
        intent = IntentResult(
            intent=IntentCategory.TRANSFER_HUMAN,
            sub_intent=SubIntent.UNKNOWN,
            confidence=0.9,
        )
        sentiment = SentimentResult(label=SentimentLabel.NEUTRAL, score=0.5)
        decision = policy.decide(intent, sentiment)
        assert decision.action.value == "transfer_human"
        assert decision.transfer_reason == TransferReason.USER_REQUEST

    def test_reply_decision(self, policy, normal_intent, neutral_sentiment):
        """测试回复决策"""
        decision = policy.decide(normal_intent, neutral_sentiment)
        assert decision.action.value == "reply"

    def test_low_confidence_transfer(self, policy):
        """测试低置信度转人工"""
        intent = IntentResult(
            intent=IntentCategory.CONSULTATION,
            sub_intent=SubIntent.UNKNOWN,
            confidence=0.2,
        )
        sentiment = SentimentResult(label=SentimentLabel.NEUTRAL, score=0.5)
        decision = policy.decide(intent, sentiment)
        assert decision.action.value == "transfer_human"
        assert decision.transfer_reason == TransferReason.LOW_CONFIDENCE

    def test_anger_transfer(self, policy):
        """测试愤怒转人工"""
        intent = IntentResult(
            intent=IntentCategory.COMPLAINT,
            sub_intent=SubIntent.QUALITY_ISSUE,
            confidence=0.8,
        )
        sentiment = SentimentResult(
            label=SentimentLabel.NEGATIVE,
            score=0.8,
            is_angry=True,
            anger_score=0.8,
        )
        decision = policy.decide(intent, sentiment)
        assert decision.action.value == "transfer_human"
        assert decision.transfer_reason == TransferReason.NEGATIVE_SENTIMENT

    def test_ask_slot_decision(self, policy):
        """测试槽位询问决策"""
        intent = IntentResult(
            intent=IntentCategory.ORDER_QUERY,
            sub_intent=SubIntent.ORDER_STATUS,
            confidence=0.8,
        )
        sentiment = SentimentResult(label=SentimentLabel.NEUTRAL, score=0.5)
        state = DialogState(session_id="test")
        decision = policy.decide(intent, sentiment, state)
        assert decision.action.value == "ask_slot"
        assert decision.pending_slot == "order_number"

    def test_failure_counting(self, policy):
        """测试失败计数"""
        policy.record_failure("session_001")
        policy.record_failure("session_001")
        assert policy.get_failure_count("session_001") == 2
        policy.reset_failure("session_001")
        assert policy.get_failure_count("session_001") == 0


class TestDialogHistory:
    """对话历史测试"""

    @pytest.fixture
    def history(self):
        return DialogHistory()

    def test_add_and_get_messages(self, history):
        """测试添加和获取消息"""
        history.add_message("s1", MessageRole.USER, "你好")
        history.add_message("s1", MessageRole.AGENT, "您好，有什么可以帮您？")
        msgs = history.get_history("s1")
        assert len(msgs) == 2

    def test_get_history_dict(self, history):
        """测试获取历史字典"""
        history.add_message("s1", MessageRole.USER, "你好")
        d = history.get_history_dict("s1")
        assert len(d) == 1
        assert d[0]["role"] == "user"

    def test_message_count(self, history):
        """测试消息计数"""
        history.add_message("s1", MessageRole.USER, "你好")
        assert history.get_message_count("s1") == 1

    def test_context_window(self, history):
        """测试上下文窗口"""
        for i in range(10):
            history.add_message("s1", MessageRole.USER, f"这是一条比较长的测试消息内容{i}")
        window = history.get_context_window("s1", max_tokens=50)
        assert len(window) < 10

    def test_summary_generation(self, history):
        """测试摘要生成"""
        for i in range(5):
            history.add_message("s1", MessageRole.USER, f"用户消息{i}")
        summary = history.generate_summary("s1")
        assert len(summary) > 0

    def test_clear_history(self, history):
        """测试清除历史"""
        history.add_message("s1", MessageRole.USER, "你好")
        history.clear_history("s1")
        assert history.get_message_count("s1") == 0

    def test_get_history_last_n(self, history):
        """测试获取最近N条"""
        for i in range(10):
            history.add_message("s1", MessageRole.USER, f"消息{i}")
        msgs = history.get_history("s1", last_n=3)
        assert len(msgs) == 3

    def test_sliding_window(self, history):
        """测试滑动窗口"""
        small_history = DialogHistory(max_window_size=5)
        for i in range(10):
            small_history.add_message("s1", MessageRole.USER, f"消息{i}")
        msgs = small_history.get_history("s1")
        assert len(msgs) == 5


class TestEscalator:
    """升级转接测试"""

    @pytest.fixture
    def escalator(self):
        return Escalator()

    def test_user_request_transfer(self, escalator):
        """测试用户请求转接"""
        rule = escalator.should_transfer(intent="transfer_human")
        assert rule is not None
        assert rule.reason == TransferReason.USER_REQUEST

    def test_no_transfer_needed(self, escalator):
        """测试无需转接"""
        rule = escalator.should_transfer(
            intent="consultation",
            confidence=0.9,
            anger_score=0.0,
        )
        assert rule is None

    def test_anger_transfer(self, escalator):
        """测试愤怒转接"""
        rule = escalator.should_transfer(anger_score=0.7)
        assert rule is not None
        assert rule.reason == TransferReason.NEGATIVE_SENTIMENT

    def test_low_confidence_transfer(self, escalator):
        """测试低置信度转接"""
        rule = escalator.should_transfer(confidence=0.2)
        assert rule is not None
        assert rule.reason == TransferReason.LOW_CONFIDENCE

    def test_custom_rule(self, escalator):
        """测试自定义规则"""
        from transfer.escalator import TransferRule
        escalator.add_rule(TransferRule(
            name="test_rule",
            reason=TransferReason.COMPLEX_QUERY,
            condition=lambda ctx: ctx.get("turn_count", 0) >= 20,
            priority=3,
        ))
        rule = escalator.should_transfer(turn_count=25)
        assert rule is not None

    def test_get_rules(self, escalator):
        """测试获取规则列表"""
        rules = escalator.get_rules()
        assert len(rules) >= 5


class TestTransferQueue:
    """排队管理测试"""

    @pytest.fixture
    def queue(self):
        return TransferQueue()

    def test_enqueue_and_dequeue(self, queue):
        """测试排队和出队"""
        queue.enqueue("s1", priority=0)
        queue.enqueue("s2", priority=5)
        ticket = queue.dequeue()
        assert ticket.session_id == "s2"  # 高优先级先出

    def test_queue_position(self, queue):
        """测试排队位置"""
        queue.enqueue("s1", priority=0)
        queue.enqueue("s2", priority=0)
        queue.enqueue("s3", priority=0)
        assert queue.get_queue_position("s1") == 1
        assert queue.get_queue_position("s3") == 3

    def test_cancel(self, queue):
        """测试取消排队"""
        queue.enqueue("s1")
        ticket = queue.cancel("s1")
        assert ticket is not None
        assert ticket.status == "cancelled"

    def test_complete_service(self, queue):
        """测试完成服务"""
        queue.enqueue("s1")
        queue.dequeue()  # 进入服务
        ticket = queue.complete_service("s1")
        assert ticket is not None
        assert ticket.status == "completed"

    def test_queue_stats(self, queue):
        """测试排队统计"""
        queue.enqueue("s1")
        queue.enqueue("s2")
        stats = queue.get_stats()
        assert stats["waiting_count"] == 2

    def test_queue_full(self, queue):
        """测试排队已满"""
        small_queue = TransferQueue(max_queue_size=2)
        small_queue.enqueue("s1")
        small_queue.enqueue("s2")
        with pytest.raises(RuntimeError):
            small_queue.enqueue("s3")
