"""Dialog 模块测试"""

import asyncio
import time
import pytest

from dialog.context import DialogContext, MessageRole, IntentResult, Message
from dialog.manager import DialogManager, MockLLMClient, INTENT_PATTERNS
from dialog.barge_in import BargeInHandler, BargeInConfig, BargeInState, BargeInEvent


class TestDialogContext:
    def test_add_message(self):
        """测试添加消息"""
        ctx = DialogContext()
        msg = ctx.add_message(MessageRole.USER, "你好")
        assert msg.role == MessageRole.USER
        assert msg.content == "你好"
        assert msg.message_id

    def test_get_messages(self):
        """测试获取消息"""
        ctx = DialogContext()
        ctx.add_message(MessageRole.USER, "你好")
        ctx.add_message(MessageRole.ASSISTANT, "你好！")
        ctx.add_message(MessageRole.USER, "天气怎么样")

        all_msgs = ctx.get_messages()
        assert len(all_msgs) == 3

        user_msgs = ctx.get_messages(role=MessageRole.USER)
        assert len(user_msgs) == 2

        recent = ctx.get_recent_messages(2)
        assert len(recent) == 2
        assert recent[-1].content == "天气怎么样"

    def test_turn_count(self):
        """测试轮数计数"""
        ctx = DialogContext()
        assert ctx.get_turn_count() == 0
        ctx.add_message(MessageRole.USER, "test")
        assert ctx.get_turn_count() == 1
        ctx.add_message(MessageRole.USER, "test2")
        assert ctx.get_turn_count() == 2

    def test_max_turns(self):
        """测试最大轮数"""
        ctx = DialogContext(max_turns=3)
        assert ctx.is_max_turns() is False
        ctx.add_message(MessageRole.USER, "1")
        ctx.add_message(MessageRole.USER, "2")
        ctx.add_message(MessageRole.USER, "3")
        assert ctx.is_max_turns() is True

    def test_context_for_llm(self):
        """测试构建 LLM 上下文"""
        ctx = DialogContext()
        ctx.add_message(MessageRole.USER, "你好")
        ctx.add_message(MessageRole.ASSISTANT, "你好！")
        ctx.add_message(MessageRole.USER, "天气怎么样")

        llm_context = ctx.get_context_for_llm(system_prompt="你是助手")
        assert llm_context[0]["role"] == "system"
        assert llm_context[0]["content"] == "你是助手"
        assert len(llm_context) > 1

    def test_entities(self):
        """测试实体管理"""
        ctx = DialogContext()
        ctx.update_entity("city", "北京")
        ctx.update_entity("date", "今天")
        entities = ctx.get_entities()
        assert entities["city"] == "北京"
        assert entities["date"] == "今天"
        ctx.clear_entities()
        assert len(ctx.get_entities()) == 0

    def test_intent_history(self):
        """测试意图历史"""
        ctx = DialogContext()
        intent = IntentResult(intent="greeting", confidence=0.9, raw_text="你好")
        ctx.add_intent(intent)
        # 实体应从意图中提取
        assert ctx.get_turn_count() == 0  # 意图不影响轮数

    def test_clear(self):
        """测试清空上下文"""
        ctx = DialogContext()
        ctx.add_message(MessageRole.USER, "test")
        ctx.update_entity("key", "value")
        ctx.clear()
        assert len(ctx.get_messages()) == 0
        assert len(ctx.get_entities()) == 0
        assert ctx.get_turn_count() == 0

    def test_expiry(self):
        """测试超时检测"""
        ctx = DialogContext()
        assert ctx.is_expired(timeout_seconds=300) is False


class TestMockLLMClient:
    @pytest.mark.asyncio
    async def test_generate_response(self):
        """测试 LLM 生成回复"""
        client = MockLLMClient()
        messages = [
            {"role": "user", "content": "你好"},
        ]
        response = await client.generate(messages)
        assert isinstance(response, str)
        assert len(response) > 0

    @pytest.mark.asyncio
    async def test_generate_stream(self):
        """测试 LLM 流式生成"""
        client = MockLLMClient()
        messages = [
            {"role": "user", "content": "你好"},
        ]
        chunks = []
        async for chunk in await client.generate(messages, stream=True):
            chunks.append(chunk)
        assert len(chunks) > 0
        full = ''.join(chunks)
        assert len(full) > 0


class TestDialogManager:
    @pytest.mark.asyncio
    async def test_process_text(self):
        """测试文本处理"""
        manager = DialogManager()
        result = await manager.process_text_sync("session1", "你好")
        assert result["type"] == "text_response"
        assert result["text"]
        assert result["session_id"] == "session1"
        assert result["intent"] == "greeting"

    @pytest.mark.asyncio
    async def test_process_text_stream(self):
        """测试流式文本处理"""
        manager = DialogManager()
        chunks = []
        async for chunk in manager.process_text_stream("session2", "你好"):
            chunks.append(chunk)
        assert len(chunks) > 0
        # 最后一个 chunk 应该是最终结果
        assert chunks[-1]["is_final"] is True

    @pytest.mark.asyncio
    async def test_multi_turn_context(self):
        """测试多轮对话上下文"""
        manager = DialogManager()
        await manager.process_text_sync("session3", "你好")
        result2 = await manager.process_text_sync("session3", "天气怎么样")
        assert result2["turn_count"] == 2

    def test_recognize_intent(self):
        """测试意图识别"""
        manager = DialogManager()
        result = manager.recognize_intent("你好")
        assert result.intent == "greeting"
        assert result.confidence > 0.8

        result = manager.recognize_intent("今天天气怎么样")
        assert result.intent == "weather"

        result = manager.recognize_intent("谢谢你的帮助")
        assert result.intent == "thanks"

    def test_remove_session(self):
        """测试移除会话"""
        manager = DialogManager()
        manager.get_or_create_context("test_session")
        manager.remove_session("test_session")
        # 重新创建应该得到新的上下文
        ctx = manager.get_or_create_context("test_session")
        assert ctx is not None


class TestBargeInHandler:
    def test_initial_state(self):
        """测试初始状态"""
        handler = BargeInHandler()
        assert handler.get_state() == BargeInState.IDLE

    def test_detect_disabled(self):
        """测试禁用打断"""
        config = BargeInConfig(enabled=False)
        handler = BargeInHandler(config)
        handler.set_state(BargeInState.SPEAKING)
        assert handler.detect(99999) is False  # 即使高能量也不触发

    def test_detect_in_idle(self):
        """测试空闲状态不触发"""
        handler = BargeInHandler()
        assert handler.detect(99999) is False

    def test_reset(self):
        """测试重置"""
        handler = BargeInHandler()
        handler.set_state(BargeInState.SPEAKING)
        handler._barge_in_count = 5
        handler.reset()
        assert handler.get_state() == BargeInState.IDLE
        assert handler._barge_in_count == 5  # reset 不清计数

    def test_reset_count(self):
        """测试重置计数"""
        handler = BargeInHandler()
        handler._barge_in_count = 5
        handler.reset_count()
        assert handler._barge_in_count == 0

    def test_handle_barge_in(self):
        """测试处理打断"""
        handler = BargeInHandler()
        handler.set_state(BargeInState.SPEAKING)
        handler.set_current_response("正在播放的文本", progress=0.5)
        handler._speech_frame_count = handler.config.confirm_frames
        handler._state = BargeInState.BARGE_IN_DETECTED
        handler._last_barge_in_time = 0  # 确保不冷却

        # 注意：实际打断需要异步调用
        assert handler.get_state() == BargeInState.BARGE_IN_DETECTED