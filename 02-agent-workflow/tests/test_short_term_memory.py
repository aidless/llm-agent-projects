# ============================================
# 短期记忆单元测试
# ============================================

import pytest
from memory.short_term import ShortTermMemory


class TestShortTermMemory:
    """短期记忆测试类"""

    def test_init_default(self):
        """测试默认初始化"""
        mem = ShortTermMemory()
        assert mem.max_turns == 20
        assert mem.get_turn_count() == 0

    def test_init_custom_max_turns(self):
        """测试自定义最大轮数"""
        mem = ShortTermMemory(max_turns=5)
        assert mem.max_turns == 5

    def test_add_and_get_messages(self):
        """测试添加和获取消息"""
        mem = ShortTermMemory()
        mem.add_user_message("你好")
        mem.add_assistant_message("你好！有什么可以帮助你的？")

        history = mem.get_history()
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["content"] == "你好"
        assert history[1]["role"] == "assistant"

    def test_get_last_n_messages(self):
        """测试获取最近 N 条消息"""
        mem = ShortTermMemory()
        for i in range(10):
            mem.add_user_message(f"消息 {i}")

        last_3 = mem.get_history(last_n=3)
        assert len(last_3) == 3
        assert last_3[0]["content"] == "消息 7"

    def test_clear(self):
        """测试清空记忆"""
        mem = ShortTermMemory()
        mem.add_user_message("测试")
        assert mem.get_turn_count() == 1

        mem.clear()
        assert mem.get_turn_count() == 0
        assert len(mem.get_history()) == 0

    def test_max_turns_limit(self):
        """测试最大轮数限制"""
        mem = ShortTermMemory(max_turns=3)
        # 添加超过限制的消息
        for i in range(10):
            mem.add_user_message(f"用户消息 {i}")
            mem.add_assistant_message(f"助手回复 {i}")

        # 应该只保留最近 3 轮（6 条消息）
        assert len(mem.get_history()) == 6

    def test_metadata(self):
        """测试元数据存储"""
        mem = ShortTermMemory()
        mem.set_metadata("user_name", "测试用户")
        assert mem.get_metadata("user_name") == "测试用户"
        assert mem.get_metadata("nonexistent") is None

    def test_to_dict_and_from_dict(self):
        """测试序列化和反序列化"""
        mem = ShortTermMemory(max_turns=5)
        mem.add_user_message("序列化测试")
        mem.set_metadata("key", "value")

        data = mem.to_dict()
        assert data["type"] == "short_term"
        assert data["max_turns"] == 5
        assert data["turn_count"] == 1

        # 反序列化
        restored = ShortTermMemory.from_dict(data)
        assert restored.max_turns == 5
        assert restored.get_turn_count() == 1
        assert restored.get_metadata("key") == "value"

    def test_tool_messages(self):
        """测试工具消息类型"""
        mem = ShortTermMemory()
        mem.add_tool_message("工具调用结果")
        assert mem.get_history()[-1]["role"] == "tool"

    def test_repr(self):
        """测试字符串表示"""
        mem = ShortTermMemory(max_turns=10)
        assert "ShortTermMemory" in repr(mem)
        assert "max=10" in repr(mem)
