"""对话管理器 - 管理对话流程、意图识别和 LLM 交互"""

import asyncio
import time
import logging
from typing import Any, AsyncIterator, Optional, Callable, Awaitable

from dialog.context import DialogContext, IntentResult, MessageRole
from dialog.barge_in import BargeInHandler, BargeInEvent, BargeInState

logger = logging.getLogger(__name__)

# 简单的意图模式匹配
INTENT_PATTERNS = {
    "greeting": ["你好", "嗨", "hello", "hi", "早上好", "晚上好", "下午好"],
    "farewell": ["再见", "拜拜", "bye", "goodbye", "下次见"],
    "weather": ["天气", "温度", "下雨", "晴天", "weather"],
    "music": ["音乐", "歌曲", "播放", "music", "play", "唱"],
    "alarm": ["闹钟", "提醒", "定时", "alarm", "timer", "reminder"],
    "time": ["几点", "时间", "日期", "what time", "what date"],
    "thanks": ["谢谢", "感谢", "thank", "thanks"],
}


class MockLLMClient:
    """Mock LLM 客户端，用于测试"""

    RESPONSES = {
        "greeting": "你好！我是语音AI助手，有什么可以帮助你的吗？",
        "farewell": "再见！期待下次与你交流。",
        "weather": "今天天气晴朗，温度适宜，最高温度25度，最低温度18度。",
        "music": "好的，我来为你播放音乐。你想听什么类型的？",
        "alarm": "好的，已经为你设置好了闹钟。还有其他需要吗？",
        "time": "现在是{time}。",
        "thanks": "不客气！如果还有其他问题，随时问我。",
        "default": "我理解了你说的话。让我想想...这是一个很有趣的话题。作为你的AI助手，我会尽力帮助你。",
    }

    async def generate(
        self,
        messages: list[dict],
        stream: bool = False,
    ) -> AsyncIterator[str] | str:
        """生成回复

        Args:
            messages: 消息列表 (OpenAI 格式)
            stream: 是否流式输出

        Returns:
            str 或 AsyncIterator[str]: 回复文本
        """
        await asyncio.sleep(0.02)

        # 获取最后一条用户消息
        last_user_msg = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                last_user_msg = msg["content"].lower()
                break

        # 匹配意图
        intent = "default"
        for key, patterns in INTENT_PATTERNS.items():
            for pattern in patterns:
                if pattern in last_user_msg:
                    intent = key
                    break
            if intent != "default":
                break

        response = self.RESPONSES.get(intent, self.RESPONSES["default"])
        if intent == "time":
            from datetime import datetime
            response = response.format(time=datetime.now().strftime("%H:%M"))

        if stream:
            return self._stream_response(response)
        return response

    async def _stream_response(self, text: str) -> AsyncIterator[str]:
        """模拟流式输出，逐字/逐词返回"""
        # 按标点分句，然后逐句返回
        import re
        chunks = re.split(r'(?<=[。！？!?\n])', text)
        for chunk in chunks:
            if chunk.strip():
                await asyncio.sleep(0.01)
                yield chunk.strip()


class DialogManager:
    """对话管理器

    协调对话上下文、意图识别、LLM 调用和打断处理。
    """

    def __init__(
        self,
        llm_client: Optional[MockLLMClient] = None,
        system_prompt: str = "你是一个友好的语音AI助手，请用简洁自然的语言回答用户的问题。",
        max_context_messages: int = 10,
        dialog_timeout: float = 300.0,
    ):
        self.llm_client = llm_client or MockLLMClient()
        self.system_prompt = system_prompt
        self.max_context_messages = max_context_messages
        self.dialog_timeout = dialog_timeout
        self._contexts: dict[str, DialogContext] = {}
        self._barge_in_handlers: dict[str, BargeInHandler] = {}

    def get_or_create_context(self, session_id: str) -> DialogContext:
        """获取或创建对话上下文"""
        if session_id not in self._contexts:
            self._contexts[session_id] = DialogContext()
            self._barge_in_handlers[session_id] = BargeInHandler()
        return self._contexts[session_id]

    def get_barge_in_handler(self, session_id: str) -> BargeInHandler:
        """获取打断处理器"""
        if session_id not in self._barge_in_handlers:
            self._barge_in_handlers[session_id] = BargeInHandler()
        return self._barge_in_handlers[session_id]

    def recognize_intent(self, text: str) -> IntentResult:
        """简单的意图识别（基于关键词匹配）

        Args:
            text: 用户输入文本

        Returns:
            IntentResult: 意图识别结果
        """
        text_lower = text.lower()

        for intent, patterns in INTENT_PATTERNS.items():
            for pattern in patterns:
                if pattern.lower() in text_lower:
                    return IntentResult(
                        intent=intent,
                        confidence=0.9,
                        raw_text=text,
                    )

        return IntentResult(
            intent="general",
            confidence=0.5,
            raw_text=text,
        )

    def _prepare_context(self, session_id: str, user_text: str) -> tuple:
        """准备对话上下文，返回 (context, intent_result, llm_messages)"""
        context = self.get_or_create_context(session_id)

        # 检查超时
        if context.is_expired(self.dialog_timeout):
            context.clear()

        # 检查最大轮数
        if context.is_max_turns():
            context.clear()

        # 添加用户消息
        context.add_message(MessageRole.USER, user_text)

        # 意图识别
        intent_result = self.recognize_intent(user_text)
        context.add_intent(intent_result)

        # 构建 LLM 上下文
        llm_messages = context.get_context_for_llm(
            system_prompt=self.system_prompt,
            max_tokens=2000,
        )

        return context, intent_result, llm_messages

    async def process_text_sync(self, session_id: str, user_text: str) -> dict:
        """非流式处理用户文本

        Args:
            session_id: 会话 ID
            user_text: 用户文本

        Returns:
            处理结果字典
        """
        context, intent_result, llm_messages = self._prepare_context(session_id, user_text)

        response = await self.llm_client.generate(llm_messages, stream=False)
        context.add_message(MessageRole.ASSISTANT, response)

        return {
            "type": "text_response",
            "text": response,
            "intent": intent_result.intent,
            "confidence": intent_result.confidence,
            "session_id": session_id,
            "turn_count": context.get_turn_count(),
            "is_final": True,
        }

    async def process_text_stream(self, session_id: str, user_text: str) -> AsyncIterator[dict]:
        """流式处理用户文本

        Args:
            session_id: 会话 ID
            user_text: 用户文本

        Yields:
            流式处理结果字典
        """
        context, intent_result, llm_messages = self._prepare_context(session_id, user_text)

        full_response = ""
        async for chunk in await self.llm_client.generate(llm_messages, stream=True):
            full_response += chunk
            yield {
                "type": "stream_chunk",
                "text": chunk,
                "intent": intent_result.intent,
                "session_id": session_id,
                "is_final": False,
            }

        # 添加助手消息
        context.add_message(MessageRole.ASSISTANT, full_response)
        yield {
            "type": "stream_end",
            "text": full_response,
            "intent": intent_result.intent,
            "session_id": session_id,
            "is_final": True,
        }

    def remove_session(self, session_id: str):
        """移除会话"""
        self._contexts.pop(session_id, None)
        self._barge_in_handlers.pop(session_id, None)

    def cleanup_expired(self, timeout: Optional[float] = None):
        """清理超时会话"""
        timeout = timeout or self.dialog_timeout
        expired = [
            sid for sid, ctx in self._contexts.items()
            if ctx.is_expired(timeout)
        ]
        for sid in expired:
            self.remove_session(sid)
        return len(expired)