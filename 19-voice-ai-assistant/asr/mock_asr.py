"""Mock ASR 引擎 - 用于测试"""

import asyncio
from collections import deque
from typing import AsyncIterator, Optional

from asr.base import ASRConfig, ASREngine, ASRResult


class MockASR(ASREngine):
    """Mock ASR 引擎，根据音频数据长度生成模拟识别文本"""

    # 模拟的中文识别文本片段
    ZH_FRAGMENTS = [
        "你好", "请问", "今天天气怎么样", "帮我查一下",
        "谢谢", "再见", "我想听音乐", "设置闹钟",
        "打开灯", "明天几点", "帮我翻译", "播放新闻",
    ]

    EN_FRAGMENTS = [
        "hello", "how are you", "what time is it",
        "thank you", "goodbye", "play music",
        "set alarm", "turn on the light", "what's the weather",
        "help me translate", "play news",
    ]

    def __init__(self, config: Optional[ASRConfig] = None, fixed_text: Optional[str] = None):
        """初始化 Mock ASR

        Args:
            config: ASR 配置
            fixed_text: 固定返回的文本（用于确定性测试）
        """
        super().__init__(config)
        self.fixed_text = fixed_text
        self._buffer = deque()
        self._chunk_count = 0

    def _generate_text(self, audio_length: int) -> str:
        """根据音频长度生成模拟文本"""
        if self.fixed_text:
            return self.fixed_text

        fragments = self.ZH_FRAGMENTS if self.config.language == "zh" else self.EN_FRAGMENTS
        # 根据音频长度选择 1-2 个片段
        num_fragments = min(2, max(1, audio_length // 1000 + 1))
        selected = fragments[:num_fragments]
        return "，".join(selected)

    async def recognize(self, audio_data: bytes) -> ASRResult:
        """识别完整音频"""
        await asyncio.sleep(0.01)  # 模拟处理延迟
        text = self._generate_text(len(audio_data))
        text = self.postprocess(text)
        return ASRResult(
            text=text,
            is_final=True,
            confidence=0.95,
            language=self.config.language,
        )

    async def stream_recognize(self, audio_stream: AsyncIterator[bytes]) -> AsyncIterator[ASRResult]:
        """流式识别音频，每收到一个 chunk 返回部分结果"""
        self._buffer.clear()
        self._chunk_count = 0

        async for chunk in audio_stream:
            self._chunk_count += 1
            self._buffer.append(chunk)

            if len(chunk) > 0:
                # 生成中间结果
                partial_text = self._generate_text(len(b''.join(self._buffer)))
                partial_text = partial_text.split('，')[0] if '，' in partial_text else partial_text

                yield ASRResult(
                    text=partial_text,
                    is_final=False,
                    confidence=0.7 + self._chunk_count * 0.05,
                    language=self.config.language,
                    partial=True,
                )

        # 最终结果
        full_audio = b''.join(self._buffer)
        final_text = self._generate_text(len(full_audio))
        final_text = self.postprocess(final_text)

        yield ASRResult(
            text=final_text,
            is_final=True,
            confidence=0.95,
            language=self.config.language,
            partial=False,
        )

    def reset(self):
        """重置内部状态"""
        self._buffer.clear()
        self._chunk_count = 0