"""SSML 解析器 - 将 SSML 标记转换为 TTS 可用的参数"""

import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SSMLSegment:
    """SSML 解析后的文本段"""
    text: str
    speed: float = 1.0
    pitch: float = 1.0
    volume: float = 1.0
    pause_ms: int = 0  # 段前停顿 (毫秒)
    voice_id: Optional[str] = None
    emphasis: Optional[str] = None  # "strong", "moderate", "reduced", "none"


@dataclass
class SSMLParseResult:
    """SSML 解析结果"""
    segments: list[SSMLSegment] = field(default_factory=list)
    has_ssml: bool = False


class SSMLParser:
    """SSML 解析器

    支持的 SSML 标签:
    - <speak> 根元素
    - <break time="500ms"/> 停顿
    - <prosody rate="fast" pitch="+10%" volume="loud"> 语速/音调/音量
    - <voice name="xxx"> 切换音色
    - <emphasis level="strong"> 强调
    - <say-as interpret-as="..."> 特殊解读方式
    """

    # rate -> speed 映射
    RATE_MAP = {
        "x-slow": 0.5,
        "slow": 0.75,
        "medium": 1.0,
        "normal": 1.0,
        "fast": 1.25,
        "x-fast": 1.5,
    }

    # pitch 百分比映射
    PITCH_MAP = {
        "x-low": 0.6,
        "low": 0.8,
        "medium": 1.0,
        "normal": 1.0,
        "high": 1.2,
        "x-high": 1.4,
    }

    # volume 映射
    VOLUME_MAP = {
        "silent": 0.0,
        "x-soft": 0.3,
        "soft": 0.5,
        "medium": 0.7,
        "normal": 0.8,
        "loud": 1.0,
        "x-loud": 1.0,
    }

    # emphasis -> volume 调整
    EMPHASIS_MAP = {
        "strong": 1.0,
        "moderate": 0.9,
        "reduced": 0.7,
        "none": 0.8,
    }

    def is_ssml(self, text: str) -> bool:
        """判断文本是否包含 SSML 标记"""
        return bool(re.search(r'<speak[\s>]', text.strip()))

    def parse(self, text: str) -> SSMLParseResult:
        """解析 SSML 文本

        Args:
            text: SSML 或纯文本

        Returns:
            SSMLParseResult: 解析结果
        """
        if not self.is_ssml(text):
            return SSMLParseResult(
                segments=[SSMLSegment(text=text)],
                has_ssml=False,
            )

        try:
            root = ET.fromstring(text)
            segments = self._parse_element(root)
            return SSMLParseResult(segments=segments, has_ssml=True)
        except ET.ParseError:
            # 解析失败，退回纯文本
            clean_text = re.sub(r'<[^>]+>', '', text)
            return SSMLParseResult(
                segments=[SSMLSegment(text=clean_text)],
                has_ssml=False,
            )

    def _parse_element(
        self,
        element: ET.Element,
        speed: float = 1.0,
        pitch: float = 1.0,
        volume: float = 1.0,
        voice_id: Optional[str] = None,
        emphasis: Optional[str] = None,
    ) -> list[SSMLSegment]:
        """递归解析 XML 元素"""
        segments = []

        for child in element:
            tag = child.tag
            pause_ms = 0

            if tag == "break":
                # 停顿标签
                pause_ms = self._parse_break(child)
                if pause_ms > 0 and segments:
                    segments[-1].pause_ms = pause_ms
                continue

            # 解析当前标签的属性
            cur_speed = speed
            cur_pitch = pitch
            cur_volume = volume
            cur_voice = voice_id
            cur_emphasis = emphasis

            if tag == "prosody":
                cur_speed = self._parse_rate(child.get("rate", ""), speed)
                cur_pitch = self._parse_pitch(child.get("pitch", ""), pitch)
                cur_volume = self._parse_volume(child.get("volume", ""), volume)
            elif tag == "voice":
                cur_voice = child.get("name")
            elif tag == "emphasis":
                cur_emphasis = child.get("level", "moderate")
                if cur_emphasis in self.EMPHASIS_MAP:
                    cur_volume = self.EMPHASIS_MAP[cur_emphasis]

            # 递归处理子元素
            child_segments = self._parse_element(
                child, cur_speed, cur_pitch, cur_volume, cur_voice, cur_emphasis
            )
            segments.extend(child_segments)

        # 提取当前元素的文本
        if element.text and element.text.strip():
            text = element.text.strip()
            # 去掉子元素导致的尾部空白
            text = re.sub(r'\s+', ' ', text).strip()
            if text:
                segments.append(SSMLSegment(
                    text=text,
                    speed=speed,
                    pitch=pitch,
                    volume=volume,
                    voice_id=voice_id,
                    emphasis=emphasis,
                ))

        # 处理尾随文本
        if element.tail and element.tail.strip():
            text = element.tail.strip()
            text = re.sub(r'\s+', ' ', text).strip()
            if text:
                segments.append(SSMLSegment(text=text))

        return segments

    def _parse_break(self, element: ET.Element) -> int:
        """解析 break 标签的停顿时长"""
        time_str = element.get("time", "0ms")
        # 支持 "500ms" / "0.5s" / "500" 格式
        match = re.match(r'([\d.]+)\s*(ms|s)?', time_str)
        if match:
            value = float(match.group(1))
            unit = match.group(2) or 'ms'
            if unit == 's':
                return int(value * 1000)
            return int(value)
        return 0

    def _parse_rate(self, rate_str: str, default: float) -> float:
        """解析 rate 属性"""
        if not rate_str:
            return default
        rate_str = rate_str.strip().lower()

        if rate_str in self.RATE_MAP:
            return self.RATE_MAP[rate_str]

        # 支持 "+10%" / "-20%" 格式
        match = re.match(r'([+-]?\d+)%', rate_str)
        if match:
            pct = int(match.group(1))
            return max(0.5, min(2.0, default * (1 + pct / 100)))

        try:
            return max(0.5, min(2.0, float(rate_str)))
        except ValueError:
            return default

    def _parse_pitch(self, pitch_str: str, default: float) -> float:
        """解析 pitch 属性"""
        if not pitch_str:
            return default
        pitch_str = pitch_str.strip().lower()

        if pitch_str in self.PITCH_MAP:
            return self.PITCH_MAP[pitch_str]

        # 支持 "+10%" / "-10st" / "+10" 格式
        match = re.match(r'([+-]?\d+)\s*(%|st)?', pitch_str)
        if match:
            value = int(match.group(1))
            return max(0.5, min(2.0, default * (1 + value / 100)))

        return default

    def _parse_volume(self, volume_str: str, default: float) -> float:
        """解析 volume 属性"""
        if not volume_str:
            return default
        volume_str = volume_str.strip().lower()

        if volume_str in self.VOLUME_MAP:
            return self.VOLUME_MAP[volume_str]

        # 支持 "+10%" 格式
        match = re.match(r'([+-]?\d+)%', volume_str)
        if match:
            pct = int(match.group(1))
            return max(0.0, min(1.0, default + pct / 100))

        return default