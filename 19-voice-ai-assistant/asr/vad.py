"""VAD (Voice Activity Detection) 语音活动检测 - 基于能量阈值和零交叉率"""

import asyncio
import struct
import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class VADState(str, Enum):
    """VAD 状态"""
    SILENCE = "silence"          # 静音
    SPEECH = "speech"            # 语音
    POSSIBLE_SPEECH = "possible_speech"  # 可能是语音（过渡态）
    POSSIBLE_SILENCE = "possible_silence"  # 可能是静音（过渡态）


@dataclass
class VADConfig:
    """VAD 配置"""
    sample_rate: int = 16000
    # 能量阈值
    energy_threshold: float = 500.0
    # 零交叉率阈值
    zcr_threshold: float = 0.15
    # 语音起始需要的连续帧数
    speech_start_frames: int = 3
    # 静音起始需要的连续帧数（用于端点检测）
    silence_end_frames: int = 15
    # 帧长 (ms)
    frame_length_ms: int = 20
    # 帧移 (ms)
    frame_shift_ms: int = 10
    # 预加重系数
    pre_emphasis: float = 0.97


@dataclass
class VADFrame:
    """VAD 分析帧"""
    index: int
    is_speech: bool
    energy: float
    zcr: float
    state: VADState
    start_sample: int
    end_sample: int


@dataclass
class VADSegment:
    """VAD 检测到的语音段"""
    start_ms: float
    end_ms: float
    frames: list = field(default_factory=list)
    avg_energy: float = 0.0


class VADDetector:
    """基于能量阈值和零交叉率的 VAD 检测器"""

    def __init__(self, config: Optional[VADConfig] = None):
        self.config = config or VADConfig()
        self._state = VADState.SILENCE
        self._speech_frame_count = 0
        self._silence_frame_count = 0
        self._frames: list[VADFrame] = []
        self._segments: list[VADSegment] = []
        self._current_segment_frames: list[VADFrame] = []
        self._segment_start_ms: Optional[float] = None

    def _bytes_to_samples(self, audio_data: bytes, sample_width: int = 2) -> list[float]:
        """将字节音频数据转为浮点样本值"""
        samples = []
        if sample_width == 2:
            fmt = '<' + 'h' * (len(audio_data) // 2)
            try:
                int_samples = struct.unpack(fmt, audio_data[:len(audio_data) - (len(audio_data) % 2)])
                samples = [s / 32768.0 for s in int_samples]
            except struct.error:
                pass
        elif sample_width == 1:
            for b in audio_data:
                samples.append((b - 128) / 128.0)
        return samples

    def _compute_energy(self, samples: list[float]) -> float:
        """计算帧能量 (RMS)"""
        if not samples:
            return 0.0
        return math.sqrt(sum(s * s for s in samples) / len(samples)) * 32768.0

    def _compute_zcr(self, samples: list[float]) -> float:
        """计算零交叉率"""
        if len(samples) < 2:
            return 0.0
        crossings = 0
        for i in range(1, len(samples)):
            if (samples[i] >= 0) != (samples[i - 1] >= 0):
                crossings += 1
        return crossings / (len(samples) - 1)

    def _pre_emphasis(self, samples: list[float]) -> list[float]:
        """预加重处理"""
        if not samples:
            return samples
        emphasized = [samples[0]]
        for i in range(1, len(samples)):
            emphasized.append(samples[i] - self.config.pre_emphasis * samples[i - 1])
        return emphasized

    def _frame_audio(self, samples: list[float]) -> list[tuple[int, int, list[float]]]:
        """将样本分帧

        Returns:
            list of (start_sample_idx, end_sample_idx, frame_samples)
        """
        frame_length = int(self.config.sample_rate * self.config.frame_length_ms / 1000)
        frame_shift = int(self.config.sample_rate * self.config.frame_shift_ms / 1000)

        frames = []
        start = 0
        idx = 0
        while start + frame_length <= len(samples):
            end = start + frame_length
            frames.append((start, end, samples[start:end]))
            start += frame_shift
            idx += 1
        return frames

    def _is_speech_frame(self, energy: float, zcr: float) -> bool:
        """判断是否为语音帧

        使用能量 + 零交叉率联合判断:
 - 高能量 -> 语音
        - 低能量 + 高零交叉率 -> 可能是噪声
        - 低能量 + 低零交叉率 -> 静音
        """
        if energy > self.config.energy_threshold:
            return True
        # 低能量时，高零交叉率可能是噪声
        if energy > self.config.energy_threshold * 0.3 and zcr < self.config.zcr_threshold:
            return True
        return False

    def _update_state(self, is_speech: bool) -> VADState:
        """基于连续帧计数更新 VAD 状态"""
        if is_speech:
            self._speech_frame_count += 1
            self._silence_frame_count = 0
            if self._speech_frame_count >= self.config.speech_start_frames:
                self._state = VADState.SPEECH
            else:
                self._state = VADState.POSSIBLE_SPEECH
        else:
            self._silence_frame_count += 1
            self._speech_frame_count = 0
            if self._silence_frame_count >= self.config.silence_end_frames:
                self._state = VADState.SILENCE
            else:
                self._state = VADState.POSSIBLE_SILENCE
        return self._state

    def process_frame(self, audio_data: bytes, sample_width: int = 2) -> VADFrame:
        """处理单帧音频数据

        Args:
            audio_data: 一帧的 PCM 音频数据
            sample_width: 采样位宽 (bytes)

        Returns:
            VADFrame: 分析结果
        """
        samples = self._bytes_to_samples(audio_data, sample_width)
        samples = self._pre_emphasis(samples)

        energy = self._compute_energy(samples)
        zcr = self._compute_zcr(samples)
        is_speech = self._is_speech_frame(energy, zcr)
        state = self._update_state(is_speech)

        frame = VADFrame(
            index=len(self._frames),
            is_speech=is_speech,
            energy=energy,
            zcr=zcr,
            state=state,
            start_sample=len(self._frames) * int(self.config.sample_rate * self.config.frame_shift_ms / 1000),
            end_sample=len(self._frames) * int(self.config.sample_rate * self.config.frame_shift_ms / 1000) + int(self.config.sample_rate * self.config.frame_length_ms / 1000),
        )

        self._frames.append(frame)
        self._handle_segment(frame)
        return frame

    def _handle_segment(self, frame: VADFrame):
        """处理语音段"""
        if frame.state == VADState.SPEECH:
            if self._segment_start_ms is None:
                self._segment_start_ms = frame.start_sample / self.config.sample_rate * 1000
            self._current_segment_frames.append(frame)
        elif frame.state == VADState.SILENCE and self._current_segment_frames:
            # 语音段结束
            end_ms = frame.start_sample / self.config.sample_rate * 1000
            avg_energy = sum(f.energy for f in self._current_segment_frames) / len(self._current_segment_frames)
            self._segments.append(VADSegment(
                start_ms=self._segment_start_ms or 0,
                end_ms=end_ms,
                frames=list(self._current_segment_frames),
                avg_energy=avg_energy,
            ))
            self._current_segment_frames = []
            self._segment_start_ms = None

    def process(self, audio_data: bytes, sample_width: int = 2) -> list[VADSegment]:
        """处理完整音频数据

        Args:
            audio_data: PCM 音频数据
            sample_width: 采样位宽

        Returns:
            检测到的语音段列表
        """
        self.reset()
        samples = self._bytes_to_samples(audio_data, sample_width)
        samples = self._pre_emphasis(samples)
        frames = self._frame_audio(samples)

        for start, end, frame_samples in frames:
            energy = self._compute_energy(frame_samples)
            zcr = self._compute_zcr(frame_samples)
            is_speech = self._is_speech_frame(energy, zcr)
            state = self._update_state(is_speech)

            frame = VADFrame(
                index=len(self._frames),
                is_speech=is_speech,
                energy=energy,
                zcr=zcr,
                state=state,
                start_sample=start,
                end_sample=end,
            )
            self._frames.append(frame)
            self._handle_segment(frame)

        # 处理最后一个未关闭的语音段
        if self._current_segment_frames:
            last_frame = self._current_segment_frames[-1]
            end_ms = last_frame.end_sample / self.config.sample_rate * 1000
            avg_energy = sum(f.energy for f in self._current_segment_frames) / len(self._current_segment_frames)
            self._segments.append(VADSegment(
                start_ms=self._segment_start_ms or 0,
                end_ms=end_ms,
                frames=list(self._current_segment_frames),
                avg_energy=avg_energy,
            ))
            self._current_segment_frames = []

        return self._segments

    def is_speaking(self) -> bool:
        """当前是否在说话"""
        return self._state in (VADState.SPEECH, VADState.POSSIBLE_SPEECH)

    def get_state(self) -> VADState:
        """获取当前 VAD 状态"""
        return self._state

    def get_segments(self) -> list[VADSegment]:
        """获取已检测到的语音段"""
        return list(self._segments)

    def reset(self):
        """重置 VAD 状态"""
        self._state = VADState.SILENCE
        self._speech_frame_count = 0
        self._silence_frame_count = 0
        self._frames = []
        self._segments = []
        self._current_segment_frames = []
        self._segment_start_ms = None