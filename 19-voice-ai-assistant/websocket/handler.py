"""WebSocket 处理器 - 处理 WebSocket 连接和消息"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from fastapi import WebSocket, WebSocketDisconnect

from pipeline.voice_pipeline import VoicePipeline
from pipeline.stream_manager import StreamEvent, StreamType
from websocket.session import SessionManager, SessionState

logger = logging.getLogger(__name__)


class MessageType(str, Enum):
    """WebSocket 消息类型"""
    # 客户端 -> 服务端
    AUDIO_DATA = "audio_data"       # 音频数据
    AUDIO_START = "audio_start"     # 开始发送音频
    AUDIO_END = "audio_end"         # 音频发送结束
    TEXT_INPUT = "text_input"       # 文本输入
    CONTROL = "control"             # 控制命令
    HEARTBEAT = "heartbeat"         # 心跳

    # 服务端 -> 客户端
    ASR_PARTIAL = "asr_partial"     # ASR 中间结果
    ASR_RESULT = "asr_result"       # ASR 最终结果
    LLM_CHUNK = "llm_chunk"         # LLM 文本块
    LLM_COMPLETE = "llm_complete"   # LLM 生成完成
    TTS_AUDIO = "tts_audio"         # TTS 音频数据
    TTS_END = "tts_end"             # TTS 发送结束
    ERROR = "error"                 # 错误
    HEARTBEAT_ACK = "heartbeat_ack" # 心跳确认
    SESSION_INFO = "session_info"   # 会话信息


class ControlCommand(str, Enum):
    """控制命令"""
    STOP = "stop"           # 停止当前处理
    RESET = "reset"         # 重置会话
    CANCEL_TTS = "cancel_tts"  # 取消 TTS 播放
    PAUSE = "pause"         # 暂停
    RESUME = "resume"       # 恢复


@dataclass
class WSConfig:
    """WebSocket 配置"""
    heartbeat_interval: float = 30.0     # 心跳间隔（秒）
    heartbeat_timeout: float = 60.0      # 心跳超时（秒）
    max_audio_chunk_size: int = 65536    # 最大音频块大小
    enable_reconnect: bool = True        # 支持重连
    reconnect_window: float = 10.0       # 重连窗口（秒）


class WebSocketHandler:
    """WebSocket 处理器

    管理 WebSocket 连接生命周期、消息路由和心跳保活。
    """

    def __init__(
        self,
        pipeline: VoicePipeline,
        session_manager: Optional[SessionManager] = None,
        config: Optional[WSConfig] = None,
    ):
        self.pipeline = pipeline
        self.session_manager = session_manager or SessionManager()
        self.config = config or WSConfig()

    async def handle_connection(self, websocket: WebSocket):
        """处理 WebSocket 连接

        Args:
            websocket: WebSocket 连接对象
        """
        await websocket.accept()

        session = self.session_manager.create_session(websocket)
        session_id = session.session_id

        logger.info(f"WebSocket connected: {session_id}")

        # 发送会话信息
        await self._send_json(websocket, {
            "type": MessageType.SESSION_INFO.value,
            "session_id": session_id,
            "state": session.state.value,
        })

        # 启动心跳任务
        heartbeat_task = asyncio.create_task(
            self._heartbeat_loop(websocket, session_id)
        )

        # 启动接收任务
        processing = False
        try:
            while True:
                try:
                    message = await asyncio.wait_for(
                        websocket.receive(),
                        timeout=self.config.heartbeat_timeout,
                    )
                except asyncio.TimeoutError:
                    logger.warning(f"Heartbeat timeout: {session_id}")
                    break

                if "bytes" in message and message["bytes"]:
                    # 二进制数据 = 音频
                    await self._handle_audio(websocket, session_id, message["bytes"])
                elif "text" in message and message["text"]:
                    # 文本消息
                    await self._handle_text_message(websocket, session_id, message["text"])

        except WebSocketDisconnect:
            logger.info(f"WebSocket disconnected: {session_id}")
        except Exception as e:
            logger.error(f"WebSocket error ({session_id}): {e}")
        finally:
            heartbeat_task.cancel()
            self.session_manager.update_state(session_id, SessionState.DISCONNECTED)
            self.pipeline.cancel_session(session_id)
            self.session_manager.remove_session(session_id)
            logger.info(f"Session cleaned up: {session_id}")

    async def _handle_text_message(self, websocket: WebSocket, session_id: str, raw_text: str):
        """处理文本消息"""
        try:
            data = json.loads(raw_text)
        except json.JSONDecodeError:
            await self._send_json(websocket, {
                "type": MessageType.ERROR.value,
                "error": "Invalid JSON",
            })
            return

        msg_type = data.get("type", "")

        if msg_type == MessageType.HEARTBEAT.value:
            await self._send_json(websocket, {
                "type": MessageType.HEARTBEAT_ACK.value,
                "timestamp": time.time(),
            })

        elif msg_type == MessageType.AUDIO_START.value:
            self.session_manager.update_state(session_id, SessionState.LISTENING)
            self.pipeline.process_audio_chunk(session_id, b'')

        elif msg_type == MessageType.AUDIO_END.value:
            self.session_manager.update_state(session_id, SessionState.PROCESSING)
            async for event in self.pipeline.finalize_audio_stream(session_id):
                await self._emit_event(websocket, event)

        elif msg_type == MessageType.TEXT_INPUT.value:
            text = data.get("text", "")
            self.session_manager.update_state(session_id, SessionState.PROCESSING)
            async for event in self.pipeline._process_text_stream(session_id, text):
                await self._emit_event(websocket, event)

        elif msg_type == MessageType.CONTROL.value:
            command = data.get("command", "")
            await self._handle_control(websocket, session_id, command)

        else:
            await self._send_json(websocket, {
                "type": MessageType.ERROR.value,
                "error": f"Unknown message type: {msg_type}",
            })

    async def _handle_audio(self, websocket: WebSocket, session_id: str, audio_data: bytes):
        """处理音频数据"""
        if len(audio_data) > self.config.max_audio_chunk_size:
            await self._send_json(websocket, {
                "type": MessageType.ERROR.value,
                "error": "Audio chunk too large",
            })
            return

        self.session_manager.update_state(session_id, SessionState.LISTENING)
        await self.pipeline.process_audio_chunk(session_id, audio_data)

    async def _handle_control(self, websocket: WebSocket, session_id: str, command: str):
        """处理控制命令"""
        if command == ControlCommand.STOP.value:
            self.pipeline.cancel_session(session_id)
            self.session_manager.update_state(session_id, SessionState.CONNECTED)
        elif command == ControlCommand.RESET.value:
            self.pipeline.cancel_session(session_id)
            self.session_manager.update_state(session_id, SessionState.CONNECTED)

    async def _emit_event(self, websocket: WebSocket, event: StreamEvent):
        """将 Pipeline 事件转换为 WebSocket 消息发送"""
        if event.type == StreamType.ASR_PARTIAL:
            await self._send_json(websocket, {
                "type": MessageType.ASR_PARTIAL.value,
                "data": event.data if isinstance(event.data, dict) else {"text": ""},
            })
        elif event.type == StreamType.ASR_FINAL:
            result = event.data
            await self._send_json(websocket, {
                "type": MessageType.ASR_RESULT.value,
                "text": result.text if hasattr(result, 'text') else str(result),
                "is_final": result.is_final if hasattr(result, 'is_final') else True,
                "confidence": result.confidence if hasattr(result, 'confidence') else 0,
            })
        elif event.type == StreamType.LLM_CHUNK:
            await self._send_json(websocket, {
                "type": MessageType.LLM_CHUNK.value,
                **event.data,
            })
        elif event.type == StreamType.LLM_COMPLETE:
            await self._send_json(websocket, {
                "type": MessageType.LLM_COMPLETE.value,
                **event.data,
            })
        elif event.type == StreamType.TTS_AUDIO_CHUNK:
            result = event.data
            if hasattr(result, 'audio_data'):
                await websocket.send_bytes(result.audio_data)
            else:
                await self._send_json(websocket, {
                    "type": MessageType.TTS_AUDIO.value,
                    "data": str(result),
                })
        elif event.type == StreamType.TTS_COMPLETE:
            await self._send_json(websocket, {
                "type": MessageType.TTS_END.value,
                **event.data,
            })
        elif event.type == StreamType.ERROR:
            await self._send_json(websocket, {
                "type": MessageType.ERROR.value,
                **event.data,
            })

    async def _heartbeat_loop(self, websocket: WebSocket, session_id: str):
        """心跳保活循环"""
        try:
            while True:
                await asyncio.sleep(self.config.heartbeat_interval)
                await self._send_json(websocket, {
                    "type": MessageType.HEARTBEAT_ACK.value,
                    "timestamp": time.time(),
                })
        except asyncio.CancelledError:
            pass
        except Exception:
            pass

    @staticmethod
    async def _send_json(websocket: WebSocket, data: dict):
        """发送 JSON 消息"""
        try:
            await websocket.send_json(data)
        except Exception:
            pass