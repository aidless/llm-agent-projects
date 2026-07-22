# Voice AI Assistant

语音 AI 助手 - 集成 ASR (语音识别) + LLM (大语言模型) + TTS (语音合成) 的全链路语音对话系统，支持 WebSocket 流式通信。

## 架构

```
音频输入 -> ASR (语音识别) -> 文本 -> LLM (对话生成) -> 回复文本 -> TTS (语音合成) -> 音频输出
```

## 技术栈

- Python 3.11+, FastAPI, WebSocket, asyncio
- ASR/TTS 使用抽象接口设计，Mock 实现用于测试
- VAD 基于能量阈值和零交叉率
- Pipeline 支持异步并行处理

## 项目结构

```
app/            FastAPI 应用入口和 API 路由
asr/            ASR 引擎抽象基类、Mock 实现、音频预处理、VAD
tts/            TTS 引擎抽象基类、Mock 实现、SSML 解析
dialog/         对话管理、上下文、打断处理
pipeline/       语音处理流水线、流管理
websocket/      WebSocket 处理器、会话管理
tests/          测试
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

# 运行测试
pytest tests/ -v
```

## Docker 部署

```bash
docker-compose up --build
```

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /health | 健康检查 |
| POST | /asr/recognize | 上传音频文件进行识别 |
| POST | /asr/recognize/base64 | Base64 音频识别 |
| GET | /asr/formats | 获取支持的音频格式 |
| POST | /tts/synthesize | 文本转语音（返回音频） |
| POST | /tts/synthesize/info | 文本转语音（返回元数据） |
| GET | /tts/voices | 获取可用音色列表 |
| POST | /chat/message | 文本聊天 |
| POST | /chat/message/stream | 文本聊天（SSE 流式） |
| DELETE | /chat/session/{id} | 删除会话 |
| WS | /ws/voice | WebSocket 语音对话 |

## WebSocket 协议

### 客户端 -> 服务端

- `audio_data`: 二进制音频数据
- `audio_start`: 开始发送音频
- `audio_end`: 音频发送结束，触发处理
- `text_input`: 文本输入 `{"type":"text_input","text":"..."}`
- `control`: 控制命令 `{"type":"control","command":"stop|reset"}`
- `heartbeat`: 心跳

### 服务端 -> 客户端

- `asr_partial`: ASR 中间结果
- `asr_result`: ASR 最终结果
- `llm_chunk`: LLM 生成的文本块
- `llm_complete`: LLM 生成完成
- `tts_audio`: TTS 音频数据（二进制）
- `tts_end`: TTS 发送结束
- `heartbeat_ack`: 心跳确认

## 核心特性

1. **异步流水线**: ASR 完成即开始 LLM，LLM 生成即开始 TTS
2. **VAD 语音检测**: 基于能量阈值和零交叉率的端点检测
3. **打断处理**: 支持 barge-in，用户可在 TTS 播放期间打断
4. **多轮对话**: 对话上下文管理、意图识别、实体跟踪
5. **流式通信**: WebSocket 双向流，SSE 文本流
6. **SSML 支持**: TTS 支持语速、音调、音量、停顿等标记