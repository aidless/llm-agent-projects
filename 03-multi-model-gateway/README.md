# 多模型智能路由网关

统一的 LLM API 网关，兼容 OpenAI API 格式，支持多模型智能路由、成本优化、失败自动降级和实时监控。

## 支持的模型

| 模型 | 服务商 | 擅长场景 |
|------|--------|----------|
| GPT-4o | OpenAI | 通用对话、数据分析、推理 |
| DeepSeek V4 | DeepSeek | 代码生成、编程 |
| Qwen Plus | 阿里云 | 翻译、文本处理、高性价比 |
| Claude Sonnet 4 | Anthropic | 创意写作、长文本 |

## 核心功能

### 统一 API 接口
兼容 OpenAI Chat Completions API 格式，现有 OpenAI SDK 客户端只需更改 `base_url` 即可接入。

### 四种智能路由策略

1. **按任务类型路由** (`task_type`，默认)
   - 代码生成 → DeepSeek
   - 创意写作 → Claude
   - 数据分析 → GPT-4o
   - 翻译 → Qwen
   - 通用 → GPT-4o

2. **按成本优化路由** (`cost`)
   - 简单任务自动选择低成本模型（DeepSeek > Qwen）
   - 复杂任务选择高能力模型（Claude > GPT-4o）

3. **按延迟优化路由** (`latency`)
   - 根据历史延迟数据选择响应最快的模型

4. **负载均衡路由** (`load_balance`)
   - 轮询分配请求到各模型

### 失败自动降级
主模型调用失败时，自动尝试备用模型，最多降级 3 次。

### 并发控制
使用 `asyncio.Semaphore` 限制最大并发请求数，防止上游服务过载。

### 多 API Key 轮询
每个模型支持配置多个 API Key，自动轮询以避免触发单 Key 速率限制。

### 实时监控
- `/health` - 健康检查
- `/stats` - 详细统计（请求量、Token、成本、延迟 P50/P99）
- `/stats/recent` - 最近请求记录
- `/stats/{model}` - 单模型统计
- `/strategies` - 可用路由策略

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

复制 `.env` 文件并填入真实的 API Key：

```bash
cp .env .env.local
# 编辑 .env.local，填入各模型的 API Key
```

### 3. 启动服务

```bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 4. 发送请求

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "写一个快速排序"}],
    "strategy": "task_type"
  }'
```

### 5. 指定策略

```bash
# 成本优化
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "messages": [{"role": "user", "content": "你好"}],
    "strategy": "cost"
  }'

# 直接指定模型
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek",
    "messages": [{"role": "user", "content": "写代码"}]
  }'
```

## 项目结构

```
03-multi-model-gateway/
├── app/
│   ├── __init__.py
│   ├── main.py           # FastAPI 主应用，所有 API 端点
│   ├── config.py          # 配置管理，从 .env 加载
│   └── models.py          # 数据模型，兼容 OpenAI 格式
├── routers/
│   ├── __init__.py
│   ├── base.py            # 适配器基类
│   ├── manager.py         # 适配器管理器
│   ├── openai_adapter.py  # GPT-4o 适配器
│   ├── deepseek_adapter.py # DeepSeek 适配器
│   ├── qwen_adapter.py    # Qwen 适配器
│   └── claude_adapter.py  # Claude 适配器
├── strategy/
│   ├── __init__.py
│   ├── task_classifier.py # 任务类型分类器
│   └── router.py          # 智能路由器（四种策略）
├── reqqueue/
│   ├── __init__.py
│   └── request_queue.py   # 请求队列、并发控制、降级管理
├── monitor/
│   ├── __init__.py
│   └── stats_collector.py # 监控统计收集器
├── tests/
│   ├── __init__.py
│   └── test_gateway.py    # 完整测试套件
├── .env                    # 环境变量模板
├── requirements.txt
├── Dockerfile
└── README.md
```

## 使用 Docker

```bash
# 构建镜像
docker build -t llm-gateway .

# 运行容器
docker run -d -p 8000:8000 --env-file .env llm-gateway
```

## 运行测试

```bash
pytest tests/ -v
```
