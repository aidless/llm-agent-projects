# LLM Inference Optimization Service

模拟 vLLM/SGLang 核心优化策略的推理服务性能优化工具集。

## 功能特性

### 1. 模拟推理引擎
- Token by token 生成模拟
- 动态批处理 (Continuous Batching)
- 请求队列和优先级调度
- 多种调度策略 (FIFO/Priority/SJF/LMF)

### 2. 量化模拟
- 支持 FP32/FP16/INT8/INT4 精度
- GPTQ/AWQ/GGUF 格式配置
- 量化感知的延迟/内存模型计算
- 精度损失评估和推荐

### 3. KV Cache 管理
- KV Cache 分配与回收
- PagedAttention 分页缓存模拟
- Cache 命中率统计
- GPU 内存池管理

### 4. 性能监控
- Token 生成速率 (tokens/s)
- 首 Token 延迟 (TTFT)
- 端到端延迟 (E2E Latency)
- GPU 利用率模拟
- Prometheus 指标格式输出

### 5. API 代理层
- OpenAI 兼容 API (/v1/chat/completions, /v1/completions)
- 请求路由和负载均衡
- 令牌桶/滑动窗口限流
- 三态熔断器 (关闭/开启/半开)

## 项目结构

```
10-inference-optimization/
├── app/
│   ├── main.py                  # FastAPI 入口
│   ├── models.py                # Pydantic 模型
│   └── api/
│       ├── inference.py         # 推理 API (OpenAI 兼容)
│       ├── monitor.py           # 监控 API
│       └── config.py            # 配置管理 API
├── engine/
│   ├── simulator.py             # 推理模拟器
│   ├── tokenizer_mock.py        # 模拟分词器
│   ├── batcher.py               # 动态批处理
│   └── scheduler.py             # 请求调度
├── quantization/
│   ├── quantizer.py             # 量化模拟
│   ├── formats.py               # 量化格式定义
│   └── evaluator.py             # 精度评估
├── cache/
│   ├── kv_cache.py              # KV Cache 管理
│   ├── paged_cache.py           # PagedAttention 分页缓存
│   └── memory_manager.py        # 内存管理
├── monitoring/
│   ├── metrics.py               # 指标收集
│   ├── prometheus_exporter.py   # Prometheus 格式
│   └── dashboard.py             # 监控数据 API
├── proxy/
│   ├── router.py                # 请求路由
│   ├── rate_limiter.py          # 限流器
│   └── circuit_breaker.py       # 熔断器
├── tests/                       # 测试套件
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 运行测试

```bash
pytest tests/ -v
```

### Docker 部署

```bash
docker-compose up --build
```

## API 端点

### 推理 API
| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /v1/chat/completions | OpenAI 兼容 Chat Completions |
| POST | /v1/completions | 文本补全 |
| GET  | /v1/models | 列出可用模型 |

### 监控 API
| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | /v1/metrics | 所有指标 |
| GET  | /v1/metrics/prometheus | Prometheus 格式 |
| GET  | /v1/metrics/summary | 指标摘要 |
| GET  | /v1/dashboard/realtime | 实时统计 |
| GET  | /v1/dashboard/gpu | GPU 报告 |
| GET  | /v1/dashboard/inference | 推理报告 |

### 配置 API
| 方法 | 路径 | 说明 |
|------|------|------|
| GET  | /v1/quantization/formats | 量化格式列表 |
| GET  | /v1/quantization/compare | 量化格式对比 |
| GET  | /v1/quantization/evaluate/{fmt} | 量化精度评估 |
| GET  | /v1/quantization/recommend | 量化推荐 |
| GET  | /v1/cache/kv/stats | KV Cache 统计 |
| GET  | /v1/cache/paged/usage | 分页缓存使用 |
| GET  | /v1/cache/memory/overview | 内存概览 |
| GET  | /v1/proxy/backends | 后端列表 |
| GET  | /v1/proxy/stats | 路由统计 |
| GET  | /v1/proxy/rate_limiter/stats | 限流统计 |
| GET  | /v1/proxy/circuit_breaker/status | 熔断器状态 |

## 技术栈

- Python 3.11+
- FastAPI - Web 框架
- Pydantic - 数据验证
- NumPy - 数值计算
- asyncio - 异步编程
- pytest - 测试框架
- Docker - 容器化部署
