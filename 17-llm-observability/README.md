# LLM 应用全链路可观测性平台

模拟 OpenTelemetry + LangSmith/LangFuse 的核心功能，提供 Trace/Span/Metrics/Log 全栈观测。

> **📋 Project Card (2026-07-22)** — for hiring managers / reviewers
>
> | **Problem** | LLM apps have opaque failures: which prompt? which tool? at what cost? Without tracing, debugging is guesswork. |
> |---|---|
> | **Solution** | OpenTelemetry-style trace model (Trace → Span → Event) with LLM-specific attributes (prompt/completion/tokens/cost/latency). SQLite persistence + REST API + cost analytics. |
> | **Evidence** | 6 typed spans (`llm.call`, `tool.execute`, `retrieval.query`, etc.); real Prometheus-style metrics (counter/gauge/histogram); trace-context propagation works across FastAPI middleware. |
> | **Limitations** | No OTLP wire-protocol export (uses in-process collectors only); UI is plain REST/Swagger, not a Grafana-style dashboard; sampling is naive (every trace). |
> | **Stack** | FastAPI · SQLite · Prometheus client conventions · asyncio |
> | **Lines / Tests** | ~3.5K Python · 89 tests (2026-11-01 实测; 此前 storage/ 被 gitignore 排除导致无法收集，已修复) |
>
> See [`results.md`](./results.md) for example traces & cost breakdown.

## 功能特性

### 分布式追踪 (Tracing)
- Trace 和 Span 模型，支持父子层级关系
- Span 类型：LLM / Tool / Agent / Retrieval / Chain / Internal
- 基于 `contextvars` 的上下文传播
- 采样策略：AlwaysOn / AlwaysOff / Ratio / ParentBased
- W3C Trace Context 规范的跨服务传播 (模拟)

### LLM Span 特化
- 捕获 LLM 请求/响应内容
- Token 统计 (prompt/completion/total)
- 模型信息、Prompt 模板版本追踪
- 流式响应 chunk 记录
- 自动成本计算

### 指标收集 (Metrics)
- **Counter**: 请求总数 / 错误数 / Token 消耗
- **Histogram**: 延迟分布 / P50 / P95 / P99
- **Gauge**: 当前并发 / 队列深度
- LLM 专用指标：Token 吞吐 / 成本
- 时间窗口聚合：1m / 5m / 15m / 1h

### 日志管理 (Logging)
- 结构化 JSON 日志
- 自动关联 Trace ID / Span ID
- 按级别、trace_id、消息内容检索过滤

### 评估和反馈
- 用户反馈收集 (thumbs up/down / 评分 / 纠正)
- 反馈统计
- 微调数据集导出 (JSONL)

### 分析 API
- Trace 查询和过滤
- 延迟分析 (P50/P95/P99)
- 错误率分析
- 成本分析
- 使用趋势

## 快速启动

```bash
# 安装依赖
pip install -r requirements.txt

# 启动服务
uvicorn app.main:app --reload --port 8000

# 运行测试
pytest tests/ -v
```

## 证明什么 / 不证明什么

- **证明**：Trace→Span→Event 模型与 W3C 风格上下文传播可跨 FastAPI 中间件工作；
  6 类 LLM 特化 span、Prometheus 风格指标（counter/gauge/histogram）与成本分析可真实
  运行（89 条测试 2026-11-01 实测通过）。
- **不证明**：没有 OTLP wire-protocol 导出（只用进程内 collector），UI 是 REST/Swagger
  而非 Grafana 仪表盘；采样是朴素的全采样。这些是实现边界，不是缺陷。
- **可复现**：`pytest tests/ -v` 从项目根直接运行；测试不依赖真实 API key。

> **2026-11-01 修复**：`storage/memory_store.py` 曾被 `.gitignore` 的 `storage/` 规则
> 排除而从未入库，fresh clone 后项目无法 import、测试全部无法收集。已将 `.gitignore`
> 收窄为只忽略运行时产物，并把 `storage/` 源码模块纳入版本控制。修复不改变任何
> 观测语义（仅让已有代码可运行）。

## 项目结构

```
app/                    # FastAPI 应用
  api/                  # API 路由 (traces/metrics/logs/feedback)
  models.py             # Pydantic 数据模型
  main.py               # 应用入口
tracing/                # 追踪核心
  tracer.py             # Tracer 实现
  span.py               # Span 模型
  context.py            # contextvars 上下文传播
  sampler.py            # 采样策略
  propagator.py         # 跨服务传播
instrumentation/        # 埋点模块
  llm.py                # LLM 调用埋点
  langchain_.py         # LangChain 埋点 (模拟)
  agent.py              # Agent 埋点
  rag.py                # RAG 埋点
metrics/                # 指标系统
  collector.py          # Counter/Histogram/Gauge
  registry.py           # 指标注册中心
  aggregation.py        # 时间窗口聚合
logging/                # 日志系统
  logger.py             # LLM 专用 Logger
  handler.py            # 结构化日志 Handler
storage/                # 存储层
  memory_store.py       # 内存存储
evaluation/             # 评估系统
  feedback.py           # 反馈收集
  exporter.py           # 数据导出
tests/                  # 测试
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/v1/traces` | 查询 Trace 列表 |
| GET | `/api/v1/traces/{trace_id}` | Trace 详情 |
| GET | `/api/v1/traces/{trace_id}/spans` | Trace 下的 Span |
| GET | `/api/v1/traces/analysis/latency` | 延迟分析 |
| GET | `/api/v1/traces/analysis/errors` | 错误率分析 |
| GET | `/api/v1/traces/analysis/cost` | 成本分析 |
| GET | `/api/v1/traces/analysis/trends` | 使用趋势 |
| GET | `/api/v1/metrics` | 指标列表 |
| GET | `/api/v1/metrics/{name}` | 指标详情 |
| GET | `/api/v1/metrics/aggregate/{window}` | 时间窗口聚合 |
| GET | `/api/v1/logs` | 日志查询 |
| POST | `/api/v1/feedback` | 创建反馈 |
| GET | `/api/v1/feedback` | 反馈列表 |
| GET | `/api/v1/feedback/stats` | 反馈统计 |
| GET | `/api/v1/feedback/export/finetuning` | 导出微调数据 |

## 使用示例

```python
from tracing.tracer import get_tracer, Tracer
from tracing.span import SpanKind
from instrumentation.llm import trace_llm, trace_llm_call
from instrumentation.agent import trace_agent, trace_tool_call

# 方式1: 上下文管理器
tracer = Tracer("my-service")
with tracer.start_span("operation", kind=SpanKind.LLM) as span:
    span.set_attribute("key", "value")
    span.set_ok()

# 方式2: LLM 埋点装饰器
@trace_llm(model="gpt-4")
def call_llm(prompt: str):
    return {"llm_output": "Hello!", "prompt_tokens": 10, "completion_tokens": 5}

# 方式3: Agent + Tool 埋点
with trace_agent("ResearchAgent", agent_type="react") as agent_span:
    with trace_tool_call("search", {"query": "hello"}) as tool_span:
        tool_span.set_attribute("tool.output", "result")
```

## Docker 部署

```bash
docker-compose up --build
```