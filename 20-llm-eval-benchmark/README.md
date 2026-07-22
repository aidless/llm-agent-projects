# LLM Eval Benchmark Platform

> AI 模型评估基准测试平台 - 支持主流评测基准的自动化评测、模型对比和排行榜

## 项目简介

这是一个完整的 AI 大语言模型评估基准测试平台，支持 **MMLU**、**GSM8K**、**HumanEval**、**MT-Bench** 四大主流评测基准的自动化评测，并提供模型管理、综合排行榜和多格式报告生成能力。

### 核心特性

- **四大主流评测基准**: MMLU（多选题）、GSM8K（数学推理）、HumanEval（代码生成）、MT-Bench（多轮对话）
- **自定义基准支持**: 通过 JSON 格式导入自定义评测题目
- **并发评测引擎**: 基于 asyncio 的异步并发执行，支持进度实时追踪
- **完善的评分系统**: 准确率、Pass@k、F1/精确率/召回率、Wilson 置信区间、McNemar 显著性检验
- **模型管理**: 模型注册、版本管理、参数量追踪
- **排行榜系统**: 综合排名、分基准排名、模型卡片、雷达图对比
- **可视化图表**: 基于 matplotlib 的柱状图、对比图、雷达图
- **多格式报告**: JSON / Markdown / CSV 格式导出
- **RESTful API**: 完整的 FastAPI 接口，支持 Swagger 文档

---

## 技术架构

```
┌──────────────────────────────────────────────────────┐
│                    FastAPI Layer                      │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌─────────┐ │
│  │Benchmark │ │  Model   │ │Leaderboard│ │ Report  │ │
│  │   API    │ │   API    │ │   API    │ │   API   │ │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └────┬────┘ │
├───────┼────────────┼────────────┼────────────┼──────┤
│       │      Core Engine Layer  │            │       │
│  ┌────▼────────────▼────────────▼────────────▼────┐ │
│  │              Evaluator (async)                  │ │
│  │  ┌─────────┐  ┌─────────┐  ┌──────────────┐  │ │
│  │  │Scorer   │  │Comparator│ │Progress Track│  │ │
│  │  └─────────┘  └─────────┘  └──────────────┘  │ │
│  └────────────────────────────────────────────────┘ │
├──────────────────────────────────────────────────────┤
│                   Benchmark Layer                     │
│  ┌──────┐ ┌──────┐ ┌──────────┐ ┌─────────┐       │
│  │ MMLU │ │GSM8K │ │HumanEval │ │MT-Bench │       │
│  └──────┘ └──────┘ └──────────┘ └─────────┘       │
│  ┌──────────────────────────────────────────┐       │
│  │          Custom Benchmark                │       │
│  └──────────────────────────────────────────┘       │
├──────────────────────────────────────────────────────┤
│                 Infrastructure Layer                  │
│  ┌──────────┐  ┌──────────┐  ┌────────────────┐   │
│  │Model Mgr │  │Cache Sys │  │  Visualizer   │   │
│  └──────────┘  └──────────┘  └────────────────┘   │
└──────────────────────────────────────────────────────┘
```

---

## 项目结构

```
20-llm-eval-benchmark/
├── app/                          # FastAPI 应用
│   ├── __init__.py
│   ├── main.py                   # 应用入口，路由注册
│   ├── models.py                 # Pydantic 数据模型（请求/响应）
│   └── api/
│       ├── __init__.py
│       ├── benchmark.py          # 评测 API（执行、进度、结果）
│       ├── model.py              # 模型管理 API（CRUD）
│       ├── leaderboard.py        # 排行榜 API（排名、卡片、图表）
│       └── report.py             # 报告 API（JSON/MD/CSV、对比）
├── benchmarks/                   # 评测基准实现
│   ├── __init__.py               # 基准注册表
│   ├── base.py                   # 抽象基类（接口定义）
│   ├── mmlu.py                   # MMLU 多选题评测
│   ├── gsm8k.py                  # GSM8K 数学推理评测
│   ├── humaneval.py              # HumanEval 代码生成评测
│   ├── mt_bench.py               # MT-Bench 多轮对话评测
│   └── custom.py                 # 自定义基准（JSON 导入）
├── engine/                       # 评测引擎
│   ├── __init__.py
│   ├── evaluator.py              # 异步并发评测引擎
│   ├── scorer.py                 # 多指标评分系统
│   └── comparator.py             # 模型对比分析
├── models/                       # 模型管理
│   ├── __init__.py
│   └── model_manager.py          # 模型注册与版本管理
├── leaderboard/                  # 排行榜与可视化
│   ├── __init__.py
│   ├── ranker.py                 # 排名系统
│   └── visualizer.py             # matplotlib 图表生成
├── data/                         # 样本数据
│   ├── mmlu_sample.json          # 50 题 MMLU（7 个学科）
│   ├── gsm8k_sample.json         # 30 题 GSM8K（3 级难度）
│   ├── humaneval_sample.json     # 10 题 HumanEval
│   └── mt_bench_sample.json      # 10 轮 MT-Bench 对话
├── tests/                        # 测试套件
│   ├── __init__.py
│   ├── test_benchmarks.py        # 基准测试（25 个测试）
│   ├── test_engine.py            # 引擎测试（6 个测试）
│   ├── test_scorer.py            # 评分测试（14 个测试）
│   ├── test_leaderboard.py       # 排行榜测试（11 个测试）
│   └── test_api.py               # API 测试（18 个测试）
├── reports/                      # 生成的报告和图表
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── README.md
```

---

## 快速开始

### 环境要求

- Python 3.11+
- pip

### 安装依赖

```bash
cd 20-llm-eval-benchmark
pip install -r requirements.txt
```

### 启动服务

```bash
# 开发模式启动
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 或直接运行
python -m app.main
```

### 运行测试

```bash
pytest tests/ -v
```

### Docker 部署

```bash
# 构建并启动
docker-compose up --build

# 运行测试
docker-compose --profile test up eval-worker
```

---

## API 文档

启动服务后访问：
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### 核心 API 端点

| 方法 | 路径 | 描述 |
|------|------|------|
| GET | `/` | 健康检查 |
| GET | `/api/benchmark/list` | 列出所有可用基准 |
| GET | `/api/benchmark/{name}/info` | 获取基准详细信息 |
| POST | `/api/benchmark/evaluate` | 执行模型评测 |
| GET | `/api/benchmark/progress/{model}` | 获取评测进度 |
| GET | `/api/benchmark/results/{model}/{benchmark}` | 获取评测结果 |
| POST | `/api/models/register` | 注册新模型 |
| GET | `/api/models/list` | 列出所有模型 |
| GET | `/api/models/{name}` | 获取模型信息 |
| PUT | `/api/models/{name}` | 更新模型信息 |
| DELETE | `/api/models/{name}` | 删除模型 |
| GET | `/api/leaderboard/overall` | 综合排行榜 |
| GET | `/api/leaderboard/benchmark/{name}` | 分基准排行榜 |
| GET | `/api/leaderboard/model/{name}` | 模型卡片 |
| GET | `/api/leaderboard/charts` | 生成图表 |
| POST | `/api/report/generate` | 生成报告（JSON/MD/CSV） |
| POST | `/api/report/compare` | 对比两个模型 |

---

## 使用示例

### 1. 执行评测

```bash
curl -X POST http://localhost:8000/api/benchmark/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "model_name": "gpt-4",
    "benchmark": "mmlu",
    "max_concurrent": 10,
    "question_limit": 20,
    "timeout_seconds": 30,
    "enable_cache": true
  }'
```

响应示例：
```json
{
  "model_name": "gpt-4",
  "benchmark": "mmlu",
  "total_questions": 20,
  "correct": 16,
  "accuracy": 0.8,
  "eval_time_seconds": 2.35,
  "scores": {
    "accuracy": {
      "metric": "accuracy",
      "value": 0.8,
      "confidence_interval": {"low": 0.5634, "high": 0.9428}
    },
    "f1": {"metric": "f1", "value": 0.8},
    "pass@1": {"metric": "pass@1", "value": 0.8},
    "precision": {"metric": "precision", "value": 0.8},
    "recall": {"metric": "recall", "value": 0.8}
  },
  "results": [...]
}
```

### 2. 注册模型

```bash
curl -X POST http://localhost:8000/api/models/register \
  -H "Content-Type: application/json" \
  -d '{
    "name": "llama-3-70b",
    "api_endpoint": "https://api.example.com/v1",
    "parameters": "70B",
    "provider": "meta",
    "description": "Llama 3 70B Instruct"
  }'
```

### 3. 查看排行榜

```bash
curl http://localhost:8000/api/leaderboard/overall
```

### 4. 生成报告

```bash
# Markdown 报告
curl -X POST http://localhost:8000/api/report/generate \
  -H "Content-Type: application/json" \
  -d '{"format": "markdown", "include_details": true}'

# CSV 报告
curl -X POST http://localhost:8000/api/report/generate \
  -H "Content-Type: application/json" \
  -d '{"format": "csv"}'
```

### 5. 对比模型

```bash
curl -X POST http://localhost:8000/api/report/compare \
  -H "Content-Type: application/json" \
  -d '{"model_a": "gpt-4", "model_b": "claude-3"}'
```

---

## 评测基准详解

### MMLU (Massive Multitask Language Understanding)

- **类型**: 多选题（4 选 1）
- **覆盖领域**: 数学、物理、计算机科学、历史、哲学、经济学、心理学（7 个学科）
- **样本量**: 50 题
- **评分**: 准确率 + Wilson 置信区间
- **答案提取**: 支持直接字母回答和 "The answer is X" 等模式

### GSM8K (Grade School Math 8K)

- **类型**: 数学推理题
- **难度分布**: easy / medium / hard
- **样本量**: 30 题
- **评分**: 数值精确匹配（容差 1e-6）
- **答案提取**: 支持 `#### 数字` 格式和末尾数字提取

### HumanEval

- **类型**: 代码生成
- **语言**: Python
- **样本量**: 10 题
- **评分**: Pass@k（函数签名检测 + 测试用例执行框架）
- **答案提取**: 支持 Markdown 代码块提取

### MT-Bench (Multi-Turn Benchmark)

- **类型**: 多轮对话评分
- **类别**: writing, roleplay, reasoning, math, coding, extraction, humanities, science, brainstorming, safety
- **样本量**: 10 轮对话（20 个评估点）
- **评估维度**: relevance, coherence, creativity, accuracy, correctness 等
- **评分**: 1-10 分制（由外部评分器或 LLM-as-judge 完成）

---

## 评分指标

| 指标 | 描述 | 适用基准 |
|------|------|----------|
| Accuracy | 准确率 + Wilson 置信区间 | MMLU, GSM8K, Custom |
| Pass@k | k 次尝试中至少一次通过的概率 | HumanEval |
| F1 Score | 精确率和召回率的调和平均 | 所有基准 |
| Precision | 精确率 (TP / (TP + FP)) | 所有基准 |
| Recall | 召回率 (TP / (TP + FN)) | 所有基准 |
| McNemar Test | 配对二分类统计显著性检验 | 模型对比 |
| Score Mean | 平均分数（1-10 分制） | MT-Bench |

---

## 评分系统技术细节

### Wilson 置信区间

准确率的置信区间使用 **Wilson score interval** 计算，相比正态近似，在小样本下具有更好的覆盖率：

```
center = (p_hat + z^2 / (2n)) / (1 + z^2 / n)
spread = z * sqrt((p_hat * (1 - p_hat) + z^2 / (4n)) / n) / (1 + z^2 / n)
CI = [center - spread, center + spread]
```

### McNemar 显著性检验

使用 McNemar 检验判断两个模型的性能差异是否具有统计显著性：

- `n_10`: 模型 A 正确而模型 B 错误的次数
- `n_01`: 模型 A 错误而模型 B 正确的次数
- 统计量: `chi^2 = (|n_10 - n_01| - 1)^2 / (n_10 + n_01)`
- 当 p < 0.05 时认为差异显著

---

## 评测引擎特性

### 并发执行

基于 `asyncio.Semaphore` 控制最大并发数，使用 `asyncio.as_completed` 实现流式结果收集。

### 超时控制

每个题目使用 `asyncio.timeout` 实现独立的超时控制，避免单题卡死影响整体评测。

### 重试机制

支持配置最大重试次数和重试延迟（指数退避），确保网络波动下的评测稳定性。

### 结果缓存

基于 MD5 哈希的评测结果缓存，避免重复评测同一模型在同一基准上的表现。缓存可按需开关。

### 进度追踪

通过回调函数实时报告评测进度，包括完成数、正确数、错误数、准确率和耗时。

---

## 自定义评测基准

支持通过 JSON 文件导入自定义评测题目：

### 多选题格式

```json
{
  "name": "my_custom_bench",
  "description": "My custom benchmark",
  "questions": [
    {
      "id": 1,
      "question": "What is the capital of Japan?",
      "answer": 1,
      "type": "multiple_choice",
      "choices": ["Beijing", "Tokyo", "Seoul", "Bangkok"],
      "category": "geography"
    }
  ]
}
```

### 精确匹配格式

```json
{
  "name": "my_qa_bench",
  "questions": [
    {
      "id": 1,
      "question": "Who wrote Romeo and Juliet?",
      "answer": "Shakespeare",
      "type": "exact_match"
    }
  ]
}
```

---

## 可视化图表

平台使用 matplotlib 生成以下图表：

1. **综合排行榜柱状图** (`leaderboard_bar.png`): 模型综合得分水平柱状图
2. **多基准对比柱状图** (`benchmark_comparison.png`): 多模型在多个基准上的分组对比
3. **雷达图** (`radar_comparison.png`): 多模型能力维度雷达对比图

---

## 测试覆盖

项目包含完整的测试套件，覆盖以下模块：

| 测试文件 | 测试数量 | 覆盖内容 |
|----------|----------|----------|
| `test_benchmarks.py` | 25 | 数据加载、提示构建、答案提取、答案检查 |
| `test_engine.py` | 6 | 并发执行、进度追踪、超时、缓存 |
| `test_scorer.py` | 14 | 所有评分指标、置信区间、显著性检验 |
| `test_leaderboard.py` | 11 | 排名、模型卡片、可视化图表 |
| `test_api.py` | 18 | 所有 API 端点的请求/响应验证 |
| **合计** | **74** | |

---

## 扩展指南

### 添加新的评测基准

1. 在 `benchmarks/` 下创建新文件
2. 继承 `BaseBenchmark` 并实现四个抽象方法
3. 在 `benchmarks/__init__.py` 中注册

```python
from .base import BaseBenchmark, BenchmarkQuestion

class MyBenchmark(BaseBenchmark):
    name = "my_benchmark"
    description = "My custom benchmark"

    def load_data(self) -> list[BenchmarkQuestion]:
        ...

    def build_prompt(self, question: BenchmarkQuestion) -> str:
        ...

    def extract_answer(self, response: str, question: BenchmarkQuestion):
        ...

    def check_answer(self, extracted, question: BenchmarkQuestion) -> bool:
        ...
```

### 集成真实 LLM API

替换 `MockLLMClient`，实现 `LLMClient` 接口：

```python
class OpenAIClient(LLMClient):
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self._api_key = api_key
        self._model = model

    async def generate(self, prompt: str, **kwargs) -> str:
        import httpx
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._model,
                    "messages": [{"role": "user", "content": prompt}],
                },
            )
            return resp.json()["choices"][0]["message"]["content"]
```

---

## License

MIT