# Prompt 评估与优化框架

支持多种评估指标（BLEU/ROUGE/BERTScore/LLM-as-Judge），提供 Prompt A/B 测试和自动优化。

## 功能特性

### 评估指标
- **BLEU** (1-4 gram) - 基于 nltk 真实计算
- **ROUGE** (ROUGE-1/2/L/sum) - 基于 rouge-score 真实计算
- **BERTScore** (F1/Precision/Recall) - 使用 sklearn TfidfVectorizer + cosine_similarity 模拟
- **Exact Match** - 精确匹配，支持忽略大小写和标点
- **F1 Score** - Token 级别 F1 分数
- **自定义指标** - 通过配置定义

### LLM-as-Judge
- 评估维度：准确性/相关性/完整性/连贯性/安全性/综合
- 评分量表：1-5 分
- 评估 Prompt 模板管理
- 多 Judge 投票一致性

### A/B 测试
- Prompt 版本管理
- 多版本并行评估
- 配对 t 检验
- Bootstrap 置信区间
- 胜率计算

### 数据集管理
- 输入/预期输出数据集
- 数据集导入 (JSON/JSONL/CSV)
- 数据增强 (同义改写/回译模拟)
- 采样策略 (random/first/last)

### 评估流水线
- 批量评估执行
- 并发控制
- 结果聚合和统计
- 评估报告生成 (JSON/Markdown)

### Prompt 优化器
- Prompt 质量分析与改进建议
- Few-shot 示例自动选择
- Prompt 模板变量化

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动服务

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

## API 文档

启动服务后访问 `http://localhost:8000/docs` 查看交互式 API 文档。

### 主要 API

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | /api/v1/metrics | 列出所有评估指标 |
| POST | /api/v1/evaluate | 执行评估 |
| POST | /api/v1/judge | LLM-as-Judge 评估 |
| GET | /api/v1/judge/dimensions | 列出评估维度 |
| GET | /api/v1/datasets/ | 列出数据集 |
| POST | /api/v1/datasets/ | 创建数据集 |
| POST | /api/v1/datasets/import | 导入数据集 |
| GET | /api/v1/prompts/versions | 列出 Prompt 版本 |
| POST | /api/v1/prompts/ab-test | 运行 A/B 测试 |
| POST | /api/v1/prompts/analyze | 分析 Prompt |
| POST | /api/v1/reports/generate | 生成评估报告 |

## 项目结构

```
12-prompt-eval-framework/
├── app/                  # FastAPI 应用
│   ├── api/              # API 路由
│   ├── main.py           # 入口
│   └── models.py         # Pydantic 模型
├── metrics/              # 评估指标
├── judge/                # LLM-as-Judge
├── ab_test/              # A/B 测试
├── datasets/             # 数据集管理
├── pipeline/             # 评估流水线与优化
├── tests/                # 测试
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```