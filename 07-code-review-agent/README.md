# AI Agent 代码审查系统

基于 LangGraph 的多 Agent 协作代码审查系统，支持 AST 分析、GitHub API 集成。

## 架构

```
┌──────────────────────────────────────────────────────────┐
│                    FastAPI Application                     │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  POST /api/v1/review/code     - 提交代码审查             │
│  POST /api/v1/review/github   - GitHub PR 审查           │
│  GET  /api/v1/reports/{id}    - 查询报告                 │
│  GET  /api/v1/reports         - 列出所有报告             │
│  GET  /health                 - 健康检查                 │
│                                                          │
├──────────────────────────────────────────────────────────┤
│                  LangGraph StateGraph                     │
│                                                          │
│         ┌─────────┐                                       │
│         │ Preprocess │  (AST Analysis)                   │
│         └────┬────┘                                       │
│     ┌────────┼────────┬────────┐                          │
│     ▼        ▼        ▼        ▼                          │
│ ┌────────┐┌────────┐┌────────┐┌────────┐                  │
│ │Security││Perform.││ Style  ││ Logic  │                  │
│ │ Agent  ││ Agent  ││ Agent  ││ Agent  │                  │
│ └───┬────┘└───┬────┘└───┬────┘└───┬────┘                  │
│     └────────┼────────┼────────┘                          │
│             ▼                                             │
│     ┌──────────────┐                                      │
│     │Summary Agent │  (Aggregate & Report)                │
│     └──────────────┘                                      │
│                                                          │
├──────────────────────────────────────────────────────────┤
│                    Analyzers                              │
│  ┌──────────┐ ┌───────────┐ ┌────────────┐             │
│  │AST       │ │Complexity  │ │Diff Parser │             │
│  │Analyzer  │ │Analyzer    │ │            │             │
│  └──────────┘ └───────────┘ └────────────┘             │
│  ┌──────────┐                                            │
│  │Dependency│                                            │
│  │Analyzer  │                                            │
│  └──────────┘                                            │
└──────────────────────────────────────────────────────────┘
```

## 技术栈

- **Python 3.11+** - 编程语言
- **FastAPI** - Web 框架
- **LangGraph** - Agent 工作流编排
- **Pydantic** - 数据模型验证
- **ast (stdlib)** - Python AST 分析
- **httpx** - 异步 HTTP 客户端
- **pytest** - 测试框架

## 5 个专业化 Agent

| Agent | 职责 | 检测项 |
|-------|------|--------|
| **SecurityAgent** | 安全漏洞检测 | SQL注入、XSS、硬编码密钥、不安全依赖 |
| **PerformanceAgent** | 性能问题检测 | 圈复杂度、N+1查询、内存泄漏 |
| **StyleAgent** | 代码规范检查 | PEP8、命名规范、类型注解、docstring |
| **LogicAgent** | 逻辑缺陷检测 | 空指针、边界条件、异常处理 |
| **SummaryAgent** | 汇总生成报告 | 聚合所有 Agent 结果、生成 Markdown/JSON 报告 |

## 快速启动

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. 运行测试

```bash
pytest tests/ -v
```

### 4. Docker 部署

```bash
docker-compose up --build
```

## API 文档

### 提交代码审查

```bash
curl -X POST http://localhost:8000/api/v1/review/code \
  -H "Content-Type: application/json" \
  -d '{
    "code": "password = \"secret\"\neval(user_input)",
    "language": "python",
    "filename": "example.py"
  }'
```

### GitHub PR 审查

```bash
curl -X POST http://localhost:8000/api/v1/review/github \
  -H "Content-Type: application/json" \
  -d '{
    "repo_owner": "octocat",
    "repo_name": "hello-world",
    "pr_number": 1,
    "github_token": "your_token_here",
    "post_comments": true
  }'
```

### 查询报告

```bash
curl http://localhost:8000/api/v1/reports/{review_id}
```

### 健康检查

```bash
curl http://localhost:8000/health
```

## 审查报告

系统生成两种格式的报告：

- **Markdown**: 结构化的审查报告，包含统计概览、各 Agent 结果、问题详情
- **JSON**: 机器可读的结构化报告，便于 CI/CD 集成

### 严重等级

| 等级 | 说明 |
|------|------|
| Critical | 严重安全漏洞，必须立即修复 |
| High | 高危问题，强烈建议修复 |
| Medium | 中等问题，建议修复 |
| Low | 低优先级问题 |
| Info | 信息提示，仅供参考 |

## 项目结构

```
07-code-review-agent/
├── app/                     # FastAPI 应用
│   ├── main.py              # 应用入口
│   ├── models.py            # Pydantic 模型
│   └── api/                 # API 路由
├── agents/                  # 审查 Agent
│   ├── base.py              # Agent 基类
│   ├── security_agent.py    # 安全审查
│   ├── performance_agent.py # 性能审查
│   ├── style_agent.py       # 风格审查
│   ├── logic_agent.py       # 逻辑审查
│   └── summary_agent.py     # 汇总报告
├── analyzers/               # 静态分析器
│   ├── ast_analyzer.py      # AST 分析
│   ├── complexity.py        # 圈复杂度
│   ├── diff_parser.py       # Diff 解析
│   └── dependency.py        # 依赖分析
├── github/                  # GitHub 集成
│   ├── client.py            # API 客户端
│   └── reviewer.py          # PR Review
├── workflows/               # LangGraph 工作流
│   └── review_graph.py      # StateGraph
├── templates/               # 报告模板
├── tests/                   # 测试
├── requirements.txt
├── Dockerfile
└── docker-compose.yml
```
