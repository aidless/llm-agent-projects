# AI Agent 自动化工作流平台

基于 **Python + LangGraph + FastAPI** 构建的多 Agent 协作自动化工作流平台。

## 架构概览

```
用户请求 → FastAPI → LangGraph 工作流编排
                        ├── PlannerAgent (规划)
                        ├── ExecutorAgent (执行 + 工具调用)
                        ├── ReviewerAgent (审核)
                        ├── 条件分支 (通过/重试/强制完成)
                        └── SummarizerAgent (报告生成)
```

## 项目结构

```
02-agent-workflow/
├── app/                    # FastAPI 主应用
│   ├── main.py             # 应用入口和生命周期管理
│   ├── models.py           # Pydantic 数据模型
│   └── routes/             # API 路由
│       ├── health_routes.py    # 健康检查
│       ├── workflow_routes.py  # 工作流执行和查询
│       ├── tool_routes.py      # 工具调用
│       └── memory_routes.py    # 记忆管理
├── agents/                 # Agent 定义
│   ├── base_agent.py       # 基类和 LLM 工厂
│   ├── planner_agent.py    # 规划 Agent
│   ├── executor_agent.py   # 执行 Agent
│   ├── reviewer_agent.py   # 审核 Agent
│   └── summarizer_agent.py # 总结 Agent
├── tools/                  # 自定义工具集
│   ├── search_tool.py      # 网页搜索
│   ├── file_tool.py        # 文件读写
│   ├── code_executor.py    # 代码执行沙箱
│   ├── calculator.py       # 数学计算器
│   └── api_caller.py       # 通用 API 调用
├── memory/                 # 记忆模块
│   ├── short_term.py       # 短期对话记忆 (deque)
│   └── long_term.py        # 长期向量记忆 (ChromaDB)
├── workflows/              # 工作流编排
│   ├── state.py            # 工作流状态定义
│   └── research_workflow.py # 研究报告生成工作流
├── config/                 # 配置模块
│   └── settings.py         # 环境配置管理
├── tests/                  # 测试
│   ├── test_short_term_memory.py
│   ├── test_long_term_memory.py
│   ├── test_tools.py
│   └── test_api.py
├── .env                    # 环境配置
├── .env.example            # 配置示例
├── requirements.txt         # Python 依赖
├── Dockerfile               # Docker 构建文件
├── pytest.ini              # 测试配置
└── README.md               # 本文件
```

## 技术栈

| 组件 | 技术 |
|------|------|
| Web 框架 | FastAPI + Uvicorn |
| LLM | OpenAI API / DeepSeek API（兼容接口） |
| Agent 框架 | LangChain + LangGraph |
| 工具调用 | Function Calling |
| 向量记忆 | ChromaDB |
| 数据验证 | Pydantic |
| 日志 | Loguru |

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
cd 02-agent-workflow

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置环境变量

```bash
# 复制配置模板
cp .env.example .env

# 编辑 .env，填入你的 API Key
# 至少需要配置 OPENAI_API_KEY 或 DEEPSEEK_API_KEY
```

### 3. 启动服务

```bash
# 开发模式（自动重载）
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

# 或直接运行
python app/main.py
```

### 4. 访问 API 文档

启动后访问：
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## API 接口

### 工作流

```bash
# 查看工作流信息
curl http://localhost:8000/api/v1/workflow/info

# 执行研究报告生成工作流
curl -X POST http://localhost:8000/api/v1/workflow/run \
  -H "Content-Type: application/json" \
  -d '{"task": "分析2024年人工智能领域的主要发展趋势"}'

# 查询任务状态
curl http://localhost:8000/api/v1/workflow/tasks/{task_id}

# Agent 对话
curl -X POST http://localhost:8000/api/v1/workflow/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "帮我制定一个项目计划"}'
```

### 工具

```bash
# 获取工具列表
curl http://localhost:8000/api/v1/tools

# 调用计算器
curl -X POST http://localhost:8000/api/v1/tools/call \
  -H "Content-Type: application/json" \
  -d '{"tool_name": "calculator_tool", "arguments": {"expression": "2 ** 32"}}'
```

### 记忆

```bash
# 搜索长期记忆
curl -X POST http://localhost:8000/api/v1/memory/search \
  -H "Content-Type: application/json" \
  -d '{"query": "AI 发展趋势", "top_k": 5}'

# 添加记忆
curl -X POST http://localhost:8000/api/v1/memory/add \
  -H "Content-Type: application/json" \
  -d '{"content": "重要知识片段", "metadata": {"source": "manual"}}'
```

## 核心功能

### 多 Agent 协作

4 个专业 Agent 各司其职，通过 LangGraph StateGraph 编排协作：

1. **PlannerAgent** - 分析任务需求，制定分步执行计划
2. **ExecutorAgent** - 按计划执行任务，调用工具获取数据
3. **ReviewerAgent** - 审核执行结果，评估质量（0-100分）
4. **SummarizerAgent** - 整理结果，生成结构化报告

### 工作流条件分支

```
规划 → 执行 → 审核 → 判断
                    ├── 通过 (>=80分) → 生成报告 → 完成
                    ├── 未通过 & 未超重试 → 重新规划（循环）
                    └── 未通过 & 超重试 → 强制生成报告
```

### 自定义工具

| 工具 | 功能 |
|------|------|
| web_search_tool | 网页搜索（支持自定义 API） |
| file_read/write_tool | 安全的文件读写操作 |
| code_execute_tool | Python 代码沙箱执行 |
| calculator_tool | 安全数学表达式计算 |
| api_call_tool | 通用 HTTP API 调用 |

### 双层记忆系统

- **短期记忆**：基于 deque 的有限窗口对话历史（默认 20 轮）
- **长期记忆**：基于 ChromaDB 的语义向量存储，支持相似度检索

## 运行测试

```bash
# 运行所有测试
pytest

# 运行特定测试
pytest tests/test_tools.py -v

# 运行单元测试（跳过集成测试）
pytest -m "not integration"
```

## Docker 部署

```bash
# 构建镜像
docker build -t agent-workflow .

# 运行容器
docker run -d \
  --name agent-workflow \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  --env-file .env \
  agent-workflow
```

## 配置说明

主要环境变量（.env 文件）：

| 变量 | 说明 | 默认值 |
|------|------|--------|
| LLM_PROVIDER | LLM 提供商 (openai/deepseek) | openai |
| OPENAI_API_KEY | OpenAI API Key | - |
| DEEPSEEK_API_KEY | DeepSeek API Key | - |
| CHROMA_PERSIST_DIR | ChromaDB 存储路径 | ./data/chroma_db |
| MAX_RETRIES | 最大重试次数 | 3 |
| TASK_TIMEOUT | 任务超时时间(秒) | 300 |

## License

MIT
