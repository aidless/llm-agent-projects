# AI Workflow Automation Platform

AI 工作流自动化平台 - 可视化工作流编排引擎，支持节点拖拽、条件分支、循环、子工作流等。

## 技术栈

- Python 3.11+, FastAPI, Pydantic
- DAG (有向无环图) 工作流引擎
- 8 种内置节点类型

## 核心功能

### 工作流引擎
- DAG 工作流定义与验证 (循环检测、孤立节点检测)
- 并行分支和串行节点执行
- 错误重试机制

### 节点类型
| 节点 | 类型 | 说明 |
|------|------|------|
| LLMNode | llm | LLM 文本生成，支持模板变量 |
| HTTPNode | http | HTTP 请求 (GET/POST/PUT/DELETE) |
| CodeNode | code | 安全 Python 代码执行 (白名单沙箱) |
| ConditionNode | condition | 条件分支 (eq/gt/lt/contains/regex 等) |
| LoopNode | loop | 循环执行 (for/while) |
| AggregateNode | aggregate | 聚合多个输入 (merge/list/sum/concat) |
| TimerNode | timer | 延时执行 |
| SubWorkflowNode | sub_workflow | 嵌套调用其他工作流 |

### 变量系统
- 节点间变量传递: `{{node_id.output.key}}`
- 环境变量引用: `{{env.VAR_NAME}}`
- 嵌套 key 支持: `{{node_id.output.data.score}}`
- 类型保持 (单变量返回原始类型)

### 工作流管理
- CRUD 操作
- 版本管理与回滚
- 导入/导出 (JSON)
- 内置模板

### 内置模板
1. **文本摘要** - LLM 摘要长文本
2. **多语言翻译** - 语言检测 + 翻译 + 质量检查
3. **数据分析报告** - 数据预处理 + 分析 + 报告生成

## 快速开始

### 安装
```bash
cd 11-dify-workflow
pip install -r requirements.txt
```

### 启动服务
```bash
python -m app.main
# 或
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
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

启动服务后访问:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 主要 API

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /api/v1/workflows | 创建工作流 |
| GET | /api/v1/workflows | 列出所有工作流 |
| GET | /api/v1/workflows/{id} | 获取工作流详情 |
| PUT | /api/v1/workflows/{id} | 更新工作流 |
| DELETE | /api/v1/workflows/{id} | 删除工作流 |
| POST | /api/v1/executions | 执行工作流 |
| GET | /api/v1/executions | 列出执行记录 |
| GET | /api/v1/templates | 列出内置模板 |
| POST | /api/v1/templates/create | 从模板创建工作流 |

## 项目结构

```
11-dify-workflow/
├── app/           # FastAPI 应用
│   ├── main.py    # 入口
│   ├── models.py  # Pydantic 模型
│   └── api/       # API 路由
├── engine/        # 工作流引擎
│   ├── dag.py     # DAG 构建/验证
│   ├── executor.py # 执行引擎
│   └── context.py  # 变量系统
├── nodes/         # 节点实现
├── storage/       # 存储层
├── templates/     # 内置模板
└── tests/         # 测试 (25+)
```
