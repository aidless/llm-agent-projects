# MCP Tool Integration System

基于 MCP (Model Context Protocol) 的工具集成系统，实现了完整的协议通信、工具注册发现、权限管理和 Agent 集成。

## 核心功能

- **JSON-RPC 2.0**: 完整的请求/响应/通知/批量消息实现
- **MCP Server SDK**: 装饰器注册工具、资源管理、Prompt 模板
- **内置工具服务器**: 文件系统、计算器、数据库查询
- **MCP Client**: 连接池、工具缓存、自动重连
- **工具注册中心**: 服务器注册、心跳检测、工具搜索、权限管理
- **Agent 集成**: 工具选择策略、链式调用、错误重试

## 项目结构

```
app/           - FastAPI 应用和 API 路由
protocol/      - JSON-RPC 和 MCP 协议类型定义
server/        - MCP 服务端基类、传输层、内置服务器
client/        - MCP 客户端、连接池、工具缓存
registry/      - 工具注册中心、权限管理
agent/         - 工具调用 Agent、链式执行器
tests/         - 测试文件
```

## 快速开始

```bash
pip install -r requirements.txt
python -m uvicorn app.main:app --reload
```

## 运行测试

```bash
pytest tests/ -v
```

## Docker 部署

```bash
docker-compose up --build
```

## API 端点

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | / | 系统信息 |
| GET | /health | 健康检查 |
| GET | /registry/servers | 列出服务器 |
| GET | /registry/tools | 搜索工具 |
| POST | /tools/call | 调用工具 |
| POST | /tools/agent/execute | Agent 执行 |
| POST | /tools/chain/execute | 链式调用 |
| GET | /servers/list | 服务器列表 |
| POST | /servers/register | 注册服务器 |
| DELETE | /servers/{name} | 注销服务器 |