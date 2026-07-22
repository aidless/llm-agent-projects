# AI Agent Security Sandbox

为 AI Agent 代码执行提供安全隔离环境，支持系统调用拦截、资源限制和行为审计。

## 核心功能

- **代码沙箱**: 基于 RestrictedPython 的 AST 级别代码限制，支持安全执行
- **系统调用拦截**: 检测并拦截 exec/eval/__import__/open/os/subprocess 等危险调用
- **资源限制**: 执行超时、内存限制、输出大小限制、并发数限制
- **行为审计**: 完整的执行日志、事件追踪、安全告警和审计报告
- **安全策略引擎**: 4级预设安全策略 (LOW/MEDIUM/HIGH/STRICT)，支持策略继承和组合

## 技术栈

- Python 3.11+
- FastAPI (Web 框架)
- RestrictedPython (AST 级别代码限制)
- psutil (资源监控)

## 项目结构

```
app/            # FastAPI 应用入口和 API 路由
sandbox/        # 沙箱执行器、RestrictedPython 环境、资源限制、输出捕获
security/       # 系统调用拦截、文件/网络/导入守卫
policy/         # 策略引擎、策略模型、预设策略
audit/          # 审计日志、事件追踪、报告生成
tests/          # 测试套件
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动服务

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Docker 部署

```bash
docker-compose up -d
```

## API 使用

### 代码执行

```bash
curl -X POST http://localhost:8000/api/v1/execute \
  -H "Content-Type: application/json" \
  -d '{"code": "print(1 + 1)", "policy_name": "medium"}'
```

### 查看策略列表

```bash
curl http://localhost:8000/api/v1/policies
```

### 审计日志查询

```bash
curl http://localhost:8000/api/v1/audit/logs
```

### 安全报告

```bash
curl http://localhost:8000/api/v1/audit/reports/security
```

## 安全等级说明

| 等级 | 执行时间 | 内存 | 网络访问 | 文件写入 | 说明 |
|------|---------|------|---------|---------|------|
| LOW  | 60s | 512MB | 允许 | 允许 | 仅拦截最危险操作 |
| MEDIUM | 30s | 256MB | 禁止 | 禁止 | 默认策略，拦截危险模块 |
| HIGH | 10s | 128MB | 禁止 | 禁止 | 严格模块白名单 |
| STRICT | 5s | 64MB | 禁止 | 禁止 | 最小模块集，完全隔离 |

## 运行测试

```bash
pytest tests/ -v
```