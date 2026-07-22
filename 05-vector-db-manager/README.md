# 向量数据库管理平台 - 快速入门指南

## 概述

向量数据库管理平台是一个基于 FastAPI 和 ChromaDB 的 Web 应用，提供向量数据库的完整管理功能，包括文档导入、分块管理、Embedding 模型切换和相似度搜索。

## 安装

### 环境要求
- Python 3.9+
- pip

### 安装依赖
```bash
cd 05-vector-db-manager
pip install -r requirements.txt
```

### 配置
复制并编辑 .env 文件：
```bash
cp .env .env.local
# 修改配置参数
```

主要配置项：
- `DEFAULT_EMBEDDING_MODEL`: 默认嵌入模型名称
- `CHROMA_PERSIST_DIR`: ChromaDB 数据持久化目录
- `EMBEDDING_MODEL_CACHE_DIR`: 模型缓存目录

## 启动

```bash
# 方式一：直接运行
python -m app.main

# 方式二：使用 uvicorn
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

启动后访问：
- Web UI: http://localhost:8000
- API 文档: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 使用流程

### 1. 加载 Embedding 模型
在"模型管理"页面加载一个嵌入模型（首次使用本地模型需要下载）。
推荐使用 BAAI/bge-small-zh-v1.5（中文小模型，速度快）。

### 2. 创建 Collection
在"集合管理"页面创建一个新的集合。

### 3. 导入文档
在"文档导入"页面选择分块策略，导入文件或目录。
支持 TXT、MD、JSON、CSV 格式。

### 4. 搜索测试
在"相似度搜索"页面输入查询文本进行搜索。
支持设置 top-k、相似度阈值和元数据过滤。

### 5. 索引对比
使用"索引对比"功能比较不同分块策略和模型的检索效果。

### 6. 快速导入示例数据
```bash
# 导入示例数据集
python scripts/import_sample.py
```

## Docker 部署

```bash
docker build -t vector-db-manager .
docker run -d -p 8000:8000 -v ./chroma_data:/app/chroma_data -v ./model_cache:/app/model_cache vector-db-manager
```

## 项目结构

```
05-vector-db-manager/
├── app/                    # FastAPI 应用
│   ├── __init__.py
│   ├── config.py          # 配置管理（从 .env 加载）
│   └── main.py            # 主应用入口
├── api/                    # REST API 路由
│   ├── __init__.py
│   └── routes.py          # API 端点定义
├── services/               # 业务服务层
│   ├── __init__.py
│   ├── chunker.py         # 文档分块器（4种策略）
│   ├── file_parser.py     # 文件解析器（TXT/MD/JSON/CSV）
│   └── vector_store.py     # 向量存储服务（ChromaDB 封装）
├── embeddings/             # Embedding 模型管理
│   ├── __init__.py
│   └── manager.py         # 模型加载器（BGE/OpenAI/本地）
├── templates/               # Jinja2 HTML 模板
│   ├── base.html          # 基础模板
│   ├── index.html         # 统计面板
│   ├── models.html        # 模型管理
│   ├── collections.html   # 集合管理
│   ├── import.html        # 文档导入
│   ├── search.html        # 相似度搜索
│   ├── compare.html       # 索引对比
│   ├── chunk.html         # 分块测试
│   └── docs.html          # API 文档
├── data/                   # 示例数据
│   └── sample_knowledge.json
├── scripts/                # 工具脚本
│   └── import_sample.py    # 示例数据导入脚本
├── tests/                  # 测试文件
│   └── test_services.py
├── .env                    # 环境变量配置
├── requirements.txt        # Python 依赖
├── Dockerfile              # Docker 构建文件
└── README.md               # 本文件
```

## 技术栈

- **Web 框架**: FastAPI
- **向量数据库**: ChromaDB
- **嵌入模型**: SentenceTransformers (BGE 系列)
- **模板引擎**: Jinja2
- **配置管理**: python-dotenv + pydantic-settings
