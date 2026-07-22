# RAG 知识库问答系统

企业级 RAG（Retrieval-Augmented Generation）知识库问答系统，支持多格式文档解析、语义分块、混合检索、重排序、流式输出和引用溯源。

## 系统架构

```
                          用户请求
                             |
                             v
                    +-----------------+
                    |   FastAPI 服务   |
                    +-----------------+
                             |
              +--------------+--------------+
              |              |              |
        +-----------+  +-----------+  +-----------+
        | 文档管理  |  |  查询接口  |  | 系统管理  |
        +-----------+  +-----------+  +-----------+
              |              |              |
              v              v              v
        +-----------------------------------------+
        |            RAG 核心服务层                  |
        +-----------------------------------------+
              |              |              |
    +---------+--------+    |    +---------+--------+
    |                  |    |    |                  |
    v                  v    v    v                  v
+--------+      +--------+  |  +--------+    +-----------+
| 文档解析 | --> | 文本分块 |  |  | LLM    |    | 引用溯源   |
| PDF/Word  |      | 固定/段落 |  |  | 生成器  |    | Citation  |
| MD/TXT    |      | 语义     |  |  | OpenAI |    | Tracker   |
+--------+      +----+---+----+  |  | 智谱AI  |    +-----------+
                      |          |  | 百炼    |
                      v          |  +--------+
                +-----------+    |
                |  向量化   |    |
                | BGE/OpenAI|    |
                | 本地模型   |    |
                +-----+-----+    |
                      |          |
                      v          |
                +-----------+    |
                | ChromaDB  |    |
                | 向量存储   |    |
                +-----+-----+    |
                      |          |
                      v          v
                +-------------------+
                |    混合检索器       |
                | BM25 + 向量检索    |
                | RRF 融合排序       |
                +---------+---------+
                          |
                          v
                   +------------+
                   |  Reranker  |
                   | 交叉编码器  |
                   | 精排 Top-K  |
                   +------------+
```

## 核心功能

| 功能 | 描述 |
|------|------|
| **多格式文档解析** | 支持 PDF、Word (docx)、Markdown、TXT 四种格式 |
| **语义分块** | 三种策略：固定长度分块、按段落分块、语义分块 |
| **多模型向量化** | 支持 BGE、OpenAI、本地 SentenceTransformers |
| **混合检索** | BM25 关键词检索 + 向量语义检索，RRF 融合排序 |
| **Reranker 精排** | 交叉编码器二次排序（模型不可用时自动降级为启发式规则） |
| **流式输出** | 基于 SSE (Server-Sent Events) 的流式回答 |
| **引用溯源** | 返回来源文档和段落信息，支持引用标注 |

## 项目结构

```
01-rag-knowledge-base/
├── app/                          # 主应用层
│   ├── api/                      # API 路由
│   │   ├── query.py              # 查询接口（同步/流式）
│   │   ├── documents.py          # 文档管理接口
│   │   └── system.py             # 系统管理接口
│   ├── models/
│   │   └── schemas.py            # Pydantic 数据模型
│   ├── services/
│   │   └── rag_service.py        # RAG 核心编排服务
│   └── __init__.py
├── core/                         # 核心模块
│   ├── config.py                 # 全局配置
│   ├── parsers/
│   │   └── document_parser.py    # 多格式文档解析
│   ├── chunking/
│   │   └── chunking.py           # 文本分块（三种策略）
│   ├── embeddings/
│   │   └── embeddings.py         # 向量化（BGE/OpenAI/本地）
│   ├── retrieval/
│   │   ├── vector_store.py       # ChromaDB 向量存储
│   │   ├── retriever.py          # 混合检索（BM25 + 向量）
│   │   └── citation.py           # 引用溯源
│   ├── reranker/
│   │   └── reranker.py           # 重排序（交叉编码器）
│   └── llm/
│       └── llm.py                # LLM 调用（OpenAI/智谱/百炼）
├── tests/                        # 测试
│   ├── test_document_parser.py   # 文档解析测试
│   ├── test_chunking.py          # 分块测试
│   └── test_config_and_api.py    # 配置和 API 测试
├── docs/                         # 文档
├── data/                         # 数据目录
│   ├── uploads/                  # 上传文件
│   ├── processed/                # 处理后的文件
│   └── chroma_db/                # ChromaDB 持久化
├── scripts/                      # 工具脚本
│   ├── start.py                  # 快速启动
│   └── import_docs.py            # 批量导入文档
├── main.py                       # FastAPI 入口
├── requirements.txt              # 依赖清单
├── Dockerfile                    # Docker 镜像
├── docker-compose.yml            # Docker Compose
├── .env                          # 环境配置
└── .gitignore
```

## 快速开始

### 1. 环境准备

```bash
# 克隆项目
cd 01-rag-knowledge-base

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 配置

编辑 `.env` 文件，根据需要修改配置：

```bash
# 如果使用 OpenAI
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-actual-api-key

# 如果使用本地 Embedding（无需 GPU，更易启动）
EMBEDDING_PROVIDER=local
LOCAL_EMBEDDING_MODEL=all-MiniLM-L6-v2

# 关闭 Reranker 节省内存（首次启动推荐）
RERANKER_ENABLED=false
```

### 3. 启动服务

```bash
# 方式一：直接启动
python main.py

# 方式二：使用启动脚本
python scripts/start.py

# 方式三：Docker
docker-compose up --build
```

### 4. 使用

启动后访问 http://localhost:8000/docs 查看 API 文档。

**上传文档：**
```bash
curl -X POST "http://localhost:8000/api/v1/documents/upload" \
  -F "file=@your-document.pdf"
```

**查询（同步）：**
```bash
curl -X POST "http://localhost:8000/api/v1/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是RAG技术？", "top_k": 5}'
```

**查询（流式）：**
```bash
curl -X POST "http://localhost:8000/api/v1/query/stream" \
  -H "Content-Type: application/json" \
  -d '{"question": "什么是RAG技术？", "stream": true}'
```

**批量导入：**
```bash
python scripts/import_docs.py /path/to/your/documents --strategy semantic
```

## API 接口

| 方法 | 路径 | 描述 |
|------|------|------|
| POST | `/api/v1/documents/upload` | 上传文档 |
| GET | `/api/v1/documents` | 获取文档列表 |
| DELETE | `/api/v1/documents` | 删除文档 |
| GET | `/api/v1/documents/supported-types` | 获取支持的文件类型 |
| POST | `/api/v1/query` | 知识库查询（同步） |
| POST | `/api/v1/query/stream` | 知识库查询（SSE 流式） |
| GET | `/api/v1/health` | 健康检查 |
| GET | `/api/v1/stats` | 系统统计信息 |
| POST | `/api/v1/reset` | 重置知识库 |

## 配置说明

| 配置项 | 默认值 | 说明 |
|--------|--------|------|
| `EMBEDDING_PROVIDER` | `bge` | Embedding 提供商 (bge/openai/local) |
| `BGE_MODEL_NAME` | `BAAI/bge-large-zh-v1.5` | BGE 模型 |
| `LOCAL_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | 本地模型 |
| `LLM_PROVIDER` | `openai` | LLM 提供商 (openai/zhipu/dashscope) |
| `LLM_MODEL` | `gpt-3.5-turbo` | LLM 模型 |
| `CHUNK_STRATEGY` | `semantic` | 分块策略 (fixed/paragraph/semantic) |
| `CHUNK_SIZE` | `512` | 分块大小 |
| `RERANKER_ENABLED` | `true` | 是否启用重排序 |
| `HYBRID_SEARCH_WEIGHT_VECTOR` | `0.7` | 向量检索权重 |
| `HYBRID_SEARCH_WEIGHT_BM25` | `0.3` | BM25 检索权重 |

## 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定测试
pytest tests/test_chunking.py -v
pytest tests/test_document_parser.py -v
```

## 技术栈

- **Web 框架**: FastAPI + Uvicorn
- **向量数据库**: ChromaDB（本地持久化）
- **文档解析**: PyMuPDF (PDF) / python-docx (Word) / 内置 (MD/TXT)
- **Embedding**: SentenceTransformers / FlagEmbedding / OpenAI
- **检索**: BM25 (rank-bm25 + jieba) + 向量检索 (ChromaDB)
- **重排序**: HuggingFace Transformers (Cross-Encoder)
- **LLM**: OpenAI / 智谱AI / 阿里云百炼
- **配置管理**: pydantic-settings + python-dotenv
- **日志**: loguru

## License

MIT