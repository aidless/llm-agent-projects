# 多模态文档理解系统

支持图片/OCR/表格解析/版面分析的多模态文档处理系统，结合 RAG 实现文档问答。

## 功能特性

- **多格式文档解析**: PDF、图片、Word、纯文本/Markdown
- **OCR 文字识别**: Tesseract OCR 集成，支持中英文，图片预处理 pipeline
- **表格识别与结构化**: 表格检测、行列识别、Markdown/HTML/JSON/CSV 格式转换
- **版面分析**: 区域检测(标题/段落/表格/图片)、阅读顺序排序、结构树生成
- **多模态理解**: 图片描述生成、图表理解、视觉问答(VQA)
- **文档问答 RAG**: 文档分块与向量化、混合检索(语义+关键词)、带来源引用的回答

## 技术栈

- Python 3.11+, FastAPI, Pillow, pytesseract, pdf2image, pypdf, python-docx
- 支持多模态 LLM API (GLM-4V/Qwen-VL/GPT-4o Vision)

## 项目结构

```
app/
  main.py              # FastAPI 入口
  models.py            # Pydantic 数据模型
  api/
    parse.py           # 文档解析 API
    ocr.py             # OCR API
    table.py           # 表格识别 API
    qa.py              # 问答 API
parsers/               # 文档解析器
  pdf_parser.py
  image_parser.py
  docx_parser.py
  text_parser.py
ocr/                   # OCR 模块
  engine.py            # OCR 引擎封装 (Tesseract/API)
  preprocessor.py      # 图片预处理 pipeline
  layout_analyzer.py   # 版面分析
table/                 # 表格模块
  detector.py          # 表格检测
  extractor.py         # 表格提取
  formatter.py         # 格式转换
vision/                # 多模态理解
  describer.py         # 图片描述生成
  chart_understanding.py # 图表理解
  vqa.py               # 视觉问答
rag/                   # RAG 模块
  document_indexer.py  # 文档索引
  retriever.py         # 混合检索
  qa_chain.py          # 问答链
utils/
  image_utils.py       # 图片工具
tests/                 # 测试
```

## 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

### 启动服务

```bash
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

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | /parse/upload | 上传并解析文档 |
| POST | /ocr/recognize | OCR 识别 |
| POST | /table/extract | 表格提取 |
| POST | /qa/ask | 文档问答 |
| GET  | /qa/documents | 已索引文档列表 |
| DELETE | /qa/documents/{id} | 删除索引 |

## 注意事项

- Tesseract OCR 需要系统安装 tesseract-ocr (Docker 镜像已包含)
- Vision LLM API 调用默认使用 Mock 实现，可通过继承抽象基类接入真实 API
- RAG 使用内存向量存储，适合中小规模文档场景