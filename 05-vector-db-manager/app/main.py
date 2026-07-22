# -*- coding: utf-8 -*-
"""
向量数据库管理平台 - FastAPI 主应用
集成 API 路由和 Jinja2 模板渲染
"""
import logging
import sys
from pathlib import Path

# 将项目根目录加入 Python 路径
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from app.config import settings
from api.routes import router as api_router

# 配置日志
logging.basicConfig(
    level=logging.DEBUG if settings.debug else logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# 创建 FastAPI 应用
app = FastAPI(
    title="向量数据库管理平台",
    description="基于 ChromaDB 的向量数据库管理系统，支持文档导入、分块管理、Embedding 模型切换和相似度搜索",
    version="1.0.0",
)

# 注册 API 路由
app.include_router(api_router)

# 配置 Jinja2 模板
templates_dir = Path(__file__).resolve().parent.parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# 静态文件（如果有）
static_dir = Path(__file__).resolve().parent.parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ========== 页面路由 ==========

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    首页 - 统计面板
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/collections", response_class=HTMLResponse)
async def collections_page(request: Request):
    """Collection 管理页面"""
    return templates.TemplateResponse("collections.html", {"request": request})


@app.get("/import", response_class=HTMLResponse)
async def import_page(request: Request):
    """文档导入页面"""
    return templates.TemplateResponse("import.html", {"request": request})


@app.get("/search", response_class=HTMLResponse)
async def search_page(request: Request):
    """相似度搜索页面"""
    return templates.TemplateResponse("search.html", {"request": request})


@app.get("/models", response_class=HTMLResponse)
async def models_page(request: Request):
    """模型管理页面"""
    return templates.TemplateResponse("models.html", {"request": request})


@app.get("/compare", response_class=HTMLResponse)
async def compare_page(request: Request):
    """多索引对比页面"""
    return templates.TemplateResponse("compare.html", {"request": request})


@app.get("/chunk", response_class=HTMLResponse)
async def chunk_page(request: Request):
    """分块测试页面"""
    return templates.TemplateResponse("chunk.html", {"request": request})


@app.get("/docs-page", response_class=HTMLResponse)
async def docs_page(request: Request):
    """API 文档页面"""
    return templates.TemplateResponse("docs.html", {"request": request})


# ========== 启动事件 ==========

@app.on_event("startup")
async def startup_event():
    """应用启动时的初始化"""
    logger.info("=" * 60)
    logger.info("向量数据库管理平台启动中...")
    logger.info(f"数据持久化目录: {settings.get_chroma_persist_path()}")
    logger.info(f"模型缓存目录: {settings.get_model_cache_path()}")
    logger.info(f"数据目录: {settings.get_data_path()}")
    logger.info("=" * 60)


# ========== 错误处理 ==========

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """全局异常处理"""
    logger.error(f"未处理的异常: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": f"服务器内部错误: {str(exc)}"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.debug,
    )
