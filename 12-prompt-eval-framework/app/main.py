"""FastAPI 应用入口。"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api import evaluation, dataset, prompt, report

app = FastAPI(
    title="Prompt 评估与优化框架",
    description="支持多种评估指标的 Prompt 评估、A/B 测试和自动优化框架",
    version="1.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(evaluation.router)
app.include_router(dataset.router)
app.include_router(prompt.router)
app.include_router(report.router)


@app.get("/")
def root():
    return {
        "name": "Prompt 评估与优化框架",
        "version": "1.0.0",
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}
