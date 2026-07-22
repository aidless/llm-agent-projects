"""FastAPI application entry point."""

from __future__ import annotations

import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.review import router as review_router
from app.api.report import router as report_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(
        title="Multi-Agent Paper Review System",
        description="5 specialised agents collaborate to review academic papers with cross-model validation.",
        version="1.0.0",
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(review_router, prefix="/api/v1", tags=["review"])
    app.include_router(report_router, prefix="/api/v1", tags=["report"])
    return app


app = create_app()


@app.get("/health")
async def health_check():
    return {"status": "ok"}
