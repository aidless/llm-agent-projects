"""
配置和 API 测试
"""
import os
import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 在导入 app 之前设置环境变量（避免加载大模型）
os.environ["RERANKER_ENABLED"] = "false"
os.environ["EMBEDDING_PROVIDER"] = "local"
os.environ["LOCAL_EMBEDDING_MODEL"] = "all-MiniLM-L6-v2"


class TestConfig:
    """配置模块测试"""

    def test_settings_load(self):
        """测试配置加载"""
        from core.config import settings
        assert settings.app_name == "RAG-Knowledge-Base"
        assert settings.chunk_size == 512
        assert settings.retrieval_top_k == 10

    def test_supported_file_types(self):
        """测试支持的文件类型"""
        from core.config import settings
        types = settings.supported_file_types_list
        assert "pdf" in types
        assert "docx" in types
        assert "md" in types
        assert "txt" in types

    def test_max_file_size(self):
        """测试文件大小限制"""
        from core.config import settings
        assert settings.max_file_size_bytes == 50 * 1024 * 1024

    def test_ensure_directories(self):
        """测试目录创建"""
        from core.config import settings
        settings.ensure_directories()
        assert os.path.isdir(settings.data_dir)


class TestSchemas:
    """数据模型测试"""

    def test_query_request(self):
        """测试查询请求模型"""
        from app.models.schemas import QueryRequest
        req = QueryRequest(question="什么是RAG？")
        assert req.question == "什么是RAG？"
        assert req.stream is False

    def test_query_request_validation(self):
        """测试请求验证"""
        from app.models.schemas import QueryRequest
        from pydantic import ValidationError

        # 空问题应报错
        with pytest.raises(ValidationError):
            QueryRequest(question="")

    def test_health_response(self):
        """测试健康检查响应"""
        from app.models.schemas import HealthResponse
        resp = HealthResponse(status="ok", version="1.0.0")
        assert resp.status == "ok"

    def test_document_upload_response(self):
        """测试文档上传响应"""
        from app.models.schemas import DocumentUploadResponse
        resp = DocumentUploadResponse(
            filename="test.pdf",
            chunk_count=10,
            total_chars=5000,
        )
        assert resp.filename == "test.pdf"

    def test_citation_source(self):
        """测试引用来源模型"""
        from app.models.schemas import CitationSource
        citation = CitationSource(
            source_file="test.pdf",
            doc_title="测试文档",
            score=0.95,
        )
        assert citation.source_file == "test.pdf"
        assert citation.score == 0.95