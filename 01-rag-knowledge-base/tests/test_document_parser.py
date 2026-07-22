"""
文档解析器测试
"""
import os
import tempfile
import pytest
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).parent.parent
import sys
sys.path.insert(0, str(PROJECT_ROOT))

from core.parsers.document_parser import DocumentParser, ParsedDocument


class TestDocumentParser:
    """文档解析器测试"""

    @pytest.fixture
    def parser(self):
        """创建解析器实例"""
        return DocumentParser()

    def test_supported_extensions(self, parser):
        """测试获取支持的文件类型"""
        exts = parser.get_supported_extensions()
        assert ".pdf" in exts
        assert ".docx" in exts
        assert ".md" in exts
        assert ".txt" in exts

    def test_is_supported(self, parser):
        """测试文件类型判断"""
        assert parser.is_supported("test.pdf") is True
        assert parser.is_supported("test.docx") is True
        assert parser.is_supported("test.md") is True
        assert parser.is_supported("test.txt") is True
        assert parser.is_supported("test.exe") is False
        assert parser.is_supported("test.csv") is False

    def test_parse_txt(self, parser):
        """测试 TXT 文件解析"""
        # 创建临时测试文件
        test_content = """第一段：这是测试内容。

第二段：RAG 系统是一种结合了检索和生成的技术。

第三段：它能够利用外部知识库来增强语言模型的回答质量。
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(test_content)
            f.flush()
            temp_path = f.name

        try:
            result = parser.parse(temp_path)

            assert isinstance(result, ParsedDocument)
            assert result.file_type == "txt"
            assert "RAG" in result.content
            assert len(result.sections) >= 1
            assert result.total_chars > 0
            assert "parser" in result.metadata
        finally:
            os.unlink(temp_path)

    def test_parse_markdown(self, parser):
        """测试 Markdown 文件解析"""
        test_content = """# RAG 知识库系统

## 概述

RAG（Retrieval-Augmented Generation）是一种增强型生成技术。

## 核心组件

### 文档解析
支持 PDF、Word、Markdown、TXT 等格式。

### 向量检索
使用 ChromaDB 进行向量相似度搜索。

## 总结

RAG 系统通过检索增强生成，提供准确的回答。
"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8"
        ) as f:
            f.write(test_content)
            f.flush()
            temp_path = f.name

        try:
            result = parser.parse(temp_path)

            assert isinstance(result, ParsedDocument)
            assert result.file_type == "markdown"
            assert result.title == "RAG 知识库系统"
            assert "RAG" in result.content
            assert len(result.sections) > 0
        finally:
            os.unlink(temp_path)

    def test_parse_unsupported_type(self, parser):
        """测试不支持的文件类型"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".exe", delete=False
        ) as f:
            f.write("test")
            temp_path = f.name

        try:
            with pytest.raises(ValueError, match="不支持"):
                parser.parse(temp_path)
        finally:
            os.unlink(temp_path)

    def test_parse_nonexistent_file(self, parser):
        """测试解析不存在的文件"""
        with pytest.raises(FileNotFoundError):
            parser.parse("/nonexistent/file.pdf")

    def test_parse_empty_txt(self, parser):
        """测试空 TXT 文件"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("")
            temp_path = f.name

        try:
            result = parser.parse(temp_path)
            assert result.content == ""
            assert result.sections == []
        finally:
            os.unlink(temp_path)