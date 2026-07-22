"""解析器模块测试。"""

import os
import tempfile

import pytest
from PIL import Image, ImageDraw

from app.models import DocumentFormat, ParsedDocument
from parsers.image_parser import ImageParser
from parsers.text_parser import TextParser
from utils.image_utils import create_test_image, load_image, image_to_base64


# ============ TextParser 测试 ============


class TestTextParser:
    """文本解析器测试。"""

    def test_parse_text_basic(self):
        """测试基本文本解析。"""
        parser = TextParser()
        result = parser.parse_text(
            "这是第一段内容。\n\n这是第二段内容。",
            filename="test.txt",
        )
        assert result.filename == "test.txt"
        assert result.format == DocumentFormat.TEXT
        assert len(result.pages) == 2
        assert "第一段" in result.full_text
        assert "第二段" in result.full_text

    def test_parse_text_markdown_headings(self):
        """测试 Markdown 标题提取。"""
        parser = TextParser()
        md_content = "# 一级标题\n\n## 二级标题\n\n正文内容。"
        result = parser.parse_text(
            md_content, filename="test.md", format=DocumentFormat.MARKDOWN
        )
        assert result.format == DocumentFormat.MARKDOWN
        assert "headings" in result.metadata
        assert len(result.metadata["headings"]) == 2
        assert result.metadata["headings"][0]["level"] == 1
        assert result.metadata["headings"][0]["text"] == "一级标题"

    def test_parse_text_from_file(self):
        """测试从文件解析文本。"""
        parser = TextParser()
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("文件内容第一行。\n\n文件内容第二行。")
            f.flush()
            tmp_path = f.name

        try:
            result = parser.parse(tmp_path)
            assert result.filename.endswith(".txt")
            assert len(result.pages) == 2
        finally:
            os.unlink(tmp_path)

    def test_parse_text_empty(self):
        """测试空文本。"""
        parser = TextParser()
        result = parser.parse_text("", filename="empty.txt")
        assert len(result.pages) == 1
        assert result.full_text == ""

    def test_parse_text_single_section(self):
        """测试单段文本。"""
        parser = TextParser()
        result = parser.parse_text("只有一段内容", filename="single.txt")
        assert len(result.pages) == 1
        assert result.pages[0].text == "只有一段内容"


# ============ ImageParser 测试 ============


class TestImageParser:
    """图片解析器测试。"""

    def test_parse_image_basic(self):
        """测试基本图片解析。"""
        parser = ImageParser()
        image = create_test_image(200, 150, text="Hello")
        result = parser.parse_image(image, filename="test.png")

        assert result.filename == "test.png"
        assert result.format == DocumentFormat.IMAGE
        assert len(result.pages) == 1
        assert result.pages[0].page_num == 1

    def test_parse_image_with_png_file(self):
        """测试解析 PNG 文件。"""
        parser = ImageParser()
        image = create_test_image(100, 100)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            image.save(f.name, "PNG")
            tmp_path = f.name

        try:
            result = parser.parse(tmp_path)
            assert result.format == DocumentFormat.IMAGE
            assert result.pages[0].images  # 应有 base64 图片
        finally:
            os.unlink(tmp_path)

    def test_parse_image_unsupported_format(self):
        """测试不支持的图片格式。"""
        parser = ImageParser()
        with tempfile.NamedTemporaryFile(suffix=".xyz", delete=False) as f:
            f.write(b"fake content")
            tmp_path = f.name

        try:
            with pytest.raises(ValueError, match="不支持的图片格式"):
                parser.parse(tmp_path)
        finally:
            os.unlink(tmp_path)

    def test_parse_image_file_not_found(self):
        """测试文件不存在。"""
        parser = ImageParser()
        with pytest.raises(FileNotFoundError):
            parser.parse("/nonexistent/path/image.png")

    def test_parse_image_grayscale_mode(self):
        """测试灰度图转 RGB。"""
        parser = ImageParser()
        gray_img = Image.new("L", (100, 100), 128)
        result = parser.parse_image(gray_img, filename="gray.png")
        assert result.format == DocumentFormat.IMAGE
        assert len(result.pages) == 1

    def test_parse_multi_page(self):
        """测试多页图片解析。"""
        parser = ImageParser()
        paths = []
        for i in range(3):
            img = create_test_image(100, 100, text=f"Page {i+1}")
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                img.save(f.name, "PNG")
                paths.append(f.name)

        try:
            result = parser.parse_multi_page(paths)
            assert len(result.pages) == 3
            assert result.metadata["page_count"] == 3
        finally:
            for p in paths:
                os.unlink(p)
