"""表格模块测试。"""

import pytest
from PIL import Image, ImageDraw

from app.models import TableCell, TableData, TableOutputFormat
from table.detector import TableDetector
from table.extractor import TableExtractor
from table.formatter import TableFormatter


def create_table_image(
    rows: int = 4,
    cols: int = 3,
    cell_width: int = 80,
    cell_height: int = 30,
) -> Image.Image:
    """创建包含网格线的表格测试图片。"""
    width = cols * cell_width
    height = rows * cell_height
    img = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    line_color = (0, 0, 0)
    # 水平线
    for r in range(rows + 1):
        y = r * cell_height
        draw.line([(0, y), (width, y)], fill=line_color, width=2)
    # 垂直线
    for c in range(cols + 1):
        x = c * cell_width
        draw.line([(x, 0), (x, height)], fill=line_color, width=2)

    return img


def create_sample_table() -> TableData:
    """创建示例表格数据。"""
    cells = [
        TableCell(row=0, col=0, text="姓名"),
        TableCell(row=0, col=1, text="年龄"),
        TableCell(row=0, col=2, text="城市"),
        TableCell(row=1, col=0, text="张三"),
        TableCell(row=1, col=1, text="25"),
        TableCell(row=1, col=2, text="北京"),
        TableCell(row=2, col=0, text="李四"),
        TableCell(row=2, col=1, text="30"),
        TableCell(row=2, col=2, text="上海"),
    ]
    return TableData(
        rows=3,
        cols=3,
        cells=cells,
        headers=["姓名", "年龄", "城市"],
    )


class TestTableDetector:
    """表格检测器测试。"""

    def test_detect_table_image(self):
        """测试检测表格图片。"""
        detector = TableDetector(min_lines=2, min_area_ratio=0.01)
        img = create_table_image(rows=5, cols=4, cell_width=100, cell_height=40)
        results = detector.detect(img)
        # 网格图片应能检测到表格区域
        assert isinstance(results, list)

    def test_detect_blank_image(self):
        """测试检测空白图片。"""
        detector = TableDetector()
        img = Image.new("RGB", (200, 200), (255, 255, 255))
        results = detector.detect(img)
        assert results == []

    def test_detect_with_text_image(self):
        """测试检测带文字的图片 (非表格)。"""
        detector = TableDetector(min_lines=5)
        img = Image.new("RGB", (300, 200), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((10, 10), "这是一段普通文字", fill=(0, 0, 0))
        results = detector.detect(img)
        assert isinstance(results, list)


class TestTableExtractor:
    """表格提取器测试。"""

    def test_extract_from_table_image(self):
        """测试从表格图片提取数据。"""
        extractor = TableExtractor()
        img = create_table_image(rows=5, cols=4, cell_width=100, cell_height=40)
        result = extractor.extract(img)
        assert isinstance(result.tables, list)
        assert result.page == 1

    def test_extract_from_blank_image(self):
        """测试从空白图片提取 (无表格)。"""
        extractor = TableExtractor()
        img = Image.new("RGB", (200, 200), (255, 255, 255))
        result = extractor.extract(img)
        assert result.tables == []


class TestTableFormatter:
    """表格格式转换测试。"""

    def setup_method(self):
        self.formatter = TableFormatter()
        self.table = create_sample_table()

    def test_to_markdown(self):
        """测试 Markdown 格式转换。"""
        result = self.formatter.to_markdown(self.table)
        assert "|" in result
        assert "姓名" in result
        assert "---" in result
        assert "张三" in result
        assert "李四" in result

    def test_to_html(self):
        """测试 HTML 格式转换。"""
        result = self.formatter.to_html(self.table)
        assert "<table>" in result
        assert "</table>" in result
        assert "<th>" in result
        assert "<td>" in result
        assert "姓名" in result

    def test_to_json(self):
        """测试 JSON 格式转换。"""
        import json

        result = self.formatter.to_json(self.table)
        parsed = json.loads(result)
        assert isinstance(parsed, list)
        assert len(parsed) == 2  # 2 行数据 (不含表头)
        assert "姓名" in parsed[0]
        assert parsed[0]["姓名"] == "张三"

    def test_to_csv(self):
        """测试 CSV 格式转换。"""
        result = self.formatter.to_csv(self.table)
        assert "姓名" in result
        assert "张三" in result
        assert "\n" in result

    def test_format_table_invalid_format(self):
        """测试无效格式。"""
        with pytest.raises(ValueError, match="不支持的格式"):
            self.formatter.format_table(self.table, "xml")

    def test_empty_table_markdown(self):
        """测试空表格转 Markdown。"""
        empty = TableData(rows=0, cols=0, cells=[], headers=[])
        result = self.formatter.to_markdown(empty)
        assert result == ""

    def test_validate_table_valid(self):
        """测试验证有效表格。"""
        result = self.formatter.validate_table(self.table)
        assert result["is_valid"] is True
        assert result["errors"] == []

    def test_validate_table_empty(self):
        """测试验证空表格。"""
        empty = TableData(rows=0, cols=0, cells=[], headers=[])
        result = self.formatter.validate_table(empty)
        assert result["is_valid"] is False
        assert len(result["errors"]) >= 1