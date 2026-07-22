"""
多格式文档解析模块
支持 PDF、Word (docx)、Markdown、TXT 四种格式的文档解析
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional

from loguru import logger

# 尝试导入文档解析库，缺失时给出友好提示
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    logger.warning("pymupdf 未安装，PDF 解析功能不可用。请运行: pip install pymupdf")

try:
    from docx import Document as DocxDocument
    HAS_DOCX = True
except ImportError:
    HAS_DOCX = False
    logger.warning("python-docx 未安装，Word 文档解析功能不可用。请运行: pip install python-docx")


@dataclass
class ParsedDocument:
    """解析后的文档结构"""

    filename: str  # 文件名
    file_type: str  # 文件类型: pdf / docx / md / txt
    title: str  # 文档标题（从文件名提取）
    content: str  # 完整文本内容
    sections: List[str] = field(default_factory=list)  # 按章节/段落拆分的文本段
    metadata: dict = field(default_factory=dict)  # 文档元数据（页数、字数等）
    raw_path: str = ""  # 原始文件路径

    @property
    def total_chars(self) -> int:
        """文档总字符数"""
        return len(self.content)

    @property
    def total_sections(self) -> int:
        """文档段落数"""
        return len(self.sections)


class DocumentParser:
    """
    多格式文档解析器

    使用示例:
        parser = DocumentParser()
        doc = parser.parse("path/to/document.pdf")
        print(doc.content)
        print(doc.sections)
    """

    # 支持的文件类型映射
    SUPPORTED_TYPES = {
        ".pdf": "pdf",
        ".docx": "docx",
        ".doc": "docx",
        ".md": "markdown",
        ".txt": "txt",
    }

    def __init__(self):
        """初始化文档解析器"""
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """检查可选依赖是否已安装"""
        if not HAS_PYMUPDF:
            logger.warning("PDF 解析依赖 pymupdf 未安装")
        if not HAS_DOCX:
            logger.warning("Word 解析依赖 python-docx 未安装")

    def parse(self, file_path: str) -> ParsedDocument:
        """
        解析文档，自动识别文件类型

        Args:
            file_path: 文档文件路径

        Returns:
            ParsedDocument: 解析后的文档对象

        Raises:
            ValueError: 不支持的文件类型
            FileNotFoundError: 文件不存在
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        # 获取文件后缀并转换为小写
        suffix = path.suffix.lower()
        if suffix not in self.SUPPORTED_TYPES:
            raise ValueError(
                f"不支持的文件类型: {suffix}，"
                f"支持类型: {list(self.SUPPORTED_TYPES.keys())}"
            )

        file_type = self.SUPPORTED_TYPES[suffix]
        logger.info(f"开始解析文档: {path.name} (类型: {file_type})")

        # 根据类型选择解析方法
        if file_type == "pdf":
            parsed = self._parse_pdf(str(path))
        elif file_type == "docx":
            parsed = self._parse_docx(str(path))
        elif file_type == "markdown":
            parsed = self._parse_markdown(str(path))
        elif file_type == "txt":
            parsed = self._parse_txt(str(path))
        else:
            raise ValueError(f"未知文件类型: {file_type}")

        logger.info(
            f"文档解析完成: {path.name}, "
            f"总字符数: {parsed.total_chars}, "
            f"段落数: {parsed.total_sections}"
        )
        return parsed

    def _parse_pdf(self, file_path: str) -> ParsedDocument:
        """
        解析 PDF 文档

        使用 PyMuPDF (fitz) 提取文本，逐页提取并保留页面信息

        Args:
            file_path: PDF 文件路径

        Returns:
            ParsedDocument: 解析后的文档对象
        """
        if not HAS_PYMUPDF:
            raise RuntimeError(
                "PDF 解析需要安装 pymupdf 库。请运行: pip install pymupdf"
            )

        doc = fitz.open(file_path)
        all_text = []
        sections = []

        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            text = page.get_text("text").strip()
            if text:
                # 将每页作为一个段落
                sections.append(text)
                all_text.append(text)

        doc.close()

        # 用换行符拼接所有页面文本
        full_content = "\n\n".join(all_text)

        # 提取文件名作为标题
        filename = Path(file_path).stem

        return ParsedDocument(
            filename=Path(file_path).name,
            file_type="pdf",
            title=filename,
            content=full_content,
            sections=sections,
            metadata={
                "page_count": len(sections),
                "parser": "pymupdf",
                "file_size": os.path.getsize(file_path),
            },
            raw_path=file_path,
        )

    def _parse_docx(self, file_path: str) -> ParsedDocument:
        """
        解析 Word 文档 (docx)

        使用 python-docx 提取段落文本，保留段落结构

        Args:
            file_path: Word 文件路径

        Returns:
            ParsedDocument: 解析后的文档对象
        """
        if not HAS_DOCX:
            raise RuntimeError(
                "Word 解析需要安装 python-docx 库。请运行: pip install python-docx"
            )

        doc = DocxDocument(file_path)
        sections = []
        all_text = []

        # 提取所有段落
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                sections.append(text)
                all_text.append(text)

        # 提取表格中的文本
        table_texts = []
        for table in doc.tables:
            for row in table.rows:
                row_text = " | ".join(
                    cell.text.strip() for cell in row.cells if cell.text.strip()
                )
                if row_text:
                    table_texts.append(row_text)
                    sections.append(row_text)

        full_content = "\n\n".join(all_text)

        filename = Path(file_path).stem

        return ParsedDocument(
            filename=Path(file_path).name,
            file_type="docx",
            title=filename,
            content=full_content,
            sections=sections,
            metadata={
                "paragraph_count": len(doc.paragraphs),
                "table_count": len(doc.tables),
                "parser": "python-docx",
                "file_size": os.path.getsize(file_path),
            },
            raw_path=file_path,
        )

    def _parse_markdown(self, file_path: str) -> ParsedDocument:
        """
        解析 Markdown 文档

        保留 Markdown 的标题层级结构，按标题拆分段落

        Args:
            file_path: Markdown 文件路径

        Returns:
            ParsedDocument: 解析后的文档对象
        """
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # 按换行拆分，过滤空行，组成段落
        lines = content.split("\n")
        sections = []
        current_section = []

        for line in lines:
            stripped = line.strip()
            if not stripped:
                # 空行作为段落分隔
                if current_section:
                    section_text = "\n".join(current_section).strip()
                    if section_text:
                        sections.append(section_text)
                    current_section = []
            else:
                current_section.append(line)

        # 处理最后一段
        if current_section:
            section_text = "\n".join(current_section).strip()
            if section_text:
                sections.append(section_text)

        filename = Path(file_path).stem

        # 提取一级标题作为文档标题
        title = filename
        for line in lines:
            if line.startswith("# "):
                title = line[2:].strip()
                break

        return ParsedDocument(
            filename=Path(file_path).name,
            file_type="markdown",
            title=title,
            content=content,
            sections=sections,
            metadata={
                "line_count": len(lines),
                "heading_count": sum(1 for l in lines if l.strip().startswith("#")),
                "parser": "built-in",
                "file_size": os.path.getsize(file_path),
            },
            raw_path=file_path,
        )

    def _parse_txt(self, file_path: str) -> ParsedDocument:
        """
        解析纯文本文件

        按空行分段落，兼容多种编码

        Args:
            file_path: 文本文件路径

        Returns:
            ParsedDocument: 解析后的文档对象
        """
        # 尝试多种编码
        content = None
        for encoding in ["utf-8", "gbk", "gb2312", "latin-1"]:
            try:
                with open(file_path, "r", encoding=encoding) as f:
                    content = f.read()
                break
            except (UnicodeDecodeError, UnicodeError):
                continue

        if content is None:
            raise RuntimeError(f"无法解码文件: {file_path}，尝试了 utf-8/gbk/gb2312/latin-1")

        # 按空行分段落
        paragraphs = content.split("\n\n")
        sections = [p.strip() for p in paragraphs if p.strip()]

        filename = Path(file_path).stem

        return ParsedDocument(
            filename=Path(file_path).name,
            file_type="txt",
            title=filename,
            content=content,
            sections=sections,
            metadata={
                "line_count": content.count("\n") + 1,
                "parser": "built-in",
                "file_size": os.path.getsize(file_path),
            },
            raw_path=file_path,
        )

    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """获取支持的文件扩展名列表"""
        return list(cls.SUPPORTED_TYPES.keys())

    @classmethod
    def is_supported(cls, file_path: str) -> bool:
        """判断文件是否为支持的类型"""
        suffix = Path(file_path).suffix.lower()
        return suffix in cls.SUPPORTED_TYPES