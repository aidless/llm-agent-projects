"""文本解析器 - 支持纯文本和 Markdown。"""

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.models import (
    DocumentFormat,
    ParsedDocument,
    ParsedPage,
)


class TextParser:
    """纯文本和 Markdown 文档解析器。

    支持:
    - .txt 纯文本
    - .md Markdown
    - 基本结构化分页 (按空行分隔)
    - Markdown 标题层级提取
    """

    SUPPORTED_EXTENSIONS = {".txt": DocumentFormat.TEXT, ".md": DocumentFormat.MARKDOWN}

    def parse(self, file_path: str, **kwargs) -> ParsedDocument:
        """解析文本文件。

        Args:
            file_path: 文件路径。
            **kwargs: 额外参数。

        Returns:
            ParsedDocument: 解析结果。
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        suffix = path.suffix.lower()
        doc_format = self.SUPPORTED_EXTENSIONS.get(suffix, DocumentFormat.TEXT)

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        return self.parse_text(content, filename=path.name, format=doc_format)

    def parse_text(
        self,
        content: str,
        filename: str = "text.txt",
        format: DocumentFormat = DocumentFormat.TEXT,
    ) -> ParsedDocument:
        """直接解析文本字符串。

        Args:
            content: 文本内容。
            filename: 文件名。
            format: 文档格式。

        Returns:
            ParsedDocument: 解析结果。
        """
        # 按空行分页 (简单策略)
        sections = re.split(r"\n\s*\n", content.strip())
        sections = [s.strip() for s in sections if s.strip()]

        pages: List[ParsedPage] = []
        all_text_parts: List[str] = []

        for i, section in enumerate(sections, 1):
            pages.append(ParsedPage(page_num=i, text=section))
            all_text_parts.append(section)

        # 如果没有分页，整个文档作为一页
        if not pages:
            pages.append(ParsedPage(page_num=1, text=content))
            all_text_parts.append(content)

        full_text = "\n\n".join(all_text_parts)

        # Markdown 特殊处理: 提取标题结构
        metadata: Dict[str, Any] = {
            "char_count": len(content),
            "section_count": len(sections),
            "parser": "TextParser",
        }

        if format == DocumentFormat.MARKDOWN:
            headings = re.findall(r"^(#{1,6})\s+(.+)$", content, re.MULTILINE)
            metadata["headings"] = [
                {"level": len(h[0]), "text": h[1].strip()} for h in headings
            ]

        return ParsedDocument(
            filename=filename,
            format=format,
            pages=pages,
            full_text=full_text,
            metadata=metadata,
        )