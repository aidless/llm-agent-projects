"""Word 文档解析器 - 支持 .docx 格式。"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.models import (
    DocumentFormat,
    ParsedDocument,
    ParsedPage,
    TableData,
    TableCell,
)
from utils.image_utils import image_to_base64

logger = logging.getLogger(__name__)


class DocxParser:
    """Word (.docx) 文档解析器。

    支持:
    - 段落文本提取
    - 表格提取
    - 图片提取
    - 基本样式信息 (标题级别)
    """

    def __init__(self):
        self._docx_available = self._check_docx()

    @staticmethod
    def _check_docx() -> bool:
        try:
            import docx  # noqa: F401
            return True
        except ImportError:
            return False

    def parse(self, file_path: str, **kwargs) -> ParsedDocument:
        """解析 .docx 文件。

        Args:
            file_path: 文件路径。
            **kwargs: 额外参数。

        Returns:
            ParsedDocument: 解析结果。
        """
        if not self._docx_available:
            raise ImportError(
                "python-docx 未安装，请运行 pip install python-docx"
            )

        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        import docx

        doc = docx.Document(file_path)
        paragraphs: List[str] = []
        tables: List[TableData] = []
        images: List[str] = []

        # 提取段落
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                paragraphs.append(text)

        # 提取表格
        for table in doc.tables:
            table_data = self._parse_table(table)
            tables.append(table_data)

        # 提取图片
        for rel in doc.part.rels.values():
            if "image" in rel.reltype:
                try:
                    image_data = rel.target_part.blob
                    from PIL import Image
                    import io

                    img = Image.open(io.BytesIO(image_data))
                    images.append(image_to_base64(img))
                except Exception:
                    continue

        full_text = "\n\n".join(paragraphs)
        page = ParsedPage(
            page_num=1,
            text=full_text,
            images=images,
            tables=tables,
        )

        metadata: Dict[str, Any] = {
            "paragraph_count": len(paragraphs),
            "table_count": len(tables),
            "image_count": len(images),
            "file_size": path.stat().st_size if path.exists() else 0,
            "parser": "DocxParser",
        }

        return ParsedDocument(
            filename=path.name,
            format=DocumentFormat.DOCX,
            pages=[page],
            full_text=full_text,
            metadata=metadata,
        )

    def _parse_table(self, table) -> TableData:
        """解析单个 docx 表格对象。"""
        rows = len(table.rows)
        cols = max(len(row.cells) for row in table.rows) if rows > 0 else 0

        headers: List[str] = []
        cells: List[TableCell] = []

        for row_idx, row in enumerate(table.rows):
            for col_idx, cell in enumerate(row.cells):
                cell_text = cell.text.strip()
                cells.append(
                    TableCell(
                        row=row_idx,
                        col=col_idx,
                        text=cell_text,
                    )
                )
                if row_idx == 0:
                    headers.append(cell_text)

        return TableData(rows=rows, cols=cols, cells=cells, headers=headers)