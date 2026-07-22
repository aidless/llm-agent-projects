# -*- coding: utf-8 -*-
"""
向量数据库管理平台 - 文件解析器
支持 TXT/MD/JSON/CSV 格式的文件批量导入与解析
"""
import json
import csv
import logging
from pathlib import Path
from typing import List, Dict, Any, Generator

logger = logging.getLogger(__name__)


class ParsedDocument:
    """解析后的文档对象"""

    def __init__(self, content: str, metadata: Dict[str, Any]):
        """
        初始化解析后的文档

        Args:
            content: 文档文本内容
            metadata: 文档元数据（来源、类型等）
        """
        self.content = content
        self.metadata = metadata

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "content": self.content,
            "metadata": self.metadata,
        }


class FileParser:
    """
    文件解析器
    根据文件扩展名自动选择解析策略，将文件内容转换为统一的文档列表
    """

    # 支持的文件扩展名
    SUPPORTED_EXTENSIONS = {".txt", ".md", ".json", ".csv"}

    def __init__(self):
        """初始化文件解析器"""
        pass

    def parse_file(self, file_path: str) -> List[ParsedDocument]:
        """
        解析单个文件

        Args:
            file_path: 文件路径

        Returns:
            解析后的文档列表

        Raises:
            ValueError: 不支持的文件格式
            FileNotFoundError: 文件不存在
        """
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"文件不存在: {file_path}")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"不支持的文件格式: {ext}，"
                f"支持的格式: {', '.join(self.SUPPORTED_EXTENSIONS)}"
            )

        base_metadata = {
            "source": str(path.name),
            "source_path": str(path.resolve()),
            "file_type": ext.lstrip("."),
        }

        # 根据文件类型选择解析器
        parsers = {
            ".txt": self._parse_text,
            ".md": self._parse_markdown,
            ".json": self._parse_json,
            ".csv": self._parse_csv,
        }
        parser_func = parsers[ext]
        return parser_func(path, base_metadata)

    def parse_directory(self, dir_path: str) -> Generator[ParsedDocument, None, None]:
        """
        递归解析目录中的所有支持格式文件

        Args:
            dir_path: 目录路径

        Yields:
            解析后的文档对象
        """
        path = Path(dir_path)
        if not path.is_dir():
            raise ValueError(f"路径不是目录: {dir_path}")

        for file_path in sorted(path.rglob("*")):
            if file_path.is_file() and file_path.suffix.lower() in self.SUPPORTED_EXTENSIONS:
                try:
                    docs = self.parse_file(str(file_path))
                    for doc in docs:
                        yield doc
                except Exception as e:
                    logger.warning(f"解析文件 {file_path} 失败: {e}")
                    continue

    def _parse_text(
        self, path: Path, base_metadata: Dict[str, Any]
    ) -> List[ParsedDocument]:
        """
        解析纯文本文件
        每个段落作为一个独立文档

        Args:
            path: 文件路径
            base_metadata: 基础元数据

        Returns:
            文档列表
        """
        content = path.read_text(encoding="utf-8")
        # 按空行分段
        paragraphs = [p.strip() for p in content.split("\n\n") if p.strip()]
        if not paragraphs:
            # 如果没有空行分隔，按换行符分段
            paragraphs = [p.strip() for p in content.split("\n") if p.strip()]

        documents = []
        for i, para in enumerate(paragraphs):
            metadata = {**base_metadata, "paragraph_index": i}
            documents.append(ParsedDocument(content=para, metadata=metadata))
        return documents

    def _parse_markdown(
        self, path: Path, base_metadata: Dict[str, Any]
    ) -> List[ParsedDocument]:
        """
        解析 Markdown 文件
        按标题层级分段，每个章节（或段落）作为一个文档

        Args:
            path: 文件路径
            base_metadata: 基础元数据

        Returns:
            文档列表
        """
        content = path.read_text(encoding="utf-8")
        lines = content.split("\n")

        documents = []
        current_section = []
        current_heading = "开头"
        section_index = 0

        for line in lines:
            stripped = line.strip()
            # 检测 Markdown 标题（# ## ### 等）
            if stripped.startswith("#"):
                # 如果当前有内容，保存为一个文档
                if current_section:
                    section_text = "\n".join(current_section).strip()
                    if section_text:
                        metadata = {
                            **base_metadata,
                            "heading": current_heading,
                            "section_index": section_index,
                        }
                        documents.append(ParsedDocument(content=section_text, metadata=metadata))
                        section_index += 1
                    current_section = []
                # 提取标题文本（去除 # 符号）
                current_heading = stripped.lstrip("#").strip()
            else:
                if stripped:
                    current_section.append(line)

        # 处理最后一个章节
        if current_section:
            section_text = "\n".join(current_section).strip()
            if section_text:
                metadata = {
                    **base_metadata,
                    "heading": current_heading,
                    "section_index": section_index,
                }
                documents.append(ParsedDocument(content=section_text, metadata=metadata))

        return documents

    def _parse_json(
        self, path: Path, base_metadata: Dict[str, Any]
    ) -> List[ParsedDocument]:
        """
        解析 JSON 文件
        支持两种格式：
        1. 对象列表 [{"title": "...", "content": "..."}, ...]
        2. 单个对象 {"key1": "value1", "key2": "value2"}

        Args:
            path: 文件路径
            base_metadata: 基础元数据

        Returns:
            文档列表
        """
        content = path.read_text(encoding="utf-8")
        data = json.loads(content)

        documents = []

        if isinstance(data, list):
            # 列表格式：每个元素一个文档
            for i, item in enumerate(data):
                if isinstance(item, dict):
                    # 优先使用 "content" 或 "text" 字段作为文档内容
                    doc_content = (
                        item.get("content")
                        or item.get("text")
                        or item.get("body")
                        or json.dumps(item, ensure_ascii=False)
                    )
                    metadata = {**base_metadata, "item_index": i}
                    # 将原始 JSON 中的其他字段作为元数据
                    for key, value in item.items():
                        if key not in ("content", "text", "body") and isinstance(value, (str, int, float)):
                            metadata[key] = value
                    documents.append(ParsedDocument(content=str(doc_content), metadata=metadata))
                else:
                    documents.append(
                        ParsedDocument(
                            content=str(item),
                            metadata={**base_metadata, "item_index": i}
                        )
                    )
        elif isinstance(data, dict):
            # 单个对象格式：每个键值对一个文档
            for key, value in data.items():
                if isinstance(value, (str, list)):
                    doc_content = json.dumps(value, ensure_ascii=False) if isinstance(value, list) else str(value)
                    metadata = {**base_metadata, "json_key": key}
                    documents.append(ParsedDocument(content=doc_content, metadata=metadata))
        else:
            # 其他类型：整体作为一个文档
            documents.append(
                ParsedDocument(
                    content=json.dumps(data, ensure_ascii=False),
                    metadata=base_metadata
                )
            )

        return documents

    def _parse_csv(
        self, path: Path, base_metadata: Dict[str, Any]
    ) -> List[ParsedDocument]:
        """
        解析 CSV 文件
        每行数据作为一个文档，将所有列拼接为文本

        Args:
            path: 文件路径
            base_metadata: 基础元数据

        Returns:
            文档列表
        """
        content = path.read_text(encoding="utf-8")
        reader = csv.DictReader(content.splitlines())

        documents = []
        for i, row in enumerate(reader):
            # 将每行数据拼接为文本
            parts = []
            row_metadata = {**base_metadata, "row_index": i}
            for key, value in row.items():
                if value:
                    parts.append(f"{key}: {value}")
                    # 将数值型字段添加到元数据
                    if isinstance(value, (int, float)):
                        row_metadata[key] = value
            doc_content = "\n".join(parts)
            if doc_content.strip():
                documents.append(ParsedDocument(content=doc_content, metadata=row_metadata))

        return documents
