# -*- coding: utf-8 -*-
"""
向量数据库管理平台 - 文档分块器
支持多种分块策略：固定长度、按句子、按段落、滑动窗口
可调整分块大小和重叠比例
"""
import re
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from services.file_parser import ParsedDocument

logger = logging.getLogger(__name__)


# 中英文句子分割正则
_SENTENCE_PATTERN = re.compile(r'(?<=[。！？\n.!?])\s+')


@dataclass
class ChunkConfig:
    """分块配置"""
    chunk_size: int = 500          # 每个分块的目标大小（字符数）
    chunk_overlap: int = 50        # 相邻分块之间的重叠字符数
    strategy: str = "fixed"       # 分块策略：fixed, sentence, paragraph, sliding_window
    separator: str = "\n"          # 分隔符（用于 fixed 策略）


@dataclass
class TextChunk:
    """文本分块"""
    text: str                                    # 分块文本内容
    chunk_index: int                             # 分块序号
    start_char: int                              # 在原文中的起始位置
    end_char: int                                # 在原文中的结束位置
    metadata: Dict[str, Any] = field(default_factory=dict)  # 分块元数据


class DocumentChunker:
    """
    文档分块器
    提供多种分块策略，将长文档拆分为适合嵌入的小块
    """

    # 支持的分块策略
    SUPPORTED_STRATEGIES = ["fixed", "sentence", "paragraph", "sliding_window"]

    def __init__(self, config: Optional[ChunkConfig] = None):
        """
        初始化分块器

        Args:
            config: 分块配置，为空时使用默认配置
        """
        self.config = config or ChunkConfig()
        if self.config.strategy not in self.SUPPORTED_STRATEGIES:
            raise ValueError(
                f"不支持的分块策略: {self.config.strategy}，"
                f"支持的策略: {', '.join(self.SUPPORTED_STRATEGIES)}"
            )

    def chunk_document(self, doc: ParsedDocument) -> List[TextChunk]:
        """
        对单个解析文档进行分块

        Args:
            doc: 解析后的文档对象

        Returns:
            分块列表
        """
        text = doc.content
        if not text or not text.strip():
            return []

        # 根据策略选择分块方法
        strategies = {
            "fixed": self._chunk_fixed,
            "sentence": self._chunk_by_sentence,
            "paragraph": self._chunk_by_paragraph,
            "sliding_window": self._chunk_sliding_window,
        }
        chunk_func = strategies[self.config.strategy]
        chunks = chunk_func(text)

        # 为每个分块附加原始文档元数据
        for chunk in chunks:
            chunk.metadata = {**doc.metadata, "chunk_index": chunk.chunk_index}

        return chunks

    def chunk_documents(self, docs: List[ParsedDocument]) -> List[TextChunk]:
        """
        批量分块多个文档

        Args:
            docs: 解析后的文档列表

        Returns:
            所有文档的分块列表
        """
        all_chunks = []
        global_index = 0
        for doc in docs:
            chunks = self.chunk_document(doc)
            for chunk in chunks:
                chunk.chunk_index = global_index
                global_index += 1
            all_chunks.extend(chunks)
        return all_chunks

    def _chunk_fixed(self, text: str) -> List[TextChunk]:
        """
        固定长度分块
        按指定字符数分割文本，保留分隔符处的完整性

        Args:
            text: 原始文本

        Returns:
            分块列表
        """
        chunks = []
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap
        separator = self.config.separator

        # 先按分隔符分割
        segments = text.split(separator)

        current_chunk_segments = []
        current_length = 0

        for segment in segments:
            segment_len = len(segment) + len(separator) if separator else len(segment)

            # 如果当前块加上这个段落不超过目标大小，则加入
            if current_length + segment_len <= chunk_size or not current_chunk_segments:
                current_chunk_segments.append(segment)
                current_length += segment_len
            else:
                # 保存当前块
                chunk_text = separator.join(current_chunk_segments).strip()
                if chunk_text:
                    start = self._find_position(text, chunk_text)
                    chunks.append(TextChunk(
                        text=chunk_text,
                        chunk_index=0,  # 稍后更新
                        start_char=start,
                        end_char=start + len(chunk_text),
                    ))

                # 从当前块末尾保留重叠部分
                overlap_text = separator.join(current_chunk_segments)
                overlap_segments = []
                overlap_len = 0
                for seg in reversed(current_chunk_segments):
                    seg_len = len(seg) + len(separator)
                    if overlap_len + seg_len > overlap:
                        break
                    overlap_segments.insert(0, seg)
                    overlap_len += seg_len

                current_chunk_segments = overlap_segments + [segment]
                current_length = sum(len(s) + len(separator) for s in current_chunk_segments)

        # 处理最后一个块
        if current_chunk_segments:
            chunk_text = separator.join(current_chunk_segments).strip()
            if chunk_text:
                start = self._find_position(text, chunk_text)
                chunks.append(TextChunk(
                    text=chunk_text,
                    chunk_index=0,
                    start_char=start,
                    end_char=start + len(chunk_text),
                ))

        return chunks

    def _chunk_by_sentence(self, text: str) -> List[TextChunk]:
        """
        按句子分块
        以句子为单位分割，尽量在句子边界处断开

        Args:
            text: 原始文本

        Returns:
            分块列表
        """
        # 按句子边界分割
        sentences = _SENTENCE_PATTERN.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            # 如果没有检测到句子边界，回退到固定分块
            return self._chunk_fixed(text)

        chunks = []
        current_sentences = []
        current_length = 0

        for sentence in sentences:
            if current_length + len(sentence) > self.config.chunk_size and current_sentences:
                # 保存当前块
                chunk_text = "".join(current_sentences).strip()
                if chunk_text:
                    start = self._find_position(text, chunk_text)
                    chunks.append(TextChunk(
                        text=chunk_text,
                        chunk_index=0,
                        start_char=start,
                        end_char=start + len(chunk_text),
                    ))
                # 保留重叠句子
                overlap_text = ""
                overlap_sents = []
                for s in reversed(current_sentences):
                    if len(overlap_text) + len(s) > self.config.chunk_overlap:
                        break
                    overlap_sents.insert(0, s)
                    overlap_text = s + overlap_text
                current_sentences = overlap_sents + [sentence]
                current_length = sum(len(s) for s in current_sentences)
            else:
                current_sentences.append(sentence)
                current_length += len(sentence)

        # 处理最后一个块
        if current_sentences:
            chunk_text = "".join(current_sentences).strip()
            if chunk_text:
                start = self._find_position(text, chunk_text)
                chunks.append(TextChunk(
                    text=chunk_text,
                    chunk_index=0,
                    start_char=start,
                    end_char=start + len(chunk_text),
                ))

        return chunks

    def _chunk_by_paragraph(self, text: str) -> List[TextChunk]:
        """
        按段落分块
        以空行分隔的段落为单位分割

        Args:
            text: 原始文本

        Returns:
            分块列表
        """
        paragraphs = re.split(r'\n\s*\n', text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        if not paragraphs:
            return self._chunk_fixed(text)

        chunks = []
        current_paras = []
        current_length = 0

        for para in paragraphs:
            if current_length + len(para) > self.config.chunk_size and current_paras:
                chunk_text = "\n\n".join(current_paras).strip()
                if chunk_text:
                    start = self._find_position(text, chunk_text)
                    chunks.append(TextChunk(
                        text=chunk_text,
                        chunk_index=0,
                        start_char=start,
                        end_char=start + len(chunk_text),
                    ))
                # 保留重叠段落
                overlap_len = 0
                overlap_paras = []
                for p in reversed(current_paras):
                    if overlap_len + len(p) > self.config.chunk_overlap:
                        break
                    overlap_paras.insert(0, p)
                    overlap_len += len(p)
                current_paras = overlap_paras + [para]
                current_length = sum(len(p) for p in current_paras)
            else:
                current_paras.append(para)
                current_length += len(para)

        if current_paras:
            chunk_text = "\n\n".join(current_paras).strip()
            if chunk_text:
                start = self._find_position(text, chunk_text)
                chunks.append(TextChunk(
                    text=chunk_text,
                    chunk_index=0,
                    start_char=start,
                    end_char=start + len(chunk_text),
                ))

        return chunks

    def _chunk_sliding_window(self, text: str) -> List[TextChunk]:
        """
        滑动窗口分块
        以固定步长和窗口大小切分文本，产生更多重叠

        Args:
            text: 原始文本

        Returns:
            分块列表
        """
        chunks = []
        chunk_size = self.config.chunk_size
        step = max(1, chunk_size - self.config.chunk_overlap)

        if step <= 0:
            step = 1

        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(TextChunk(
                    text=chunk_text,
                    chunk_index=0,
                    start_char=start,
                    end_char=end,
                ))
            start += step
            if end >= len(text):
                break

        return chunks

    def _find_position(self, text: str, sub_text: str) -> int:
        """
        在原文中查找子文本的起始位置

        Args:
            text: 原始文本
            sub_text: 子文本

        Returns:
            起始字符位置
        """
        pos = text.find(sub_text)
        return pos if pos >= 0 else 0

    @staticmethod
    def get_strategy_descriptions() -> Dict[str, str]:
        """
        获取所有分块策略的描述信息

        Returns:
            策略名到描述的映射
        """
        return {
            "fixed": "固定长度分块：按指定字符数分割文本",
            "sentence": "按句子分块：在句子边界处断开，保留语义完整性",
            "paragraph": "按段落分块：以空行分隔的段落为单位分割",
            "sliding_window": "滑动窗口分块：固定步长和窗口大小，产生更多重叠",
        }
