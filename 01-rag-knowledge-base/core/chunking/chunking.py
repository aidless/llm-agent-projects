"""
文本分块模块
提供三种分块策略：固定长度分块、段落分块、语义分块
"""
import re
import math
from dataclasses import dataclass, field
from typing import List, Optional

from loguru import logger


@dataclass
class Chunk:
    """文本块数据结构"""

    content: str  # 文本块内容
    chunk_id: int  # 块编号
    start_index: int = 0  # 在原文中的起始位置
    end_index: int = 0  # 在原文中的结束位置
    metadata: dict = field(default_factory=dict)  # 元数据（来源文件、段落索引等）

    @property
    def char_count(self) -> int:
        """文本块字符数"""
        return len(self.content)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "content": self.content,
            "chunk_id": self.chunk_id,
            "start_index": self.start_index,
            "end_index": self.end_index,
            "metadata": self.metadata,
        }


class FixedLengthChunker:
    """
    固定长度分块策略

    按固定字符长度切分文本，保留指定重叠区域以避免上下文断裂
    """

    def __init__(self, chunk_size: int = 512, overlap: int = 64, separator: str = "\n"):
        """
        初始化固定长度分块器

        Args:
            chunk_size: 每个文本块的目标字符数
            overlap: 块之间的重叠字符数
            separator: 分隔符
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separator = separator

        if overlap >= chunk_size:
            raise ValueError(f"重叠长度 ({overlap}) 不能大于等于块大小 ({chunk_size})")

    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        """
        执行分块

        Args:
            text: 待分块的文本
            metadata: 附加元数据

        Returns:
            List[Chunk]: 文本块列表
        """
        if not text or not text.strip():
            return []

        meta = metadata or {}
        chunks = []
        step = self.chunk_size - self.overlap
        chunk_id = 0

        # 按步长滑动窗口切分
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(
                    Chunk(
                        content=chunk_text,
                        chunk_id=chunk_id,
                        start_index=start,
                        end_index=min(end, len(text)),
                        metadata={**meta, "strategy": "fixed_length"},
                    )
                )
                chunk_id += 1

            # 如果剩余文本不足一个步长，直接结束
            if start + step >= len(text):
                break
            start += step

        logger.debug(f"固定长度分块完成: {len(chunks)} 个块 (size={self.chunk_size}, overlap={self.overlap})")
        return chunks


class ParagraphChunker:
    """
    按段落分块策略

    以空行或换行符为段落分隔，合并过短的段落直到达到目标长度
    """

    def __init__(self, max_chunk_size: int = 512, min_chunk_size: int = 100):
        """
        初始化段落分块器

        Args:
            max_chunk_size: 单个块的最大字符数
            min_chunk_size: 单个块的最小字符数（过短段落会被合并）
        """
        self.max_chunk_size = max_chunk_size
        self.min_chunk_size = min_chunk_size

    def _split_paragraphs(self, text: str) -> List[str]:
        """
        将文本按段落拆分

        支持多种段落分隔方式：双换行、单换行后接空白
        """
        # 先按双换行分段
        raw_paragraphs = re.split(r"\n\s*\n", text)
        paragraphs = []

        for para in raw_paragraphs:
            stripped = para.strip()
            if stripped:
                paragraphs.append(stripped)

        # 如果只有一个大段（没有双换行），尝试按单换行分段
        if len(paragraphs) <= 1 and "\n" in text:
            lines = text.split("\n")
            long_lines = [l.strip() for l in lines if l.strip() and len(l.strip()) > 20]
            if len(long_lines) > 2:
                paragraphs = long_lines

        return paragraphs

    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        """
        执行段落分块

        将段落合并/拆分以控制块大小在合理范围内

        Args:
            text: 待分块的文本
            metadata: 附加元数据

        Returns:
            List[Chunk]: 文本块列表
        """
        if not text or not text.strip():
            return []

        meta = metadata or {}
        paragraphs = self._split_paragraphs(text)
        chunks = []
        chunk_id = 0
        current_content = ""
        current_start = 0

        for para in paragraphs:
            # 如果当前累积内容加上新段落超过上限，先保存当前块
            if current_content and len(current_content) + len(para) > self.max_chunk_size:
                if current_content.strip():
                    chunks.append(
                        Chunk(
                            content=current_content.strip(),
                            chunk_id=chunk_id,
                            start_index=current_start,
                            end_index=current_start + len(current_content),
                            metadata={**meta, "strategy": "paragraph"},
                        )
                    )
                    chunk_id += 1

                # 新段落开始新的块
                current_start = text.find(para, current_start) if para in text else current_start
                current_content = para + "\n\n"
            else:
                if not current_content:
                    current_start = text.find(para) if para in text else 0
                current_content += para + "\n\n"

        # 保存最后一个块
        if current_content.strip():
            chunks.append(
                Chunk(
                    content=current_content.strip(),
                    chunk_id=chunk_id,
                    start_index=current_start,
                    end_index=current_start + len(current_content),
                    metadata={**meta, "strategy": "paragraph"},
                )
            )

        logger.debug(f"段落分块完成: {len(chunks)} 个块")
        return chunks


class SemanticChunker:
    """
    语义分块策略

    通过分析句子间的语义相似度来决定分块边界
    语义相似度低的句子之间是自然的分块边界

    该实现使用基于标点符号和句子长度的启发式方法进行语义边界检测，
    可结合 embedding 模型进行真正的语义分块
    """

    def __init__(
        self,
        chunk_size: int = 512,
        overlap: int = 64,
        similarity_threshold: float = 0.3,
        embedding_model=None,
    ):
        """
        初始化语义分块器

        Args:
            chunk_size: 目标块大小
            overlap: 块间重叠
            similarity_threshold: 语义相似度阈值，低于此值则断开
            embedding_model: 可选的 embedding 模型，用于计算真正的语义相似度
        """
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model

    def _split_sentences(self, text: str) -> List[str]:
        """
        将文本拆分为句子列表

        支持中英文标点作为句子分隔符
        """
        # 使用正则按中英文句号、问号、感叹号分割
        # 保留分隔符
        sentence_pattern = r'(?<=[。！？.!?])\s*'
        sentences = re.split(sentence_pattern, text)
        # 过滤空句子
        return [s.strip() for s in sentences if s.strip()]

    def _compute_boundary_scores(self, sentences: List[str]) -> List[float]:
        """
        计算相邻句子之间的边界分数

        分数越低表示两句话之间的语义差异越大，适合作为分块边界

        Args:
            sentences: 句子列表

        Returns:
            List[float]: 边界分数列表（长度为 len(sentences) - 1）
        """
        if len(sentences) <= 1:
            return []

        if self.embedding_model is not None:
            return self._compute_semantic_boundaries(sentences)
        else:
            return self._compute_heuristic_boundaries(sentences)

    def _compute_heuristic_boundaries(self, sentences: List[str]) -> List[float]:
        """
        基于启发式规则的边界检测

        考虑因素：
        1. 句子长度差异（长度突变暗示话题转换）
        2. 标点符号类型（问号、感叹号后更可能断开）
        3. 关键词线索（如"总结"、"首先"、"其次"等）
        """
        scores = []
        transition_keywords = [
            "总结", "综上所述", "首先", "其次", "最后", "另外",
            "总之", "因此", "综上", "接下来", "此外", "另外",
            "however", "therefore", "furthermore", "in conclusion",
            "first", "second", "finally", "in summary",
        ]

        for i in range(len(sentences) - 1):
            s1 = sentences[i]
            s2 = sentences[i + 1]

            # 基础分（长句之间相似度更高）
            base_score = 0.5

            # 长度相似度因子（0~1，越接近1表示长度越相似）
            len1, len2 = len(s1), len(s2)
            max_len = max(len1, len2, 1)
            length_similarity = 1.0 - abs(len1 - len2) / max_len
            base_score = 0.3 + 0.7 * length_similarity

            # 检查句子2是否以过渡词开头
            for keyword in transition_keywords:
                if s2.startswith(keyword):
                    base_score -= 0.3  # 过渡词出现，降低相似度
                    break

            # 前一句以问号或感叹号结尾，后一句大概率是新的话题
            if s1.rstrip() and s1.rstrip()[-1] in "！?！?":
                base_score -= 0.2

            # 首字符距离（如果两句话开头字符完全不同，可能是新话题）
            if s1 and s2:
                if s1[0] != s2[0] and not s2[0].islower():
                    base_score -= 0.1

            # 限制在 [0, 1] 范围内
            score = max(0.0, min(1.0, base_score))
            scores.append(score)

        return scores

    def _compute_semantic_boundaries(self, sentences: List[str]) -> List[float]:
        """
        基于 embedding 模型计算语义边界分数

        计算相邻句子的余弦相似度

        Args:
            sentences: 句子列表

        Returns:
            List[float]: 相似度分数列表
        """
        try:
            # 批量获取所有句子的向量
            embeddings = self.embedding_model.embed_documents(sentences)
            scores = []

            for i in range(len(embeddings) - 1):
                sim = self._cosine_similarity(embeddings[i], embeddings[i + 1])
                scores.append(float(sim))

            return scores
        except Exception as e:
            logger.warning(f"语义边界计算失败，回退到启发式方法: {e}")
            return self._compute_heuristic_boundaries(sentences)

    @staticmethod
    def _cosine_similarity(v1: list, v2: list) -> float:
        """计算两个向量的余弦相似度"""
        import numpy as np

        a = np.array(v1)
        b = np.array(v2)
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))

    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        """
        执行语义分块

        流程：
        1. 将文本拆分为句子
        2. 计算相邻句子间的边界分数
        3. 找到低分边界作为分块点
        4. 合并句子为文本块，控制块大小

        Args:
            text: 待分块的文本
            metadata: 附加元数据

        Returns:
            List[Chunk]: 文本块列表
        """
        if not text or not text.strip():
            return []

        meta = metadata or {}
        sentences = self._split_sentences(text)

        if len(sentences) <= 1:
            # 文本太短，无需分块
            return [
                Chunk(
                    content=text.strip(),
                    chunk_id=0,
                    start_index=0,
                    end_index=len(text),
                    metadata={**meta, "strategy": "semantic"},
                )
            ]

        # 计算边界分数
        boundary_scores = self._compute_boundary_scores(sentences)

        # 找到分块边界点（分数低于阈值的相邻句子之间断开）
        boundaries = [0]  # 第一个句子总是一个块的开始
        for i, score in enumerate(boundary_scores):
            if score < self.similarity_threshold:
                boundaries.append(i + 1)  # 在下一个句子之前断开
        boundaries.append(len(sentences))  # 最后一个句子的后面

        # 合并句子为文本块
        chunks = []
        chunk_id = 0
        current_start = 0

        for i in range(len(boundaries) - 1):
            start = boundaries[i]
            end = boundaries[i + 1]
            chunk_sentences = sentences[start:end]
            chunk_text = "".join(chunk_sentences)

            if not chunk_text.strip():
                continue

            # 如果块太大，使用固定长度二次切分
            if len(chunk_text) > self.chunk_size * 1.5:
                sub_chunker = FixedLengthChunker(
                    chunk_size=self.chunk_size, overlap=self.overlap
                )
                sub_chunks = sub_chunker.chunk(chunk_text, meta)
                for sc in sub_chunks:
                    sc.chunk_id = chunk_id
                    sc.metadata["strategy"] = "semantic+fixed"
                    chunks.append(sc)
                    chunk_id += 1
            else:
                # 计算在原文中的位置
                start_pos = text.find(chunk_sentences[0]) if chunk_sentences else current_start
                end_pos = start_pos + len(chunk_text)

                chunks.append(
                    Chunk(
                        content=chunk_text.strip(),
                        chunk_id=chunk_id,
                        start_index=start_pos,
                        end_index=end_pos,
                        metadata={
                            **meta,
                            "strategy": "semantic",
                            "sentence_range": f"{start}-{end}",
                        },
                    )
                )
                chunk_id += 1
                current_start = end_pos

        logger.debug(f"语义分块完成: {len(chunks)} 个块 (阈值={self.similarity_threshold})")
        return chunks


class Chunker:
    """
    统一分块器入口

    根据配置的策略名称自动选择对应的分块器

    使用示例:
        chunker = Chunker(strategy="semantic", chunk_size=512)
        chunks = chunker.chunk(text, metadata={"source": "doc.pdf"})
    """

    STRATEGY_MAP = {
        "fixed": FixedLengthChunker,
        "fixed_length": FixedLengthChunker,
        "paragraph": ParagraphChunker,
        "semantic": SemanticChunker,
    }

    def __init__(
        self,
        strategy: str = "semantic",
        chunk_size: int = 512,
        overlap: int = 64,
        separator: str = "\n",
        embedding_model=None,
    ):
        """
        初始化分块器

        Args:
            strategy: 分块策略名称 (fixed/fixed_length/paragraph/semantic)
            chunk_size: 目标块大小
            overlap: 块间重叠字符数
            separator: 分隔符
            embedding_model: 可选的 embedding 模型（用于语义分块）
        """
        self.strategy = strategy
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.separator = separator

        # 根据策略选择分块器
        chunker_class = self.STRATEGY_MAP.get(strategy)
        if chunker_class is None:
            raise ValueError(
                f"未知的分块策略: {strategy}，"
                f"可选: {list(self.STRATEGY_MAP.keys())}"
            )

        if strategy in ("fixed", "fixed_length"):
            self._chunker = chunker_class(
                chunk_size=chunk_size, overlap=overlap, separator=separator
            )
        elif strategy == "paragraph":
            self._chunker = chunker_class(
                max_chunk_size=chunk_size, min_chunk_size=chunk_size // 5
            )
        elif strategy == "semantic":
            self._chunker = chunker_class(
                chunk_size=chunk_size,
                overlap=overlap,
                embedding_model=embedding_model,
            )
        else:
            self._chunker = chunker_class()

        logger.info(f"分块器初始化完成: 策略={strategy}, 块大小={chunk_size}")

    def chunk(self, text: str, metadata: Optional[dict] = None) -> List[Chunk]:
        """
        执行分块

        Args:
            text: 待分块的文本
            metadata: 附加元数据

        Returns:
            List[Chunk]: 文本块列表
        """
        if not text or not text.strip():
            logger.warning("输入文本为空，返回空列表")
            return []

        return self._chunker.chunk(text, metadata)

    def chunk_document(self, parsed_doc) -> List[Chunk]:
        """
        对解析后的文档进行分块

        将文档元信息（文件名、标题等）注入到每个块的 metadata 中

        Args:
            parsed_doc: ParsedDocument 对象

        Returns:
            List[Chunk]: 文本块列表
        """
        doc_meta = {
            "source_file": parsed_doc.filename,
            "doc_title": parsed_doc.title,
            "doc_type": parsed_doc.file_type,
            "raw_path": parsed_doc.raw_path,
        }

        chunks = self.chunk(parsed_doc.content, metadata=doc_meta)

        # 为每个块添加文档级别的元数据
        for chunk in chunks:
            chunk.metadata.update({
                "total_chunks": len(chunks),
                "doc_total_chars": parsed_doc.total_chars,
            })

        logger.info(f"文档 [{parsed_doc.filename}] 分块完成: 共 {len(chunks)} 个块")
        return chunks