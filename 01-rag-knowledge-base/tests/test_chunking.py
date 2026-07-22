"""
文本分块测试
"""
import pytest
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.chunking.chunking import (
    FixedLengthChunker,
    ParagraphChunker,
    SemanticChunker,
    Chunker,
    Chunk,
)


# 测试用文本
SAMPLE_TEXT_SHORT = "这是一个简短的测试文本。"

SAMPLE_TEXT_LONG = """RAG（Retrieval-Augmented Generation）是一种结合了检索和生成的AI技术。

它的核心思想是：在生成回答之前，先从知识库中检索相关的文档段落，然后将这些段落作为上下文提供给语言模型。

这种方式可以有效解决大语言模型的几个关键问题：首先，它可以减少模型幻觉，因为回答基于真实的文档内容。其次，它可以让模型访问训练数据之外的知识。第三，通过引用来源，用户可以验证回答的准确性。

RAG 系统的主要组件包括：文档解析器、文本分块器、向量化模型、向量数据库、检索器和语言模型。

文档解析器负责将各种格式的文档转换为纯文本。文本分块器将长文档拆分为适合检索的小段落。向量化模型将文本转换为数值向量。向量数据库存储这些向量并提供高效的相似度搜索。检索器根据用户查询找到最相关的文档段落。语言模型基于检索到的上下文生成最终回答。

总结来说，RAG 技术是当前企业知识库问答系统的核心方案。"""

SAMPLE_TEXT_PARAGRAPHS = """
第一段内容：这是一个测试段落。

第二段内容：这是另一个测试段落，内容更长一些，包含更多的信息。

第三段内容：这第三个段落包含了关于RAG系统的描述。
""".strip()


class TestFixedLengthChunker:
    """固定长度分块测试"""

    def test_basic_chunking(self):
        """测试基本分块功能"""
        chunker = FixedLengthChunker(chunk_size=100, overlap=20)
        chunks = chunker.chunk(SAMPLE_TEXT_LONG)

        assert len(chunks) > 1
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert len(chunk.content) > 0
            assert chunk.metadata["strategy"] == "fixed_length"

    def test_short_text_single_chunk(self):
        """短文本应该产生一个块"""
        chunker = FixedLengthChunker(chunk_size=100, overlap=20)
        chunks = chunker.chunk(SAMPLE_TEXT_SHORT)

        assert len(chunks) == 1
        assert SAMPLE_TEXT_SHORT in chunks[0].content

    def test_empty_text(self):
        """空文本应返回空列表"""
        chunker = FixedLengthChunker(chunk_size=100, overlap=20)
        chunks = chunker.chunk("")
        assert chunks == []

    def test_overlap(self):
        """测试块间重叠"""
        chunker = FixedLengthChunker(chunk_size=50, overlap=10)
        chunks = chunker.chunk(SAMPLE_TEXT_LONG)

        if len(chunks) >= 2:
            # 相邻块应有重叠内容
            chunk1_end = chunks[0].content[-10:]
            chunk2_start = chunks[1].content[:10]
            # 由于分块基于字符位置，重叠应该是完全一致的
            assert chunks[1].start_index < chunks[0].end_index

    def test_invalid_overlap(self):
        """重叠大于块大小应报错"""
        with pytest.raises(ValueError, match="重叠长度"):
            FixedLengthChunker(chunk_size=50, overlap=60)


class TestParagraphChunker:
    """段落分块测试"""

    def test_basic_paragraph_chunking(self):
        """测试基本段落分块"""
        # 使用较小的 max_chunk_size 确保产生多个块
        chunker = ParagraphChunker(max_chunk_size=40, min_chunk_size=5)
        chunks = chunker.chunk(SAMPLE_TEXT_PARAGRAPHS)

        assert len(chunks) >= 2
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.metadata["strategy"] == "paragraph"

    def test_empty_text(self):
        """空文本"""
        chunker = ParagraphChunker()
        chunks = chunker.chunk("")
        assert chunks == []


class TestSemanticChunker:
    """语义分块测试"""

    def test_basic_semantic_chunking(self):
        """测试基本语义分块"""
        chunker = SemanticChunker(chunk_size=200, overlap=30)
        chunks = chunker.chunk(SAMPLE_TEXT_LONG)

        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, Chunk)
            assert chunk.metadata.get("strategy") in ("semantic", "semantic+fixed")

    def test_short_text(self):
        """短文本"""
        chunker = SemanticChunker(chunk_size=200)
        chunks = chunker.chunk(SAMPLE_TEXT_SHORT)
        assert len(chunks) == 1


class TestChunker:
    """统一分块器入口测试"""

    def test_fixed_strategy(self):
        """固定长度策略"""
        chunker = Chunker(strategy="fixed", chunk_size=100, overlap=20)
        chunks = chunker.chunk(SAMPLE_TEXT_LONG)
        assert len(chunks) > 1

    def test_paragraph_strategy(self):
        """段落策略"""
        chunker = Chunker(strategy="paragraph", chunk_size=40)
        chunks = chunker.chunk(SAMPLE_TEXT_PARAGRAPHS)
        assert len(chunks) >= 2

    def test_semantic_strategy(self):
        """语义策略"""
        chunker = Chunker(strategy="semantic", chunk_size=200)
        chunks = chunker.chunk(SAMPLE_TEXT_LONG)
        assert len(chunks) >= 1

    def test_unknown_strategy(self):
        """未知策略应报错"""
        with pytest.raises(ValueError, match="未知"):
            Chunker(strategy="unknown_strategy")

    def test_chunk_to_dict(self):
        """Chunk to_dict 方法"""
        chunker = Chunker(strategy="fixed", chunk_size=100)
        chunks = chunker.chunk(SAMPLE_TEXT_SHORT)
        d = chunks[0].to_dict()
        assert "content" in d
        assert "chunk_id" in d
        assert "metadata" in d
