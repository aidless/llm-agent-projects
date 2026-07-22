"""文档索引器 - 文档分块与向量化。"""

import hashlib
import logging
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

from app.models import DocumentIndexResult, ParsedDocument

logger = logging.getLogger(__name__)


class DocumentChunk:
    """文档分块。"""

    def __init__(
        self,
        text: str,
        chunk_id: str = "",
        document_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        self.text = text
        self.chunk_id = chunk_id or str(uuid.uuid4())
        self.document_id = document_id
        self.metadata = metadata or {}

    def __repr__(self):
        return f"DocumentChunk(id={self.chunk_id[:8]}, doc={self.document_id[:8]}, len={len(self.text)})"


class SimpleVectorStore:
    """简单的内存向量存储 (基于 TF-IDF 相似度)。"""

    def __init__(self):
        self.chunks: List[DocumentChunk] = []
        self._vocabulary: Dict[str, int] = {}
        self._idf: Dict[str, float] = {}
        self._doc_vectors: List[Dict[str, float]] = []

    def add(self, chunks: List[DocumentChunk]):
        """添加分块到存储。"""
        self.chunks.extend(chunks)
        self._rebuild_index()

    def _tokenize(self, text: str) -> List[str]:
        """简单分词 (中英文混合)。"""
        # 英文词
        en_words = re.findall(r"[a-zA-Z]{2,}", text.lower())
        # 中文字 (简单按字切分)
        cn_chars = list(re.findall(r"[\u4e00-\u9fff]", text))
        # 中文词 (简单 2-gram)
        cn_words = []
        for i in range(len(cn_chars) - 1):
            cn_words.append(cn_chars[i] + cn_chars[i + 1])

        return en_words + cn_chars + cn_words

    def _rebuild_index(self):
        """重建倒排索引和 IDF。"""
        n = len(self.chunks)
        if n == 0:
            return

        # 构建 TF-IDF
        df: Dict[str, int] = {}
        self._doc_vectors = []

        for chunk in self.chunks:
            tokens = self._tokenize(chunk.text)
            tf: Dict[str, int] = {}
            for t in tokens:
                tf[t] = tf.get(t, 0) + 1

            for t in set(tokens):
                df[t] = df.get(t, 0) + 1

            self._doc_vectors.append(tf)

        # 计算 IDF
        self._idf = {}
        for term, freq in df.items():
            self._idf[term] = max(0.1, (n - freq + 0.5) / (freq + 0.5) + 1)

    def search(self, query: str, top_k: int = 5) -> List[Tuple[int, float]]:
        """搜索最相关的分块。

        Args:
            query: 查询文本。
            top_k: 返回前 k 个结果。

        Returns:
            List[Tuple[int, float]]: (分块索引, 相似度分数) 列表。
        """
        if not self.chunks:
            return []

        query_tokens = self._tokenize(query)
        query_tf: Dict[str, int] = {}
        for t in query_tokens:
            query_tf[t] = query_tf.get(t, 0) + 1

        scores = []
        for i, doc_tf in enumerate(self._doc_vectors):
            score = 0.0
            for term, qf in query_tf.items():
                if term in self._idf and term in doc_tf:
                    idf = self._idf[term]
                    tf_norm = doc_tf[term] / (1 + doc_tf[term])
                    score += qf * idf * tf_norm
            scores.append((i, score))

        scores.sort(key=lambda x: -x[1])
        return scores[:top_k]

    def delete_by_document(self, document_id: str):
        """删除指定文档的所有分块。"""
        self.chunks = [
            c for c in self.chunks if c.document_id != document_id
        ]
        self._rebuild_index()

    def count(self) -> int:
        return len(self.chunks)


class DocumentIndexer:
    """文档索引器。

    支持:
    - 文档分块 (按段落/固定长度)
    - 简单向量化 (TF-IDF)
    - 内存向量存储
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        vector_store: Optional[SimpleVectorStore] = None,
    ):
        """初始化文档索引器。

        Args:
            chunk_size: 分块大小 (字符数)。
            chunk_overlap: 分块重叠大小。
            vector_store: 向量存储实例。
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.store = vector_store or SimpleVectorStore()
        self._indexed_docs: Dict[str, str] = {}  # doc_id -> filename

    def index_document(
        self, document: ParsedDocument, document_id: str = ""
    ) -> DocumentIndexResult:
        """索引文档。

        Args:
            document: 解析后的文档。
            document_id: 文档 ID (可选，自动生成)。

        Returns:
            DocumentIndexResult: 索引结果。
        """
        doc_id = document_id or self._generate_doc_id(document.filename)

        # 分块
        chunks = self._chunk_document(document, doc_id)

        # 存储
        self.store.add(chunks)
        self._indexed_docs[doc_id] = document.filename

        return DocumentIndexResult(
            document_id=doc_id,
            filename=document.filename,
            chunks_count=len(chunks),
            status="success",
        )

    def _chunk_document(
        self, document: ParsedDocument, document_id: str
    ) -> List[DocumentChunk]:
        """对文档进行分块。

        优先按段落分块，超过 chunk_size 的段落再按长度切分。
        """
        chunks = []

        text = document.full_text
        if not text.strip():
            return chunks

        # 按段落分割
        paragraphs = re.split(r"\n\s*\n", text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        current_chunk = ""
        chunk_num = 0

        for para in paragraphs:
            # 如果单个段落超过 chunk_size，直接切分
            if len(para) > self.chunk_size:
                # 先保存当前 chunk
                if current_chunk.strip():
                    chunk_num += 1
                    chunks.append(
                        DocumentChunk(
                            text=current_chunk.strip(),
                            document_id=document_id,
                            metadata={
                                "chunk_num": chunk_num,
                                "filename": document.filename,
                            },
                        )
                    )
                    current_chunk = ""

                # 切分长段落
                sub_chunks = self._split_long_text(para, document.filename, document_id, chunk_num)
                for sc in sub_chunks:
                    chunk_num += 1
                    sc.metadata["chunk_num"] = chunk_num
                    chunks.append(sc)
            else:
                # 合并短段落
                if len(current_chunk) + len(para) + 2 > self.chunk_size:
                    if current_chunk.strip():
                        chunk_num += 1
                        chunks.append(
                            DocumentChunk(
                                text=current_chunk.strip(),
                                document_id=document_id,
                                metadata={
                                    "chunk_num": chunk_num,
                                    "filename": document.filename,
                                },
                            )
                        )
                    current_chunk = para
                else:
                    current_chunk = current_chunk + "\n\n" + para if current_chunk else para

        # 保存最后一个 chunk
        if current_chunk.strip():
            chunk_num += 1
            chunks.append(
                DocumentChunk(
                    text=current_chunk.strip(),
                    document_id=document_id,
                    metadata={
                        "chunk_num": chunk_num,
                        "filename": document.filename,
                    },
                )
            )

        return chunks

    def _split_long_text(
        self, text: str, filename: str, document_id: str, base_num: int
    ) -> List[DocumentChunk]:
        """切分长文本。"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            # 尝试在句号/换行处切分
            if end < len(text):
                # 查找最近的断句位置
                for sep in ["。", "\n", "！", "？", ".", "!", "?"]:
                    sep_pos = text.rfind(sep, start, end)
                    if sep_pos != -1:
                        end = sep_pos + 1
                        break

            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(
                    DocumentChunk(
                        text=chunk_text,
                        document_id=document_id,
                        metadata={
                            "chunk_num": base_num + len(chunks) + 1,
                            "filename": filename,
                        },
                    )
                )

            # 重叠
            start = end - self.chunk_overlap if end < len(text) else end

        return chunks

    @staticmethod
    def _generate_doc_id(filename: str) -> str:
        """生成文档 ID。"""
        unique_str = f"{filename}-{uuid.uuid4()}"
        return hashlib.md5(unique_str.encode()).hexdigest()[:16]

    def get_indexed_documents(self) -> Dict[str, str]:
        """获取已索引的文档列表。"""
        return dict(self._indexed_docs)

    def delete_document(self, document_id: str) -> bool:
        """删除指定文档的索引。"""
        if document_id in self._indexed_docs:
            self.store.delete_by_document(document_id)
            del self._indexed_docs[document_id]
            return True
        return False