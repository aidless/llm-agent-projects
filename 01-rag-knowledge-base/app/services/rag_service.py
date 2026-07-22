"""
RAG 核心服务
编排文档解析 -> 分块 -> 向量化 -> 存储 -> 检索 -> 重排序 -> 引用溯源 -> LLM 生成 的完整流程
"""
import json
import time
from typing import List, Dict, Any, Optional, Generator

from loguru import logger

from core.config import settings
from core.parsers.document_parser import DocumentParser, ParsedDocument
from core.chunking.chunking import Chunker, Chunk
from core.embeddings.embeddings import EmbeddingManager, get_embedding_manager
from core.retrieval.vector_store import VectorStoreManager
from core.retrieval.retriever import HybridRetriever, RetrievalResult
from core.retrieval.citation import CitationExtractor, CitationSource
from core.reranker.reranker import RerankerManager
from core.llm.llm import LLMManager, get_llm_manager

# RAG 系统提示词
RAG_SYSTEM_PROMPT = """你是一个专业的企业知识库问答助手。你的任务是基于提供的参考资料，准确、完整地回答用户的问题。

回答规则：
1. 优先使用参考资料中的信息进行回答，确保回答的准确性
2. 如果参考资料中没有相关信息，请明确告知用户，并基于你的知识给出补充回答（需标注）
3. 回答应结构清晰、逻辑严谨、语言简洁
4. 使用 Markdown 格式化回答（如标题、列表、加粗等）
5. 如果涉及数据、日期、人名等关键信息，请确保准确引用
"""

RAG_USER_PROMPT_TEMPLATE = """请基于以下参考资料回答用户的问题。

## 参考资料
{context}

## 用户问题
{question}
{citation_instruction}
"""


class RAGService:
    """
    RAG 核心服务

    提供文档管理和问答的完整功能

    使用示例:
        service = RAGService()
        service.init_components()  # 初始化组件
        service.upload_document("path/to/doc.pdf")  # 上传文档
        answer = service.query("如何使用RAG系统？")  # 查询
    """

    def __init__(self):
        """初始化 RAG 服务"""
        # 核心组件（延迟初始化）
        self._parser: Optional[DocumentParser] = None
        self._chunker: Optional[Chunker] = None
        self._embedding_manager: Optional[EmbeddingManager] = None
        self._vector_store: Optional[VectorStoreManager] = None
        self._retriever: Optional[HybridRetriever] = None
        self._reranker: Optional[RerankerManager] = None
        self._llm_manager: Optional[LLMManager] = None
        self._citation_extractor: Optional[CitationExtractor] = None

        # 初始化状态
        self._initialized = False

    def init_components(self) -> None:
        """
        初始化所有核心组件

        按顺序初始化：文档解析器 -> 分块器 -> Embedding -> 向量存储 -> 检索器 -> Reranker -> LLM
        """
        if self._initialized:
            logger.info("RAG 服务已初始化，跳过重复初始化")
            return

        logger.info("开始初始化 RAG 服务组件...")

        try:
            # 1. 文档解析器
            self._parser = DocumentParser()
            logger.info("[1/7] 文档解析器初始化完成")

            # 2. 分块器
            self._chunker = Chunker(
                strategy=settings.chunk_strategy,
                chunk_size=settings.chunk_size,
                overlap=settings.chunk_overlap,
            )
            logger.info(f"[2/7] 分块器初始化完成 (策略: {settings.chunk_strategy})")

            # 3. 向量化
            self._embedding_manager = get_embedding_manager()
            logger.info(f"[3/7] Embedding 初始化完成 (维度: {self._embedding_manager.dimension})")

            # 4. 向量存储
            self._vector_store = VectorStoreManager()
            logger.info(f"[4/7] 向量存储初始化完成 (当前文档数: {self._vector_store.count()})")

            # 5. 混合检索器
            self._retriever = HybridRetriever(vector_store=self._vector_store)
            # 如果已有文档，构建 BM25 索引
            if self._vector_store.count() > 0:
                self._retriever.build_index()
            logger.info("[5/7] 混合检索器初始化完成")

            # 6. Reranker
            self._reranker = RerankerManager()
            logger.info(f"[6/7] Reranker 初始化完成 (启用: {self._reranker.is_enabled})")

            # 7. LLM
            self._llm_manager = get_llm_manager()
            logger.info(f"[7/7] LLM 初始化完成 (模型: {self._llm_manager.model_name})")

            # 8. 引用提取器
            self._citation_extractor = CitationExtractor()

            self._initialized = True
            logger.info("RAG 服务所有组件初始化完成!")

        except Exception as e:
            logger.error(f"RAG 服务初始化失败: {e}")
            raise

    def _ensure_initialized(self) -> None:
        """确保服务已初始化"""
        if not self._initialized:
            self.init_components()

    # ========== 文档管理 ==========

    def upload_document(self, file_path: str) -> Dict[str, Any]:
        """
        上传并处理文档

        完整流程：解析 -> 分块 -> 向量化 -> 存储 -> 重建索引

        Args:
            file_path: 文档文件路径

        Returns:
            Dict: 处理结果（文件名、块数、总字符数等）
        """
        self._ensure_initialized()
        start_time = time.time()

        logger.info(f"开始处理文档: {file_path}")

        # 1. 解析文档
        parsed_doc = self._parser.parse(file_path)

        # 2. 分块
        chunks = self._chunker.chunk_document(parsed_doc)

        if not chunks:
            logger.warning(f"文档分块结果为空: {file_path}")
            return {"filename": parsed_doc.filename, "chunk_count": 0, "status": "empty"}

        # 3. 向量化
        texts = [chunk.content for chunk in chunks]
        metadatas = [chunk.metadata for chunk in chunks]

        logger.info(f"开始向量化 {len(texts)} 个文本块...")
        embeddings = self._embedding_manager.embed_documents(texts)
        logger.info(f"向量化完成，耗时: {time.time() - start_time:.2f}s")

        # 4. 存储到向量数据库
        ids = self._vector_store.add_texts(
            texts=texts,
            metadatas=metadatas,
            embeddings=embeddings,
        )

        # 5. 重建 BM25 索引
        logger.info("重建 BM25 索引...")
        self._retriever.build_index()

        elapsed = time.time() - start_time
        result = {
            "filename": parsed_doc.filename,
            "file_type": parsed_doc.file_type,
            "title": parsed_doc.title,
            "chunk_count": len(chunks),
            "total_chars": parsed_doc.total_chars,
            "doc_count": self._vector_store.count(),
            "elapsed_seconds": round(elapsed, 2),
            "status": "success",
        }

        logger.info(f"文档处理完成: {result}")
        return result

    def delete_document(self, filename: str) -> Dict[str, Any]:
        """
        删除指定文件的所有文本块

        Args:
            filename: 文件名

        Returns:
            Dict: 删除结果
        """
        self._ensure_initialized()

        # 通过元数据过滤删除
        self._vector_store.delete(where={"source_file": filename})
        # 重建 BM25 索引
        self._retriever.build_index()

        logger.info(f"已删除文档: {filename}")
        return {"filename": filename, "status": "deleted", "doc_count": self._vector_store.count()}

    def list_documents(self) -> List[Dict[str, Any]]:
        """
        列出所有已上传的文档

        Returns:
            List[Dict]: 文档信息列表
        """
        self._ensure_initialized()
        all_docs = self._vector_store.get_all_documents()

        # 按 source_file 聚合
        doc_map = {}
        for doc in all_docs:
            source = doc["metadata"].get("source_file", "unknown")
            if source not in doc_map:
                doc_map[source] = {
                    "filename": source,
                    "title": doc["metadata"].get("doc_title", ""),
                    "file_type": doc["metadata"].get("doc_type", ""),
                    "chunk_count": 0,
                }
            doc_map[source]["chunk_count"] += 1

        return list(doc_map.values())

    def get_stats(self) -> Dict[str, Any]:
        """获取系统统计信息"""
        self._ensure_initialized()
        return {
            "total_documents": len(self.list_documents()),
            "total_chunks": self._vector_store.count(),
            "embedding_provider": self._embedding_manager.provider,
            "embedding_dimension": self._embedding_manager.dimension,
            "llm_provider": self._llm_manager.provider,
            "llm_model": self._llm_manager.model_name,
            "reranker_enabled": self._reranker.is_enabled,
            "chunk_strategy": settings.chunk_strategy,
            "chunk_size": settings.chunk_size,
        }

    # ========== 问答功能 ==========

    def query(
        self,
        question: str,
        top_k: int = None,
        stream: bool = False,
    ) -> Dict[str, Any]:
        """
        知识库问答（同步）

        流程：检索 -> 重排序 -> 构建上下文 -> LLM 生成 -> 引用溯源

        Args:
            question: 用户问题
            top_k: 检索结果数量
            stream: 是否流式输出

        Returns:
            Dict: 回答结果（answer, citations, sources 等）
        """
        self._ensure_initialized()
        start_time = time.time()

        logger.info(f"收到查询: {question[:50]}...")

        # 1. 检索
        retrieval_results = self._retriever.retrieve(question, top_k=top_k)

        # 2. 重排序
        if retrieval_results:
            retrieval_results = self._reranker.rerank(question, retrieval_results, top_k)

        # 3. 引用溯源
        citations = self._citation_extractor.extract_citations(retrieval_results)

        # 4. 构建上下文
        context = self._citation_extractor.build_context_with_sources(retrieval_results)
        citation_instruction = self._citation_extractor.build_citation_prompt()

        # 5. 构建 prompt
        user_prompt = RAG_USER_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
            citation_instruction=citation_instruction,
        )

        # 6. LLM 生成
        if stream:
            # 流式生成在 API 层处理
            return {
                "stream": True,
                "prompt": user_prompt,
                "citations": [c.to_dict() for c in citations],
            }

        answer = self._llm_manager.generate(
            prompt=user_prompt,
            system_prompt=RAG_SYSTEM_PROMPT,
        )

        # 7. 添加引用标注
        answer = self._citation_extractor.add_citation_marks(answer, citations)

        elapsed = time.time() - start_time
        result = {
            "question": question,
            "answer": answer,
            "citations": [c.to_dict() for c in citations],
            "retrieval_count": len(retrieval_results),
            "elapsed_seconds": round(elapsed, 2),
        }

        logger.info(f"查询完成，耗时: {elapsed:.2f}s")
        return result

    def query_stream(
        self,
        question: str,
        top_k: int = None,
    ) -> Generator[str, None, None]:
        """
        知识库问答（流式输出）

        流式返回 LLM 生成的文本片段

        Args:
            question: 用户问题
            top_k: 检索结果数量

        Yields:
            str: 文本片段（SSE 格式）
        """
        self._ensure_initialized()
        start_time = time.time()

        logger.info(f"收到流式查询: {question[:50]}...")

        # 1. 检索
        retrieval_results = self._retriever.retrieve(question, top_k=top_k)

        # 2. 重排序
        if retrieval_results:
            retrieval_results = self._reranker.rerank(question, retrieval_results, top_k)

        # 3. 引用溯源
        citations = self._citation_extractor.extract_citations(retrieval_results)

        # 4. 构建上下文
        context = self._citation_extractor.build_context_with_sources(retrieval_results)
        citation_instruction = self._citation_extractor.build_citation_prompt()

        # 5. 构建 prompt
        user_prompt = RAG_USER_PROMPT_TEMPLATE.format(
            context=context,
            question=question,
            citation_instruction=citation_instruction,
        )

        # 6. 先发送引用信息（作为第一个 SSE 事件）
        citations_data = json.dumps([c.to_dict() for c in citations], ensure_ascii=False)
        yield f"data: {json.dumps({'type': 'citations', 'citations': [c.to_dict() for c in citations]}, ensure_ascii=False)}\n\n"

        # 7. 流式生成
        full_answer = ""
        for token in self._llm_manager.stream_generate(
            prompt=user_prompt,
            system_prompt=RAG_SYSTEM_PROMPT,
        ):
            full_answer += token
            yield f"data: {json.dumps({'type': 'content', 'text': token}, ensure_ascii=False)}\n\n"

        # 8. 发送完成事件（包含引用标注后的完整回答）
        marked_answer = self._citation_extractor.add_citation_marks(full_answer, citations)
        elapsed = time.time() - start_time

        yield f"data: {json.dumps({'type': 'done', 'full_answer': marked_answer, 'elapsed_seconds': round(elapsed, 2)}, ensure_ascii=False)}\n\n"
        yield "data: [DONE]\n\n"

        logger.info(f"流式查询完成，耗时: {elapsed:.2f}s")

    def reset(self) -> None:
        """重置知识库（清空所有数据）"""
        self._ensure_initialized()
        self._vector_store.reset()
        self._retriever = HybridRetriever(vector_store=self._vector_store)
        logger.info("知识库已重置")