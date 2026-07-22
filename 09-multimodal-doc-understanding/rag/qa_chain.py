"""问答链 - 带来源引用的文档问答。"""

import logging
from typing import Dict, List, Optional

from app.models import QAResult
from rag.document_indexer import DocumentIndexer
from rag.retriever import HybridRetriever

logger = logging.getLogger(__name__)


class MockLLMClient:
    """Mock LLM 客户端 - 用于生成回答。"""

    def generate(self, prompt: str) -> str:
        """生成回答 (mock)。

        Args:
            prompt: 提示文本。

        Returns:
            str: 生成的回答。
        """
        return f"基于文档内容的回答 (mock): {prompt[:100]}..."


class QAChain:
    """文档问答链。

    流程:
    1. 检索相关文档分块
    2. 构建提示
    3. 调用 LLM 生成回答
    4. 返回带来源引用的结果
    """

    def __init__(
        self,
        indexer: DocumentIndexer,
        retriever: Optional[HybridRetriever] = None,
        llm_client=None,
    ):
        """初始化问答链。

        Args:
            indexer: 文档索引器。
            retriever: 检索器 (可选，自动从 indexer 创建)。
            llm_client: LLM 客户端 (可选，默认 MockLLMClient)。
        """
        self.indexer = indexer
        self.retriever = retriever or HybridRetriever(indexer.store)
        self.llm = llm_client or MockLLMClient()

    def ask(
        self,
        question: str,
        top_k: int = 3,
        document_ids: Optional[List[str]] = None,
    ) -> QAResult:
        """对文档进行问答。

        Args:
            question: 问题。
            top_k: 检索 top-k 个分块。
            document_ids: 限定文档范围。

        Returns:
            QAResult: 问答结果。
        """
        # 检索
        retrieved = self.retriever.retrieve(
            question, top_k=top_k, document_ids=document_ids
        )

        if not retrieved:
            return QAResult(
                question=question,
                answer="未找到相关文档内容，无法回答该问题。",
                sources=[],
                score=0.0,
            )

        # 构建上下文
        contexts = []
        sources = []
        max_score = 0.0

        for chunk, score in retrieved:
            contexts.append(f"[来源: {chunk.metadata.get('filename', 'unknown')}] {chunk.text}")
            sources.append(f"{chunk.metadata.get('filename', 'unknown')} (chunk {chunk.metadata.get('chunk_num', '?')})")
            max_score = max(max_score, score)

        context_text = "\n\n---\n\n".join(contexts)

        # 构建提示
        prompt = f"""请根据以下文档内容回答问题。如果文档中没有相关信息，请说明。

## 文档内容:
{context_text}

## 问题:
{question}

## 回答:"""

        # 生成回答
        answer = self.llm.generate(prompt)

        return QAResult(
            question=question,
            answer=answer,
            sources=sources,
            score=max_score,
        )

    def compare_documents(
        self,
        question: str,
        document_ids: List[str],
    ) -> Dict:
        """多文档对比问答。

        Args:
            question: 问题。
            document_ids: 文档 ID 列表。

        Returns:
            dict: 对比结果。
        """
        per_doc_results = {}
        for doc_id in document_ids:
            result = self.ask(question, top_k=3, document_ids=[doc_id])
            per_doc_results[doc_id] = {
                "answer": result.answer,
                "score": result.score,
                "sources": result.sources,
            }

        return {
            "question": question,
            "documents": per_doc_results,
            "summary": f"对比了 {len(document_ids)} 个文档 (mock 摘要)",
        }