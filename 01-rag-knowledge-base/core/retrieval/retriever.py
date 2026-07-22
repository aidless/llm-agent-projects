"""
混合检索模块
实现 BM25 + 向量检索的混合检索策略，通过加权融合排序结果
"""
import math
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from rank_bm25 import BM25Okapi
import jieba
from loguru import logger
import numpy as np

from core.config import settings
from core.retrieval.vector_store import VectorStoreManager


@dataclass
class RetrievalResult:
    """单条检索结果"""

    id: str  # 文档 ID
    content: str  # 文本内容
    score: float  # 综合得分（归一化后）
    vector_score: float = 0.0  # 向量检索得分
    bm25_score: float = 0.0  # BM25 检索得分
    metadata: Dict[str, Any] = None  # 元数据

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "id": self.id,
            "content": self.content,
            "score": self.score,
            "vector_score": self.vector_score,
            "bm25_score": self.bm25_score,
            "metadata": self.metadata,
        }


class BM25Index:
    """
    BM25 索引管理器

    使用 jieba 分词 + rank_bm25 实现中文 BM25 检索

    使用示例:
        index = BM25Index()
        index.build(documents)
        results = index.search("查询文本", top_k=5)
    """

    def __init__(self, k1: float = None, b: float = None):
        """
        初始化 BM25 索引

        Args:
            k1: BM25 参数 k1（词频饱和度）
            b: BM25 参数 b（文档长度归一化）
        """
        self.k1 = k1 or settings.bm25_k1
        self.b = b or settings.bm25_b
        self._bm25: Optional[BM25Okapi] = None
        self._doc_ids: List[str] = []
        self._doc_contents: List[str] = []
        self._doc_metadatas: List[Dict] = []
        self._tokenized_docs: List[List[str]] = []

    def _tokenize(self, text: str) -> List[str]:
        """
        使用 jieba 对文本进行分词

        Args:
            text: 待分词文本

        Returns:
            List[str]: 分词结果列表
        """
        # 使用 jieba 精确模式分词
        words = jieba.lcut_for_search(text)
        # 过滤停用词和单字符（中文单字通常无意义）
        stop_words = {"的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都",
                      "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你",
                      "会", "着", "没有", "看", "好", "自己", "这"}
        return [w for w in words if w.strip() and w not in stop_words and len(w) > 0]

    def build(self, documents: List[Dict[str, Any]]) -> None:
        """
        构建 BM25 索引

        Args:
            documents: 文档列表，每个元素需包含 id, document, metadata
        """
        if not documents:
            logger.warning("文档列表为空，跳过 BM25 索引构建")
            return

        self._doc_ids = [doc["id"] for doc in documents]
        self._doc_contents = [doc.get("document", "") for doc in documents]
        self._doc_metadatas = [doc.get("metadata", {}) for doc in documents]

        # 对所有文档进行分词
        logger.info(f"开始构建 BM25 索引，文档数: {len(self._doc_contents)}")
        self._tokenized_docs = [self._tokenize(doc) for doc in self._doc_contents]

        # 创建 BM25 模型
        self._bm25 = BM25Okapi(self._tokenized_docs, k1=self.k1, b=self.b)
        logger.info(f"BM25 索引构建完成，文档数: {len(self._tokenized_docs)}")

    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        BM25 检索

        Args:
            query: 查询文本
            top_k: 返回结果数量

        Returns:
            List[Dict]: 检索结果列表，包含 id, document, metadata, bm25_score
        """
        if self._bm25 is None:
            logger.warning("BM25 索引未构建")
            return []

        if top_k is None:
            top_k = settings.retrieval_top_k

        # 对查询进行分词
        tokenized_query = self._tokenize(query)
        if not tokenized_query:
            logger.warning(f"查询分词结果为空: {query}")
            return []

        # 获取 BM25 分数
        scores = self._bm25.get_scores(tokenized_query)

        # 获取 top_k 个结果的索引
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0:  # 只返回分数大于 0 的结果
                results.append({
                    "id": self._doc_ids[idx],
                    "document": self._doc_contents[idx],
                    "metadata": self._doc_metadatas[idx],
                    "bm25_score": score,
                })

        logger.debug(f"BM25 检索完成: query='{query[:30]}...', 返回 {len(results)} 条结果")
        return results

    @property
    def is_built(self) -> bool:
        """索引是否已构建"""
        return self._bm25 is not None

    @property
    def doc_count(self) -> int:
        """已索引文档数量"""
        return len(self._doc_ids)


class HybridRetriever:
    """
    混合检索器

    结合向量检索（语义相似度）和 BM25 检索（关键词匹配），
    通过加权分数融合（Reciprocal Rank Fusion）实现更全面的检索

    使用示例:
        retriever = HybridRetriever()
        retriever.build_index()
        results = retriever.retrieve("查询文本", top_k=5)
    """

    def __init__(
        self,
        vector_store: Optional[VectorStoreManager] = None,
        vector_weight: float = None,
        bm25_weight: float = None,
    ):
        """
        初始化混合检索器

        Args:
            vector_store: 向量存储管理器
            vector_weight: 向量检索权重
            bm25_weight: BM25 检索权重
        """
        self._vector_store = vector_store or VectorStoreManager()
        self._bm25_index = BM25Index()
        self._vector_weight = vector_weight or settings.hybrid_search_weight_vector
        self._bm25_weight = bm25_weight or settings.hybrid_search_weight_bm25
        self._index_built = False

    def build_index(self) -> None:
        """
        构建 BM25 索引

        从向量存储中获取所有文档，构建 BM25 倒排索引
        """
        logger.info("开始构建混合检索索引...")
        all_docs = self._vector_store.get_all_documents()
        self._bm25_index.build(all_docs)
        self._index_built = True
        logger.info(f"混合检索索引构建完成，文档数: {self._bm25_index.doc_count}")

    def retrieve(
        self,
        query: str,
        top_k: int = None,
        use_vector: bool = True,
        use_bm25: bool = True,
    ) -> List[RetrievalResult]:
        """
        执行混合检索

        流程：
        1. 并行执行向量检索和 BM25 检索
        2. 对两路结果进行归一化
        3. 使用加权 RRF (Reciprocal Rank Fusion) 融合排序
        4. 返回 top_k 结果

        Args:
            query: 查询文本
            top_k: 返回结果数量
            use_vector: 是否使用向量检索
            use_bm25: 是否使用 BM25 检索

        Returns:
            List[RetrievalResult]: 检索结果列表
        """
        if top_k is None:
            top_k = settings.retrieval_top_k

        logger.info(f"混合检索: query='{query[:50]}...', top_k={top_k}")

        # 1. 向量检索
        vector_results = {}
        if use_vector:
            try:
                raw_results = self._vector_store.search(query=query, top_k=top_k * 2)
                for r in raw_results:
                    vector_results[r["id"]] = r
            except Exception as e:
                logger.warning(f"向量检索失败，仅使用 BM25: {e}")

        # 2. BM25 检索
        bm25_results = {}
        if use_bm25 and self._bm25_index.is_built:
            try:
                raw_results = self._bm25_index.search(query, top_k=top_k * 2)
                for r in raw_results:
                    bm25_results[r["id"]] = r
            except Exception as e:
                logger.warning(f"BM25 检索失败，仅使用向量检索: {e}")

        # 3. 合并所有候选文档 ID
        all_ids = set(vector_results.keys()) | set(bm25_results.keys())
        if not all_ids:
            logger.warning("混合检索无结果")
            return []

        # 4. 归一化分数并加权融合
        # 使用 RRF (Reciprocal Rank Fusion) 算法
        rrf_results = self._reciprocal_rank_fusion(
            vector_results=vector_results,
            bm25_results=bm25_results,
            all_ids=all_ids,
            vector_weight=self._vector_weight,
            bm25_weight=self._bm25_weight,
        )

        # 5. 排序并返回 top_k
        rrf_results.sort(key=lambda x: x.score, reverse=True)
        final_results = rrf_results[:top_k]

        logger.info(f"混合检索完成: 共 {len(final_results)} 条结果")
        return final_results

    def _reciprocal_rank_fusion(
        self,
        vector_results: Dict[str, dict],
        bm25_results: Dict[str, dict],
        all_ids: set,
        vector_weight: float,
        bm25_weight: float,
        k: int = 60,
    ) -> List[RetrievalResult]:
        """
        RRF (Reciprocal Rank Fusion) 算法

        RRF 公式: score(d) = sum(w_i / (k + rank_i(d)))

        Args:
            vector_results: 向量检索结果 {id: {score, document, metadata, ...}}
            bm25_results: BM25 检索结果 {id: {bm25_score, document, metadata, ...}}
            all_ids: 所有候选文档 ID
            vector_weight: 向量检索权重
            bm25_weight: BM25 检索权重
            k: RRF 常数（默认 60）

        Returns:
            List[RetrievalResult]: 融合后的结果列表
        """
        results = []

        # 按分数排序，构建排名映射
        sorted_vector = sorted(
            vector_results.items(),
            key=lambda x: x[1].get("score", 0),
            reverse=True,
        )
        sorted_bm25 = sorted(
            bm25_results.items(),
            key=lambda x: x[1].get("bm25_score", 0),
            reverse=True,
        )

        # 构建 rank 映射
        vector_rank = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(sorted_vector)}
        bm25_rank = {doc_id: rank + 1 for rank, (doc_id, _) in enumerate(sorted_bm25)}

        for doc_id in all_ids:
            # 计算 RRF 分数
            rrf_score = 0.0
            v_score = 0.0
            b_score = 0.0

            # 向量检索贡献
            if doc_id in vector_rank:
                rank = vector_rank[doc_id]
                rrf_score += vector_weight / (k + rank)
                v_score = vector_results[doc_id].get("score", 0.0)

            # BM25 检索贡献
            if doc_id in bm25_rank:
                rank = bm25_rank[doc_id]
                rrf_score += bm25_weight / (k + rank)
                b_score = bm25_results[doc_id].get("bm25_score", 0.0)

            # 获取文档内容（优先从向量结果获取）
            doc_content = ""
            metadata = {}
            if doc_id in vector_results:
                doc_content = vector_results[doc_id].get("document", "")
                metadata = vector_results[doc_id].get("metadata", {})
            elif doc_id in bm25_results:
                doc_content = bm25_results[doc_id].get("document", "")
                metadata = bm25_results[doc_id].get("metadata", {})

            results.append(
                RetrievalResult(
                    id=doc_id,
                    content=doc_content,
                    score=rrf_score,
                    vector_score=v_score,
                    bm25_score=b_score,
                    metadata=metadata,
                )
            )

        return results

    def retrieve_vector_only(self, query: str, top_k: int = None) -> List[RetrievalResult]:
        """仅使用向量检索"""
        results = self.retrieve(query, top_k, use_vector=True, use_bm25=False)
        return results

    def retrieve_bm25_only(self, query: str, top_k: int = None) -> List[RetrievalResult]:
        """仅使用 BM25 检索"""
        results = self.retrieve(query, top_k, use_vector=False, use_bm25=True)
        return results

    @property
    def index_built(self) -> bool:
        """BM25 索引是否已构建"""
        return self._index_built