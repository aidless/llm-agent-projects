"""
Reranker 重排序模块
使用交叉编码器（Cross-Encoder）对检索结果进行二次精排
支持 BGE-Reranker 和本地 Transformer 模型
"""
from typing import List, Optional, Dict, Any
from loguru import logger

from core.config import settings
from core.retrieval.retriever import RetrievalResult


class RerankerManager:
    """
    Reranker 管理器

    使用交叉编码器对检索结果进行精排，显著提升检索质量

    使用示例:
        reranker = RerankerManager()
        reranked = reranker.rerank(query="查询文本", results=retrieval_results, top_k=3)
    """

    def __init__(
        self,
        model_name: str = None,
        top_k: int = None,
        enabled: bool = None,
    ):
        """
        初始化 Reranker

        Args:
            model_name: 重排序模型名称
            top_k: 重排后保留的结果数量
            enabled: 是否启用重排序
        """
        self._model_name = model_name or settings.reranker_model
        self._top_k = top_k or settings.reranker_top_k
        self._enabled = enabled if enabled is not None else settings.reranker_enabled
        self._model = None
        self._device = None

        if self._enabled:
            self._load_model()

    def _load_model(self) -> None:
        """加载重排序模型"""
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch

            # 检测设备
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Reranker 设备: {self._device}")

            # 加载模型
            self._tokenizer = AutoTokenizer.from_pretrained(self._model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self._model_name
            )
            self._model.to(self._device)
            self._model.eval()

            logger.info(f"Reranker 模型加载成功: {self._model_name}")

        except ImportError:
            logger.warning(
                "transformers/torch 未安装，Reranker 将使用基于规则的降级方案。"
                "请运行: pip install transformers torch"
            )
            self._enabled = False
            self._model = None
        except Exception as e:
            logger.warning(f"Reranker 模型加载失败，使用降级方案: {e}")
            self._enabled = False
            self._model = None

    def rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int = None,
    ) -> List[RetrievalResult]:
        """
        对检索结果进行重排序

        Args:
            query: 用户查询文本
            results: 初步检索结果列表
            top_k: 重排后保留的结果数量

        Returns:
            List[RetrievalResult]: 重排序后的结果列表
        """
        if not results:
            return []

        if not self._enabled or self._model is None:
            logger.debug("Reranker 未启用或模型未加载，返回原始结果")
            return results[:top_k or self._top_k]

        k = top_k or self._top_k
        logger.info(f"开始重排序: {len(results)} 条结果 -> top {k}")

        try:
            if self._model is not None:
                reranked = self._model_rerank(query, results, k)
            else:
                reranked = self._heuristic_rerank(query, results, k)

            logger.info(f"重排序完成: 保留 {len(reranked)} 条结果")
            return reranked

        except Exception as e:
            logger.error(f"重排序失败，返回原始结果: {e}")
            return results[:k]

    def _model_rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        """
        使用模型进行重排序

        构造 (query, document) 对，通过交叉编码器计算相关性分数
        """
        import torch

        # 构造输入对
        pairs = [(query, r.content) for r in results]

        # 批量编码
        batch_size = 8
        all_scores = []

        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i:i + batch_size]
            inputs = self._tokenizer(
                batch_pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                # 对于 BGE-Reranker，使用 sigmoid 转换 logits
                scores = torch.sigmoid(outputs.logits.squeeze(-1))
                all_scores.extend(scores.cpu().tolist())

        # 按重排分数排序
        scored_results = []
        for result, score in zip(results, all_scores):
            new_result = RetrievalResult(
                id=result.id,
                content=result.content,
                score=float(score),
                vector_score=result.vector_score,
                bm25_score=result.bm25_score,
                metadata={
                    **result.metadata,
                    "reranker_score": float(score),
                },
            )
            scored_results.append(new_result)

        scored_results.sort(key=lambda x: x.score, reverse=True)
        return scored_results[:top_k]

    def _heuristic_rerank(
        self,
        query: str,
        results: List[RetrievalResult],
        top_k: int,
    ) -> List[RetrievalResult]:
        """
        基于启发式规则的重排序（模型不可用时的降级方案）

        综合考虑：
        1. 关键词命中数
        2. 文本长度适宜性
        3. 原始检索分数
        """
        import jieba

        query_words = set(jieba.lcut_for_search(query))
        # 过滤停用词
        stop_words = {"的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都",
                      "一", "上", "也", "很", "到", "说", "要", "去", "你", "会", "着"}
        query_words = query_words - stop_words

        scored_results = []
        for result in results:
            content_words = set(jieba.lcut_for_search(result.content))
            # 关键词命中率
            keyword_hit = len(query_words & content_words) / max(len(query_words), 1)

            # 长度适宜性（偏好中等长度的文本）
            length = len(result.content)
            if length <= 0:
                length_score = 0
            elif length < 50:
                length_score = 0.3
            elif length < 200:
                length_score = 0.8
            elif length < 500:
                length_score = 1.0
            elif length < 1000:
                length_score = 0.7
            else:
                length_score = 0.4

            # 综合得分
            final_score = (
                0.4 * keyword_hit +
                0.2 * length_score +
                0.4 * min(result.score, 1.0) if result.score < 1.0 else 0.4
            )

            new_result = RetrievalResult(
                id=result.id,
                content=result.content,
                score=final_score,
                vector_score=result.vector_score,
                bm25_score=result.bm25_score,
                metadata={
                    **result.metadata,
                    "reranker_score": final_score,
                    "heuristic": True,
                },
            )
            scored_results.append(new_result)

        scored_results.sort(key=lambda x: x.score, reverse=True)
        return scored_results[:top_k]

    @property
    def is_enabled(self) -> bool:
        """是否启用重排序"""
        return self._enabled and self._model is not None
