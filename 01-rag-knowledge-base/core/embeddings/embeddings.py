"""
向量化模块
支持多种 Embedding 模型：BGE、OpenAI、本地 SentenceTransformer
提供统一的 Embedding 接口
"""
from abc import ABC, abstractmethod
from typing import List, Optional
import numpy as np
from loguru import logger

from core.config import settings


class BaseEmbedding(ABC):
    """Embedding 模型抽象基类"""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量文档向量化"""
        ...

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """单条查询向量化"""
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """向量维度"""
        ...


class BGEEmbedding(BaseEmbedding):
    """
    BGE (BAAI General Embedding) 模型

    使用 FlagEmbedding 或 SentenceTransformers 加载 BGE 系列模型
    适合中文场景，推荐 bge-large-zh-v1.5
    """

    def __init__(self, model_name: str = None, dimension: int = None):
        self._model_name = model_name or settings.bge_model_name
        self._dimension = dimension or settings.bge_embedding_dimension
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        """延迟加载模型"""
        try:
            # 优先尝试 FlagEmbedding
            from FlagEmbedding import FlagModel
            self._model = FlagModel(
                self._model_name,
                query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                use_fp16=True,
            )
            logger.info(f"BGE 模型加载成功: {self._model_name}")
        except ImportError:
            logger.info("FlagEmbedding 未安装，尝试使用 SentenceTransformers")
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._model_name)
                logger.info(f"SentenceTransformer 模型加载成功: {self._model_name}")
            except ImportError:
                raise RuntimeError(
                    "需要安装 FlagEmbedding 或 sentence-transformers。"
                    "请运行: pip install FlagEmbedding sentence-transformers"
                )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量文档向量化"""
        if not texts:
            return []
        try:
            if hasattr(self._model, "encode"):
                # SentenceTransformers 接口
                embeddings = self._model.encode(texts, show_progress_bar=False)
                return [emb.tolist() for emb in embeddings]
            elif hasattr(self._model, "encode_queries"):
                # FlagEmbedding 接口 - 文档不需要 instruction
                embeddings = self._model.encode(texts)
                return [emb.tolist() for emb in embeddings]
            else:
                raise AttributeError("模型没有可用的 encode 方法")
        except Exception as e:
            logger.error(f"BGE 文档向量化失败: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """单条查询向量化"""
        try:
            if hasattr(self._model, "encode_queries"):
                # FlagEmbedding 查询接口（会自动加 instruction）
                embedding = self._model.encode_queries([text])
                return embedding[0].tolist()
            elif hasattr(self._model, "encode"):
                embedding = self._model.encode([text], show_progress_bar=False)
                return embedding[0].tolist()
            else:
                raise AttributeError("模型没有可用的 encode 方法")
        except Exception as e:
            logger.error(f"BGE 查询向量化失败: {e}")
            raise

    @property
    def dimension(self) -> int:
        return self._dimension


class OpenAIEmbedding(BaseEmbedding):
    """
    OpenAI Embedding 模型

    调用 OpenAI API 进行向量化，支持自定义 API 地址
    """

    def __init__(
        self,
        api_key: str = None,
        api_base: str = None,
        model_name: str = None,
        dimension: int = 1536,
    ):
        self._api_key = api_key or settings.openai_api_key
        self._api_base = api_base or settings.openai_api_base
        self._model_name = model_name or settings.openai_embedding_model
        self._dimension = dimension
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        """初始化 OpenAI 客户端"""
        try:
            from openai import OpenAI
            self._client = OpenAI(
                api_key=self._api_key,
                base_url=self._api_base,
            )
            logger.info(f"OpenAI Embedding 客户端初始化成功: {self._model_name}")
        except ImportError:
            raise RuntimeError("需要安装 openai 库。请运行: pip install openai")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量文档向量化"""
        if not texts:
            return []

        all_embeddings = []
        # OpenAI API 有批量限制（约 2048 条），分批处理
        batch_size = 100
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self._client.embeddings.create(
                    input=batch,
                    model=self._model_name,
                )
                batch_embeddings = [item.embedding for item in response.data]
                all_embeddings.extend(batch_embeddings)
            except Exception as e:
                logger.error(f"OpenAI Embedding 批次 {i} 失败: {e}")
                raise

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """单条查询向量化"""
        try:
            response = self._client.embeddings.create(
                input=[text],
                model=self._model_name,
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"OpenAI 查询向量化失败: {e}")
            raise

    @property
    def dimension(self) -> int:
        return self._dimension


class LocalEmbedding(BaseEmbedding):
    """
    本地 SentenceTransformers 模型

    使用 sentence-transformers 库加载本地模型
    默认使用 all-MiniLM-L6-v2（英文），可配置为中文模型
    """

    def __init__(self, model_name: str = None, dimension: int = None):
        self._model_name = model_name or settings.local_embedding_model
        self._dimension = dimension or settings.local_embedding_dimension
        self._model = None
        self._load_model()

    def _load_model(self) -> None:
        """加载本地模型"""
        try:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self._model_name)
            # 从模型获取实际维度
            test_emb = self._model.encode(["test"])
            self._dimension = len(test_emb[0])
            logger.info(f"本地 Embedding 模型加载成功: {self._model_name}, 维度={self._dimension}")
        except ImportError:
            raise RuntimeError(
                "需要安装 sentence-transformers。请运行: pip install sentence-transformers"
            )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量文档向量化"""
        if not texts:
            return []
        try:
            embeddings = self._model.encode(texts, show_progress_bar=False, normalize_embeddings=True)
            return [emb.tolist() for emb in embeddings]
        except Exception as e:
            logger.error(f"本地模型文档向量化失败: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """单条查询向量化"""
        try:
            embedding = self._model.encode([text], show_progress_bar=False, normalize_embeddings=True)
            return embedding[0].tolist()
        except Exception as e:
            logger.error(f"本地模型查询向量化失败: {e}")
            raise

    @property
    def dimension(self) -> int:
        return self._dimension


class EmbeddingManager:
    """
    Embedding 管理器

    统一管理不同 Embedding 提供商，提供单例模式

    使用示例:
        manager = EmbeddingManager()
        embeddings = manager.embed_documents(["文本1", "文本2"])
        query_emb = manager.embed_query("查询文本")
    """

    PROVIDER_MAP = {
        "bge": BGEEmbedding,
        "openai": OpenAIEmbedding,
        "local": LocalEmbedding,
        "sentence-transformers": LocalEmbedding,
    }

    def __init__(self, provider: str = None):
        """
        初始化 Embedding 管理器

        Args:
            provider: Embedding 提供商 (bge/openai/local)
        """
        self._provider_name = provider or settings.embedding_provider
        self._embedding: Optional[BaseEmbedding] = None
        self._init_embedding()

    def _init_embedding(self) -> None:
        """根据配置初始化 Embedding 模型"""
        provider_class = self.PROVIDER_MAP.get(self._provider_name)
        if provider_class is None:
            raise ValueError(
                f"不支持的 Embedding 提供商: {self._provider_name}，"
                f"可选: {list(self.PROVIDER_MAP.keys())}"
            )

        try:
            self._embedding = provider_class()
            logger.info(f"Embedding 管理器初始化成功: 提供商={self._provider_name}, 维度={self._embedding.dimension}")
        except Exception as e:
            logger.error(f"Embedding 模型初始化失败: {e}")
            raise

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """批量文档向量化"""
        return self._embedding.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        """单条查询向量化"""
        return self._embedding.embed_query(text)

    @property
    def dimension(self) -> int:
        """获取当前模型的向量维度"""
        return self._embedding.dimension

    @property
    def provider(self) -> str:
        """获取当前提供商名称"""
        return self._provider_name


# 全局单例（延迟初始化）
_embedding_manager: Optional[EmbeddingManager] = None


def get_embedding_manager(force_reload: bool = False) -> EmbeddingManager:
    """
    获取全局 Embedding 管理器单例

    Args:
        force_reload: 是否强制重新加载

    Returns:
        EmbeddingManager: Embedding 管理器实例
    """
    global _embedding_manager
    if _embedding_manager is None or force_reload:
        _embedding_manager = EmbeddingManager()
    return _embedding_manager