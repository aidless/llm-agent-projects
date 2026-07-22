# -*- coding: utf-8 -*-
"""
向量数据库管理平台 - Embedding 模型管理器
支持 BGE/SentenceTransformers 本地模型和 OpenAI 远程模型
提供模型加载、切换、缓存和嵌入向量生成功能
"""
import hashlib
import logging
import time
from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod

import numpy as np
from sentence_transformers import SentenceTransformer

from app.config import settings

logger = logging.getLogger(__name__)


class BaseEmbeddingProvider(ABC):
    """Embedding 提供者抽象基类"""

    @property
    @abstractmethod
    def name(self) -> str:
        """返回模型名称"""
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        """返回向量维度"""
        pass

    @property
    @abstractmethod
    def provider_type(self) -> str:
        """返回提供者类型：local / openai"""
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """对文本列表进行嵌入"""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """对查询文本进行嵌入"""
        pass


class LocalEmbeddingProvider(BaseEmbeddingProvider):
    """本地 SentenceTransformers 模型提供者（支持 BGE 等模型）"""

    def __init__(self, model_name: str, cache_dir: str):
        """
        初始化本地模型

        Args:
            model_name: HuggingFace 模型名称或本地路径
            cache_dir: 模型缓存目录
        """
        self._model_name = model_name
        self._cache_dir = cache_dir
        logger.info(f"正在加载本地模型: {model_name}")
        self._model = SentenceTransformer(
            model_name,
            cache_folder=cache_dir
        )
        # 获取向量维度
        self._dimension = self._model.get_sentence_embedding_dimension()
        logger.info(f"模型加载完成: {model_name}, 维度: {self._dimension}")

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def provider_type(self) -> str:
        return "local"

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        对文本列表进行批量嵌入

        Args:
            texts: 待嵌入的文本列表

        Returns:
            嵌入向量列表
        """
        if not texts:
            return []
        start_time = time.time()
        embeddings = self._model.encode(texts, show_progress_bar=False)
        elapsed = time.time() - start_time
        logger.debug(f"嵌入 {len(texts)} 条文本耗时: {elapsed:.2f}s")
        return [emb.tolist() for emb in embeddings]

    def embed_query(self, text: str) -> List[float]:
        """
        对查询文本进行嵌入

        Args:
            text: 查询文本

        Returns:
            嵌入向量
        """
        embedding = self._model.encode([text], show_progress_bar=False)
        return embedding[0].tolist()


class OpenAIEmbeddingProvider(BaseEmbeddingProvider):
    """
    OpenAI 远程模型提供者
    通过 OpenAI API 生成嵌入向量
    """

    # OpenAI 模型维度映射
    MODEL_DIMENSIONS = {
        "text-embedding-ada-002": 1536,
        "text-embedding-3-small": 1536,
        "text-embedding-3-large": 3072,
    }

    def __init__(self, model_name: str, api_key: str, api_base: str):
        """
        初始化 OpenAI 模型

        Args:
            model_name: OpenAI 模型名称
            api_key: API 密钥
            api_base: API 基础地址
        """
        self._model_name = model_name
        self._api_key = api_key
        self._api_base = api_base
        self._dimension = self.MODEL_DIMENSIONS.get(model_name, 1536)
        logger.info(f"初始化 OpenAI 模型: {model_name}, 维度: {self._dimension}")

    @property
    def name(self) -> str:
        return self._model_name

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def provider_type(self) -> str:
        return "openai"

    def _call_openai_api(self, texts: List[str]) -> List[List[float]]:
        """
        调用 OpenAI Embedding API

        Args:
            texts: 待嵌入的文本列表

        Returns:
            嵌入向量列表

        Raises:
            RuntimeError: API 调用失败时抛出
        """
        try:
            import httpx
        except ImportError:
            raise RuntimeError("使用 OpenAI 模型需要安装 httpx: pip install httpx")

        url = f"{self._api_base}/embeddings"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

        # 分批请求（OpenAI 单次最多支持 2048 条）
        batch_size = 128
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            payload = {
                "input": batch,
                "model": self._model_name,
            }
            with httpx.Client(timeout=60.0) as client:
                response = client.post(url, json=payload, headers=headers)
                if response.status_code != 200:
                    error_msg = f"OpenAI API 调用失败: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    raise RuntimeError(error_msg)
                data = response.json()
                # 按原始顺序排列结果
                sorted_data = sorted(data["data"], key=lambda x: x["index"])
                batch_embeddings = [item["embedding"] for item in sorted_data]
                all_embeddings.extend(batch_embeddings)

        return all_embeddings

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        对文本列表进行批量嵌入（通过 OpenAI API）

        Args:
            texts: 待嵌入的文本列表

        Returns:
            嵌入向量列表
        """
        if not texts:
            return []
        start_time = time.time()
        embeddings = self._call_openai_api(texts)
        elapsed = time.time() - start_time
        logger.debug(f"OpenAI 嵌入 {len(texts)} 条文本耗时: {elapsed:.2f}s")
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        对查询文本进行嵌入

        Args:
            text: 查询文本

        Returns:
            嵌入向量
        """
        result = self.embed_texts([text])
        return result[0]


class EmbeddingManager:
    """
    Embedding 模型管理器
    管理多个嵌入模型的加载、切换和缓存
    """

    def __init__(self):
        """初始化模型管理器"""
        self._providers: Dict[str, BaseEmbeddingProvider] = {}
        self._current_provider: Optional[BaseEmbeddingProvider] = None
        self._current_model_name: Optional[str] = None

    def load_model(
        self,
        model_name: str,
        provider_type: str = "local",
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> BaseEmbeddingProvider:
        """
        加载或切换嵌入模型

        Args:
            model_name: 模型名称
            provider_type: 提供者类型 (local / openai)
            api_key: OpenAI API 密钥（仅 OpenAI 模型需要）
            api_base: OpenAI API 基础地址（仅 OpenAI 模型需要）

        Returns:
            加载的模型提供者实例
        """
        cache_key = f"{provider_type}:{model_name}"

        # 如果已经加载过该模型，直接返回
        if cache_key in self._providers:
            self._current_provider = self._providers[cache_key]
            self._current_model_name = cache_key
            logger.info(f"切换到已加载的模型: {model_name}")
            return self._current_provider

        # 根据提供者类型加载模型
        if provider_type == "openai":
            api_key = api_key or settings.openai_api_key
            api_base = api_base or settings.openai_api_base
            provider = OpenAIEmbeddingProvider(model_name, api_key, api_base)
        else:
            cache_dir = str(settings.get_model_cache_path())
            provider = LocalEmbeddingProvider(model_name, cache_dir)

        # 缓存并设为当前模型
        self._providers[cache_key] = provider
        self._current_provider = provider
        self._current_model_name = cache_key
        logger.info(f"模型加载并切换成功: {model_name}")
        return provider

    def get_current_provider(self) -> BaseEmbeddingProvider:
        """
        获取当前激活的模型提供者

        Returns:
            当前模型提供者实例

        Raises:
            RuntimeError: 未加载任何模型时抛出
        """
        if self._current_provider is None:
            raise RuntimeError("未加载任何 Embedding 模型，请先调用 load_model()")
        return self._current_provider

    def get_current_model_name(self) -> str:
        """获取当前模型名称"""
        return self._current_model_name or ""

    def list_loaded_models(self) -> List[Dict[str, Any]]:
        """
        列出所有已加载的模型信息

        Returns:
            模型信息列表
        """
        result = []
        for key, provider in self._providers.items():
            result.append({
                "key": key,
                "name": provider.name,
                "dimension": provider.dimension,
                "provider_type": provider.provider_type,
                "is_current": key == self._current_model_name,
            })
        return result

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        使用当前模型对文本列表进行嵌入

        Args:
            texts: 待嵌入的文本列表

        Returns:
            嵌入向量列表
        """
        provider = self.get_current_provider()
        return provider.embed_texts(texts)

    def embed_query(self, text: str) -> List[float]:
        """
        使用当前模型对查询文本进行嵌入

        Args:
            text: 查询文本

        Returns:
            嵌入向量
        """
        provider = self.get_current_provider()
        return provider.embed_query(text)


# 全局模型管理器单例
embedding_manager = EmbeddingManager()
