"""
向量存储模块
使用 ChromaDB 作为本地向量数据库
提供文档的存储、查询、删除等操作
"""
import uuid
from typing import List, Optional, Dict, Any
from loguru import logger
from chromadb import Client, PersistentClient
from chromadb.config import Settings as ChromaSettings

from core.config import settings


class VectorStoreManager:
    """
    向量存储管理器

    基于 ChromaDB 实现向量存储，支持持久化到本地磁盘

    使用示例:
        store = VectorStoreManager()
        store.add_texts(texts=["文本1", "文本2"], metadatas=[{"source": "a"}, {"source": "b"}])
        results = store.search(query="查询", top_k=5)
    """

    DEFAULT_COLLECTION = "rag_knowledge_base"

    def __init__(
        self,
        persist_directory: str = None,
        collection_name: str = None,
        embedding_fn=None,
    ):
        """
        初始化向量存储

        Args:
            persist_directory: 持久化目录
            collection_name: 集合名称
            embedding_fn: ChromaDB 兼容的 embedding 函数
        """
        self._persist_dir = persist_directory or settings.chroma_persist_dir
        self._collection_name = collection_name or self.DEFAULT_COLLECTION
        self._embedding_fn = embedding_fn
        self._client: Optional[Client] = None
        self._collection = None

        self._init_client()
        self._init_collection()

    def _init_client(self) -> None:
        """初始化 ChromaDB 客户端"""
        try:
            # 使用持久化客户端
            self._client = PersistentClient(
                path=self._persist_dir,
                settings=ChromaSettings(
                    anonymized_telemetry=False,
                    allow_reset=True,
                ),
            )
            logger.info(f"ChromaDB 客户端初始化成功，持久化目录: {self._persist_dir}")
        except Exception as e:
            logger.error(f"ChromaDB 客户端初始化失败: {e}")
            raise

    def _init_collection(self) -> None:
        """初始化或获取集合"""
        try:
            self._collection = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"description": "RAG 知识库向量存储"},
            )
            count = self._collection.count()
            logger.info(f"ChromaDB 集合 [{self._collection_name}] 就绪，当前文档数: {count}")
        except Exception as e:
            logger.error(f"ChromaDB 集合初始化失败: {e}")
            raise

    def _get_embedding_fn(self):
        """
        获取 embedding 函数

        如果初始化时未传入，则创建一个适配 ChromaDB 接口的包装类
        """
        if self._embedding_fn is not None:
            return self._embedding_fn

        # 创建一个适配器，将 EmbeddingManager 适配为 ChromaDB 需要的接口
        class ChromaEmbeddingAdapter:
            """适配 EmbeddingManager 到 ChromaDB 的 embedding function 接口"""

            def __init__(self, manager=None):
                self._manager = manager

            def __call__(self, input: List[str]) -> List[List[float]]:
                if self._manager is None:
                    from core.embeddings import get_embedding_manager
                    self._manager = get_embedding_manager()
                return self._manager.embed_documents(input)

        return ChromaEmbeddingAdapter()

    def add_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None,
        embeddings: Optional[List[List[float]]] = None,
    ) -> List[str]:
        """
        添加文本到向量存储

        Args:
            texts: 文本列表
            metadatas: 元数据列表，与 texts 一一对应
            ids: 可选的文档 ID 列表
            embeddings: 可选的预计算 embedding 列表

        Returns:
            List[str]: 生成的文档 ID 列表
        """
        if not texts:
            logger.warning("添加的文本列表为空")
            return []

        # 生成文档 ID
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in texts]

        # 确保元数据列表长度匹配
        if metadatas is None:
            metadatas = [{}] * len(texts)
        elif len(metadatas) != len(texts):
            raise ValueError(f"元数据数量 ({len(metadatas)}) 与文本数量 ({len(texts)}) 不匹配")

        try:
            # ChromaDB 的 add 方法
            add_kwargs = {
                "ids": ids,
                "documents": texts,
                "metadatas": metadatas,
            }

            # 如果提供了预计算的 embedding，直接使用
            if embeddings is not None:
                add_kwargs["embeddings"] = embeddings
            else:
                # 使用 embedding function
                add_kwargs["embedding_function"] = None  # 禁用默认的
                # 手动计算 embedding
                emb_fn = self._get_embedding_fn()
                computed_embeddings = emb_fn(texts)
                add_kwargs["embeddings"] = computed_embeddings

            self._collection.add(**add_kwargs)
            logger.info(f"成功添加 {len(texts)} 条文档到向量存储")

            return ids

        except Exception as e:
            logger.error(f"添加文档到向量存储失败: {e}")
            raise

    def search(
        self,
        query: str,
        top_k: int = None,
        query_embedding: Optional[List[float]] = None,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None,
    ) -> List[Dict[str, Any]]:
        """
        向量相似度检索

        Args:
            query: 查询文本
            top_k: 返回结果数量
            query_embedding: 可选的预计算查询向量
            where: 元数据过滤条件
            where_document: 文档内容过滤条件

        Returns:
            List[Dict]: 检索结果列表，每个元素包含 id, document, metadata, distance
        """
        if top_k is None:
            top_k = settings.retrieval_top_k

        try:
            search_kwargs = {
                "n_results": top_k,
                "include": ["documents", "metadatas", "distances"],
            }

            # 过滤条件
            if where:
                search_kwargs["where"] = where
            if where_document:
                search_kwargs["where_document"] = where_document

            if query_embedding is not None:
                # 使用预计算的查询向量
                search_kwargs["query_embeddings"] = [query_embedding]
            else:
                # 使用查询文本
                search_kwargs["query_embeddings"] = None
                # 手动计算查询向量
                emb_fn = self._get_embedding_fn()
                query_emb = emb_fn([query])
                search_kwargs["query_embeddings"] = query_emb

            results = self._collection.query(**search_kwargs)

            # 将 ChromaDB 结果格式化为统一结构
            formatted_results = []
            if results and results["ids"]:
                for i in range(len(results["ids"][0])):
                    result = {
                        "id": results["ids"][0][i],
                        "document": results["documents"][0][i] if results["documents"] else "",
                        "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                        "distance": results["distances"][0][i] if results["distances"] else 0.0,
                        "score": 1.0 - results["distances"][0][i] if results["distances"] else 0.0,
                    }
                    formatted_results.append(result)

            logger.debug(f"向量检索完成: query='{query[:30]}...', 返回 {len(formatted_results)} 条结果")
            return formatted_results

        except Exception as e:
            logger.error(f"向量检索失败: {e}")
            raise

    def delete(self, ids: Optional[List[str]] = None, where: Optional[Dict] = None) -> None:
        """
        删除文档

        Args:
            ids: 要删除的文档 ID 列表
            where: 元数据过滤条件
        """
        try:
            if ids:
                self._collection.delete(ids=ids)
                logger.info(f"已删除 {len(ids)} 条文档")
            elif where:
                self._collection.delete(where=where)
                logger.info(f"已按条件删除文档: {where}")
            else:
                logger.warning("删除操作需要指定 ids 或 where 条件")
        except Exception as e:
            logger.error(f"删除文档失败: {e}")
            raise

    def get(self, ids: Optional[List[str]] = None, where: Optional[Dict] = None) -> List[Dict]:
        """
        获取文档

        Args:
            ids: 文档 ID 列表
            where: 元数据过滤条件

        Returns:
            List[Dict]: 文档列表
        """
        try:
            get_kwargs = {
                "include": ["documents", "metadatas"],
            }
            if ids:
                get_kwargs["ids"] = ids
            if where:
                get_kwargs["where"] = where

            results = self._collection.get(**get_kwargs)

            formatted = []
            if results and results["ids"]:
                for i in range(len(results["ids"])):
                    formatted.append({
                        "id": results["ids"][i],
                        "document": results["documents"][i] if results["documents"] else "",
                        "metadata": results["metadatas"][i] if results["metadatas"] else {},
                    })
            return formatted
        except Exception as e:
            logger.error(f"获取文档失败: {e}")
            raise

    def count(self) -> int:
        """获取集合中的文档总数"""
        try:
            return self._collection.count()
        except Exception as e:
            logger.error(f"获取文档数量失败: {e}")
            return 0

    def list_collections(self) -> List[str]:
        """列出所有集合名称"""
        try:
            collections = self._client.list_collections()
            return [c.name for c in collections]
        except Exception as e:
            logger.error(f"列出集合失败: {e}")
            return []

    def reset(self) -> None:
        """重置（清空）当前集合"""
        try:
            self._client.delete_collection(self._collection_name)
            self._init_collection()
            logger.info(f"集合 [{self._collection_name}] 已重置")
        except Exception as e:
            logger.error(f"重置集合失败: {e}")
            raise

    def get_all_documents(self) -> List[Dict]:
        """获取集合中的所有文档（用于 BM25 索引构建）"""
        try:
            results = self._collection.get(
                include=["documents", "metadatas"],
            )
            formatted = []
            if results and results["ids"]:
                for i in range(len(results["ids"])):
                    formatted.append({
                        "id": results["ids"][i],
                        "document": results["documents"][i] if results["documents"] else "",
                        "metadata": results["metadatas"][i] if results["metadatas"] else {},
                    })
            return formatted
        except Exception as e:
            logger.error(f"获取所有文档失败: {e}")
            return []