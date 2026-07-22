# -*- coding: utf-8 -*-
"""
向量数据库管理平台 - 向量存储服务
封装 ChromaDB 操作，提供 Collection 管理、文档存储和相似度搜索功能
"""
import logging
import shutil
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from app.config import settings
from services.file_parser import FileParser, ParsedDocument
from services.chunker import DocumentChunker, ChunkConfig, TextChunk
from embeddings.manager import embedding_manager

logger = logging.getLogger(__name__)


class VectorStoreService:
    """
    向量存储服务
    封装 ChromaDB 的所有操作，与 Embedding 模型和分块器协同工作
    """

    def __init__(self):
        """初始化向量存储服务"""
        persist_dir = str(settings.get_chroma_persist_path())
        # 使用持久化客户端
        self._client = chromadb.PersistentClient(path=persist_dir)
        logger.info(f"ChromaDB 初始化完成，持久化目录: {persist_dir}")

    # ========== Collection 管理 ==========

    def list_collections(self) -> List[Dict[str, Any]]:
        """
        列出所有 Collection

        Returns:
            Collection 信息列表
        """
        collections = self._client.list_collections()
        result = []
        for col in collections:
            try:
                count = col.count()
            except Exception:
                count = 0
            result.append({
                "name": col.name,
                "id": col.id,
                "count": count,
            })
        return result

    def create_collection(
        self,
        name: str,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        创建新的 Collection

        Args:
            name: Collection 名称
            description: 描述信息
            metadata: 额外的元数据

        Returns:
            创建结果
        """
        meta = metadata or {}
        meta["description"] = description
        # 记录当前使用的模型和分块策略
        meta["embedding_model"] = embedding_manager.get_current_model_name()
        meta.setdefault("chunk_strategy", "fixed")
        meta.setdefault("chunk_size", settings.default_chunk_size)
        meta.setdefault("chunk_overlap", settings.default_chunk_overlap)

        collection = self._client.get_or_create_collection(
            name=name,
            metadata=meta,
        )
        return {
            "name": collection.name,
            "id": collection.id,
            "count": collection.count(),
            "metadata": collection.metadata,
        }

    def delete_collection(self, name: str) -> Dict[str, Any]:
        """
        删除指定 Collection

        Args:
            name: Collection 名称

        Returns:
            删除结果
        """
        self._client.delete_collection(name=name)
        return {"message": f"Collection '{name}' 已删除", "name": name}

    def get_collection(self, name: str):
        """
        获取 Collection 对象

        Args:
            name: Collection 名称

        Returns:
            ChromaDB Collection 对象

        Raises:
            ValueError: Collection 不存在
        """
        try:
            return self._client.get_collection(name=name)
        except Exception as e:
            raise ValueError(f"Collection '{name}' 不存在: {e}")

    def get_collection_metadata(self, name: str) -> Dict[str, Any]:
        """
        获取 Collection 的详细信息

        Args:
            name: Collection 名称

        Returns:
            Collection 详细信息
        """
        collection = self.get_collection(name)
        try:
            count = collection.count()
        except Exception:
            count = 0

        meta = collection.metadata or {}
        return {
            "name": collection.name,
            "id": collection.id,
            "count": count,
            "metadata": meta,
            "embedding_model": meta.get("embedding_model", "未知"),
            "chunk_strategy": meta.get("chunk_strategy", "未知"),
            "chunk_size": meta.get("chunk_size", "未知"),
            "chunk_overlap": meta.get("chunk_overlap", "未知"),
        }

    # ========== 文档导入 ==========

    def import_files(
        self,
        collection_name: str,
        file_paths: List[str],
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        chunk_strategy: str = "fixed",
        batch_size: int = 100,
    ) -> Dict[str, Any]:
        """
        批量导入文件到指定 Collection

        Args:
            collection_name: 目标 Collection 名称
            file_paths: 文件路径列表
            chunk_size: 分块大小
            chunk_overlap: 分块重叠
            chunk_strategy: 分块策略
            batch_size: 批量插入大小

        Returns:
            导入结果统计
        """
        collection = self.get_collection(collection_name)

        # 初始化解析器和分块器
        parser = FileParser()
        chunker = DocumentChunker(ChunkConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=chunk_strategy,
        ))

        total_docs = 0
        total_chunks = 0
        failed_files = []

        # 获取当前 Embedding 模型
        provider = embedding_manager.get_current_provider()

        for file_path in file_paths:
            try:
                # 解析文件
                documents = parser.parse_file(file_path)
                logger.info(f"文件 {file_path} 解析出 {len(documents)} 个文档")

                # 分块处理
                chunks = chunker.chunk_documents(documents)
                if not chunks:
                    logger.warning(f"文件 {file_path} 没有产生任何分块")
                    continue

                # 生成嵌入向量并批量插入
                texts = [chunk.text for chunk in chunks]
                metadatas = [chunk.metadata for chunk in chunks]
                ids = [f"{Path(file_path).stem}_{chunk.chunk_index}" for chunk in chunks]

                # 分批插入
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i + batch_size]
                    batch_metas = metadatas[i:i + batch_size]
                    batch_ids = ids[i:i + batch_size]

                    # 生成嵌入向量
                    embeddings = provider.embed_texts(batch_texts)

                    collection.add(
                        documents=batch_texts,
                        embeddings=embeddings,
                        metadatas=batch_metas,
                        ids=batch_ids,
                    )
                    logger.debug(f"批量插入 {len(batch_texts)} 条数据到 {collection_name}")

                total_docs += len(documents)
                total_chunks += len(chunks)

            except Exception as e:
                logger.error(f"导入文件 {file_path} 失败: {e}")
                failed_files.append({"file": file_path, "error": str(e)})

        return {
            "collection_name": collection_name,
            "total_files": len(file_paths),
            "total_documents": total_docs,
            "total_chunks": total_chunks,
            "failed_files": failed_files,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "chunk_strategy": chunk_strategy,
        }

    def import_directory(
        self,
        collection_name: str,
        dir_path: str,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        chunk_strategy: str = "fixed",
        batch_size: int = 100,
    ) -> Dict[str, Any]:
        """
        导入整个目录的文件到指定 Collection

        Args:
            collection_name: 目标 Collection 名称
            dir_path: 目录路径
            chunk_size: 分块大小
            chunk_overlap: 分块重叠
            chunk_strategy: 分块策略
            batch_size: 批量插入大小

        Returns:
            导入结果统计
        """
        parser = FileParser()
        # 收集所有文件路径
        file_paths = []
        for fp in Path(dir_path).rglob("*"):
            if fp.is_file() and fp.suffix.lower() in FileParser.SUPPORTED_EXTENSIONS:
                file_paths.append(str(fp))

        return self.import_files(
            collection_name, file_paths,
            chunk_size, chunk_overlap, chunk_strategy, batch_size
        )

    # ========== 相似度搜索 ==========

    def search(
        self,
        collection_name: str,
        query: str,
        top_k: int = 5,
        threshold: float = 0.0,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        在指定 Collection 中进行相似度搜索

        Args:
            collection_name: Collection 名称
            query: 查询文本
            top_k: 返回 top-k 个结果
            threshold: 相似度阈值过滤（0~1，仅保留大于此值的结果）
            metadata_filter: 元数据过滤条件

        Returns:
            搜索结果
        """
        collection = self.get_collection(collection_name)

        # 对查询文本进行嵌入
        query_embedding = embedding_manager.embed_query(query)

        # 构建查询参数
        where = None
        if metadata_filter:
            where = self._build_where_clause(metadata_filter)

        # 执行查询
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # 解析结果
        search_results = []
        if results and results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                distance = results["distances"][0][i] if results["distances"] else 0
                metadata = results["metadatas"][0][i] if results["metadatas"] else {}
                similarity = max(0, 1 - distance)  # 将距离转换为相似度

                # 相似度阈值过滤
                if similarity >= threshold:
                    search_results.append({
                        "content": doc,
                        "metadata": metadata,
                        "similarity": round(similarity, 4),
                        "distance": round(distance, 4),
                    })

        return {
            "query": query,
            "collection_name": collection_name,
            "total_results": len(search_results),
            "results": search_results,
        }

    def _build_where_clause(
        self, filter_dict: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        构建 ChromaDB 的 where 查询子句
        支持简单的键值对过滤

        Args:
            filter_dict: 过滤条件字典

        Returns:
            ChromaDB where 子句
        """
        where_conditions = []
        for key, value in filter_dict.items():
            if isinstance(value, list):
                where_conditions.append({key: {"$in": value}})
            else:
                where_conditions.append({key: value})

        if len(where_conditions) == 1:
            return where_conditions[0]
        elif len(where_conditions) > 1:
            return {"$and": where_conditions}
        return {}

    # ========== 多索引对比 ==========

    def compare_search(
        self,
        collection_names: List[str],
        query: str,
        top_k: int = 5,
    ) -> List[Dict[str, Any]]:
        """
        在多个 Collection 上执行相同查询，对比搜索结果

        Args:
            collection_names: Collection 名称列表
            query: 查询文本
            top_k: 每个集合返回的结果数量

        Returns:
            各 Collection 的搜索结果对比列表
        """
        comparison_results = []
        for col_name in collection_names:
            try:
                result = self.search(col_name, query, top_k=top_k)
                col_meta = self.get_collection_metadata(col_name)
                comparison_results.append({
                    "collection_name": col_name,
                    "embedding_model": col_meta.get("embedding_model", "未知"),
                    "chunk_strategy": col_meta.get("chunk_strategy", "未知"),
                    "chunk_size": col_meta.get("chunk_size", "未知"),
                    "total_results": result["total_results"],
                    "results": result["results"],
                })
            except Exception as e:
                comparison_results.append({
                    "collection_name": col_name,
                    "error": str(e),
                    "results": [],
                })
        return comparison_results

    # ========== 统计信息 ==========

    def get_statistics(self) -> Dict[str, Any]:
        """
        获取系统统计信息

        Returns:
            系统统计信息
        """
        collections = self.list_collections()

        total_documents = 0
        collection_details = []
        for col in collections:
            total_documents += col["count"]
            try:
                col_obj = self._client.get_collection(name=col["name"])
                meta = col_obj.metadata or {}
                collection_details.append({
                    "name": col["name"],
                    "count": col["count"],
                    "embedding_model": meta.get("embedding_model", "未知"),
                    "chunk_strategy": meta.get("chunk_strategy", "未知"),
                })
            except Exception:
                collection_details.append({
                    "name": col["name"],
                    "count": col["count"],
                })

        # 计算 ChromaDB 存储占用
        persist_path = settings.get_chroma_persist_path()
        storage_size = self._get_directory_size(persist_path)

        # 当前模型信息
        model_info = {}
        try:
            provider = embedding_manager.get_current_provider()
            model_info = {
                "name": embedding_manager.get_current_model_name(),
                "dimension": provider.dimension,
                "provider_type": provider.provider_type,
            }
        except RuntimeError:
            model_info = {"name": "未加载", "dimension": 0, "provider_type": "N/A"}

        # 已加载模型列表
        loaded_models = embedding_manager.list_loaded_models()

        return {
            "total_collections": len(collections),
            "total_documents": total_documents,
            "storage_size_mb": round(storage_size / (1024 * 1024), 2),
            "storage_path": str(persist_path),
            "collections": collection_details,
            "current_model": model_info,
            "loaded_models": loaded_models,
        }

    @staticmethod
    def _get_directory_size(path: Path) -> int:
        """
        计算目录的总大小（字节）

        Args:
            path: 目录路径

        Returns:
            总字节数
        """
        total_size = 0
        if path.exists():
            for f in path.rglob("*"):
                if f.is_file():
                    try:
                        total_size += f.stat().st_size
                    except OSError:
                        continue
        return total_size

    # ========== Collection 清空 ==========

    def clear_collection(self, name: str) -> Dict[str, Any]:
        """
        清空指定 Collection 中的所有文档（但保留 Collection 本身）

        Args:
            name: Collection 名称

        Returns:
            操作结果
        """
        # ChromaDB 没有直接清空的 API，需要删除后重建
        try:
            col = self._client.get_collection(name=name)
            meta = col.metadata or {}
            col_id = col.id
        except Exception:
            raise ValueError(f"Collection '{name}' 不存在")

        # 删除并重建
        self._client.delete_collection(name=name)
        new_col = self._client.create_collection(name=name, metadata=meta)

        return {
            "message": f"Collection '{name}' 已清空",
            "name": new_col.name,
            "id": new_col.id,
            "count": 0,
        }
