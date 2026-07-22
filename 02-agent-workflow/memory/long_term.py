# ============================================
# 长期向量记忆
# 基于 ChromaDB 实现语义化长期记忆存储与检索
# 支持将重要信息持久化并在未来查询中语义召回
# ============================================

import uuid
from datetime import datetime
from typing import Optional, List
from loguru import logger


class LongTermMemory:
    """长期记忆：基于 ChromaDB 的向量存储与语义检索"""

    def __init__(
        self,
        persist_dir: str = "./data/chroma_db",
        collection_name: str = "agent_memory",
        similarity_top_k: int = 5,
    ):
        """
        初始化长期记忆

        Args:
            persist_dir: ChromaDB 持久化目录
            collection_name: 集合名称
            similarity_top_k: 检索时返回的最相似结果数
        """
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.similarity_top_k = similarity_top_k
        self._client = None
        self._collection = None

    def _ensure_collection(self):
        """延迟初始化 ChromaDB 客户端和集合"""
        if self._collection is not None:
            return

        import chromadb
        from chromadb.config import Settings as ChromaSettings

        # 初始化 ChromaDB 客户端（持久化模式）
        self._client = chromadb.PersistentClient(
            path=self.persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        # 获取或创建集合
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "AI Agent 长期向量记忆库"},
        )
        logger.info(
            f"[长期记忆] ChromaDB 已初始化, "
            f"集合={self.collection_name}, "
            f"现有记录数={self._collection.count()}"
        )

    def add_memory(
        self,
        content: str,
        metadata: Optional[dict] = None,
        memory_id: Optional[str] = None,
    ) -> str:
        """
        添加一条长期记忆

        Args:
            content: 记忆内容文本
            metadata: 附带元数据（如来源、类型等）
            memory_id: 指定记忆 ID（可选，默认自动生成 UUID）

        Returns:
            str: 记忆的唯一 ID
        """
        self._ensure_collection()

        if memory_id is None:
            memory_id = str(uuid.uuid4())

        # 构建元数据，包含时间戳
        entry_metadata = metadata or {}
        entry_metadata.setdefault("created_at", datetime.now().isoformat())
        entry_metadata.setdefault("source", "agent")

        self._collection.add(
            documents=[content],
            metadatas=[entry_metadata],
            ids=[memory_id],
        )
        logger.debug(f"[长期记忆] 添加记忆 id={memory_id}, 内容长度={len(content)}")
        return memory_id

    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        where: Optional[dict] = None,
    ) -> List[dict]:
        """
        语义检索长期记忆

        Args:
            query: 查询文本
            top_k: 返回最相似的 K 条（默认使用初始化设置）
            where: 元数据过滤条件（如 {"source": "planner"}）

        Returns:
            list[dict]: 检索结果列表，每条包含 id, content, metadata, distance
        """
        self._ensure_collection()

        if self._collection.count() == 0:
            logger.debug("[长期记忆] 记忆库为空，跳过检索")
            return []

        k = top_k or self.similarity_top_k
        results = self._collection.query(
            query_texts=[query],
            n_results=min(k, self._collection.count()),
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        memories = []
        if results["ids"] and results["ids"][0]:
            for i, mem_id in enumerate(results["ids"][0]):
                memories.append({
                    "id": mem_id,
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0.0,
                })

        logger.debug(f"[长期记忆] 检索 query='{query[:50]}...', 返回 {len(memories)} 条")
        return memories

    def delete_memory(self, memory_id: str) -> bool:
        """
        删除指定记忆

        Args:
            memory_id: 记忆 ID

        Returns:
            bool: 是否删除成功
        """
        self._ensure_collection()

        try:
            self._collection.delete(ids=[memory_id])
            logger.debug(f"[长期记忆] 删除记忆 id={memory_id}")
            return True
        except Exception as e:
            logger.error(f"[长期记忆] 删除失败 id={memory_id}, 错误={e}")
            return False

    def update_memory(self, memory_id: str, content: str, metadata: Optional[dict] = None) -> bool:
        """
        更新已有记忆

        Args:
            memory_id: 记忆 ID
            content: 新内容
            metadata: 新元数据

        Returns:
            bool: 是否更新成功
        """
        self._ensure_collection()

        try:
            update_data = {"documents": [content]}
            if metadata:
                update_data["metadatas"] = [metadata]
            self._collection.update(ids=[memory_id], **update_data)
            logger.debug(f"[长期记忆] 更新记忆 id={memory_id}")
            return True
        except Exception as e:
            logger.error(f"[长期记忆] 更新失败 id={memory_id}, 错误={e}")
            return False

    def get_all_memories(self, limit: int = 100) -> List[dict]:
        """
        获取所有记忆（分页）

        Args:
            limit: 最大返回数

        Returns:
            list[dict]: 记忆列表
        """
        self._ensure_collection()

        if self._collection.count() == 0:
            return []

        results = self._collection.get(
            limit=limit,
            include=["documents", "metadatas"],
        )

        memories = []
        if results["ids"]:
            for i, mem_id in enumerate(results["ids"]):
                memories.append({
                    "id": mem_id,
                    "content": results["documents"][i],
                    "metadata": results["metadatas"][i] if results["metadatas"] else {},
                })
        return memories

    def count(self) -> int:
        """获取记忆总数"""
        self._ensure_collection()
        return self._collection.count()

    def clear(self) -> None:
        """清空所有长期记忆"""
        self._ensure_collection()
        import chromadb
        # 删除并重建集合
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "AI Agent 长期向量记忆库"},
        )
        logger.info("[长期记忆] 已清空所有记忆")

    def __repr__(self) -> str:
        return f"LongTermMemory(collection={self.collection_name}, count={self.count() if self._collection else '未初始化'})"
