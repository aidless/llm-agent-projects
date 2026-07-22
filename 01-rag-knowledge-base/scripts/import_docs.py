"""
批量导入脚本
将指定目录下的文档批量导入到知识库
"""
import os
import sys
import argparse
import time
from pathlib import Path

# 确保项目根目录在 sys.path 中
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from core.config import settings, setup_logging
from core.parsers.document_parser import DocumentParser
from core.chunking.chunking import Chunker
from core.embeddings.embeddings import get_embedding_manager
from core.retrieval.vector_store import VectorStoreManager
from core.retrieval.retriever import HybridRetriever
from loguru import logger


def import_documents(directory: str, chunk_strategy: str = "semantic") -> None:
    """
    批量导入文档

    Args:
        directory: 文档目录路径
        chunk_strategy: 分块策略
    """
    setup_logging()
    settings.ensure_directories()

    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"目录不存在: {directory}")
        return

    # 找到所有支持的文件
    supported_exts = {".pdf", ".docx", ".doc", ".md", ".txt"}
    files = []
    for f in dir_path.rglob("*"):
        if f.suffix.lower() in supported_exts:
            files.append(f)

    if not files:
        print(f"未找到支持的文档文件。支持格式: {supported_exts}")
        return

    print(f"找到 {len(files)} 个文档文件")

    # 初始化组件
    print("正在初始化组件...")
    parser = DocumentParser()
    chunker = Chunker(strategy=chunk_strategy, chunk_size=settings.chunk_size)
    embedding_manager = get_embedding_manager()
    vector_store = VectorStoreManager()
    retriever = HybridRetriever(vector_store=vector_store)

    # 逐个处理文档
    success_count = 0
    fail_count = 0
    total_start = time.time()

    for i, file_path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] 处理: {file_path.name}")
        try:
            # 解析
            parsed = parser.parse(str(file_path))

            # 分块
            chunks = chunker.chunk_document(parsed)
            if not chunks:
                print(f"  警告: 分块结果为空，跳过")
                fail_count += 1
                continue

            # 向量化
            texts = [c.content for c in chunks]
            metadatas = [c.metadata for c in chunks]
            embeddings = embedding_manager.embed_documents(texts)

            # 存储
            vector_store.add_texts(
                texts=texts,
                metadatas=metadatas,
                embeddings=embeddings,
            )

            success_count += 1
            print(f"  成功: {len(chunks)} 个文本块")

        except Exception as e:
            print(f"  失败: {e}")
            fail_count += 1

    # 重建 BM25 索引
    print("\n重建 BM25 索引...")
    retriever.build_index()

    elapsed = time.time() - total_start
    print(f"\n{'=' * 50}")
    print(f"导入完成!")
    print(f"  成功: {success_count} 个文档")
    print(f"  失败: {fail_count} 个文档")
    print(f"  总耗时: {elapsed:.1f}s")
    print(f"  知识库文档总数: {vector_store.count()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="批量导入文档到 RAG 知识库")
    parser.add_argument("directory", help="文档目录路径")
    parser.add_argument("--strategy", default="semantic", choices=["fixed", "paragraph", "semantic"],
                        help="分块策略 (默认: semantic)")
    args = parser.parse_args()

    import_documents(args.directory, args.strategy)