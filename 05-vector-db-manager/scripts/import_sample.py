# -*- coding: utf-8 -*-
"""
示例数据导入脚本
自动加载模型、创建集合并导入示例数据集
使用方法: python scripts/import_sample.py
"""
import sys
import os
from pathlib import Path

# 将项目根目录加入 Python 路径
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from app.config import settings
from embeddings.manager import embedding_manager
from services.vector_store import VectorStoreService
from services.chunker import ChunkConfig


def main():
    """导入示例数据"""
    print("=" * 60)
    print("向量数据库管理平台 - 示例数据导入")
    print("=" * 60)

    # Step 1: 加载 Embedding 模型
    model_name = settings.default_embedding_model
    print(f"\n[1/4] 正在加载 Embedding 模型: {model_name}")
    print("      首次加载需要下载模型，可能需要几分钟...")
    try:
        provider = embedding_manager.load_model(model_name, provider_type="local")
        print(f"      模型加载成功! 维度: {provider.dimension}")
    except Exception as e:
        print(f"      模型加载失败: {e}")
        print("      请检查网络连接或更换模型名称")
        sys.exit(1)

    # Step 2: 初始化向量存储
    print(f"\n[2/4] 初始化向量存储 (目录: {settings.get_chroma_persist_path()})")
    store = VectorStoreService()

    # Step 3: 创建测试集合
    configs = [
        {
            "name": "demo_fixed_500",
            "desc": "固定分块-500字符",
            "strategy": "fixed",
            "chunk_size": 500,
            "overlap": 50,
        },
        {
            "name": "demo_sentence_300",
            "desc": "句子分块-300字符",
            "strategy": "sentence",
            "chunk_size": 300,
            "overlap": 30,
        },
        {
            "name": "demo_paragraph_800",
            "desc": "段落分块-800字符",
            "strategy": "paragraph",
            "chunk_size": 800,
            "overlap": 100,
        },
    ]

    data_dir = settings.get_data_path()
    sample_file = data_dir / "sample_knowledge.json"

    if not sample_file.exists():
        print(f"\n      示例数据文件不存在: {sample_file}")
        print("      请确保 data/sample_knowledge.json 文件存在")
        sys.exit(1)

    print(f"\n[3/4] 创建集合并导入数据...")

    for cfg in configs:
        print(f"\n      --- 集合: {cfg['name']} ({cfg['desc']}) ---")
        try:
            # 创建集合
            collection = store.create_collection(
                name=cfg["name"],
                description=cfg["desc"],
            )
            print(f"      集合创建成功: {collection['name']}")

            # 导入文件
            result = store.import_files(
                collection_name=cfg["name"],
                file_paths=[str(sample_file)],
                chunk_size=cfg["chunk_size"],
                chunk_overlap=cfg["overlap"],
                chunk_strategy=cfg["strategy"],
            )
            print(f"      导入完成: {result['total_documents']} 文档 -> {result['total_chunks']} 分块")
            if result["failed_files"]:
                print(f"      失败文件: {result['failed_files']}")
        except Exception as e:
            print(f"      创建/导入失败: {e}")

    # Step 4: 搜索测试
    print(f"\n[4/4] 搜索测试...")
    test_queries = [
        "什么是向量数据库？",
        "RAG 技术是什么？",
        "如何选择分块策略？",
        "HNSW 算法的原理",
    ]

    for query in test_queries:
        print(f"\n      查询: \"{query}\"")
        try:
            result = store.search(
                collection_name="demo_fixed_500",
                query=query,
                top_k=2,
            )
            if result["results"]:
                for i, r in enumerate(result["results"]):
                    print(f"        #{i+1} [相似度: {r['similarity']:.4f}] {r['content'][:80]}...")
            else:
                print(f"        无结果")
        except Exception as e:
            print(f"        搜索失败: {e}")

    # 打印统计
    print(f"\n" + "=" * 60)
    stats = store.get_statistics()
    print(f"导入完成!")
    print(f"  集合数量: {stats['total_collections']}")
    print(f"  文档总数: {stats['total_documents']}")
    print(f"  存储占用: {stats['storage_size_mb']} MB")
    print(f"\n现在可以访问 http://localhost:8000 使用 Web UI")
    print("=" * 60)


if __name__ == "__main__":
    main()
