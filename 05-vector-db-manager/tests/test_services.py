# -*- coding: utf-8 -*-
"""
向量数据库管理平台 - 单元测试
测试文件解析、文档分块、向量存储等核心功能
"""
import sys
import os
import json
import tempfile
from pathlib import Path

# 将项目根目录加入 Python 路径
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


class TestFileParser:
    """文件解析器测试"""

    def setup_method(self):
        from services.file_parser import FileParser
        self.parser = FileParser()
        self.tmp_dir = tempfile.mkdtemp()

    def teardown_method(self):
        import shutil
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def _write_file(self, filename, content):
        """辅助方法：写入临时文件"""
        path = Path(self.tmp_dir) / filename
        path.write_text(content, encoding="utf-8")
        return str(path)

    def test_parse_txt(self):
        """测试 TXT 文件解析"""
        content = "第一段落的内容。\n\n第二段落的内容。\n\n第三段落的内容。"
        filepath = self._write_file("test.txt", content)
        docs = self.parser.parse_file(filepath)
        assert len(docs) == 3
        assert "第一段落" in docs[0].content
        assert docs[0].metadata["source"] == "test.txt"
        assert docs[0].metadata["file_type"] == "txt"
        print("[PASS] test_parse_txt")

    def test_parse_markdown(self):
        """测试 Markdown 文件解析"""
        content = "# 标题一\n\n标题一下的内容。\n\n## 标题二\n\n标题二下的内容。"
        filepath = self._write_file("test.md", content)
        docs = self.parser.parse_file(filepath)
        assert len(docs) >= 1
        print("[PASS] test_parse_markdown")

    def test_parse_json_list(self):
        """测试 JSON 列表格式解析"""
        data = [
            {"title": "文档1", "content": "这是第一个文档的内容"},
            {"title": "文档2", "content": "这是第二个文档的内容"},
        ]
        filepath = self._write_file("test.json", json.dumps(data, ensure_ascii=False))
        docs = self.parser.parse_file(filepath)
        assert len(docs) == 2
        assert "第一个文档" in docs[0].content
        assert docs[0].metadata["title"] == "文档1"
        print("[PASS] test_parse_json_list")

    def test_parse_csv(self):
        """测试 CSV 文件解析"""
        content = "姓名,年龄,描述\n张三,25,软件工程师\n李四,30,产品经理"
        filepath = self._write_file("test.csv", content)
        docs = self.parser.parse_file(filepath)
        assert len(docs) == 2
        assert "张三" in docs[0].content
        print("[PASS] test_parse_csv")

    def test_unsupported_format(self):
        """测试不支持的文件格式"""
        filepath = self._write_file("test.pdf", "fake content")
        try:
            self.parser.parse_file(filepath)
            assert False, "应该抛出异常"
        except ValueError as e:
            assert "不支持" in str(e)
            print("[PASS] test_unsupported_format")

    def test_file_not_found(self):
        """测试文件不存在"""
        try:
            self.parser.parse_file("/nonexistent/file.txt")
            assert False, "应该抛出异常"
        except FileNotFoundError:
            print("[PASS] test_file_not_found")


class TestDocumentChunker:
    """文档分块器测试"""

    def test_fixed_chunking(self):
        """测试固定长度分块"""
        from services.chunker import DocumentChunker, ChunkConfig
        from services.file_parser import ParsedDocument

        # 使用换行符分隔的文本
        text = "\n".join(["这是一段测试文本。"] * 100)
        chunker = DocumentChunker(ChunkConfig(
            chunk_size=200,
            chunk_overlap=20,
            strategy="fixed",
        ))
        doc = ParsedDocument(content=text, metadata={"source": "test"})
        chunks = chunker.chunk_document(doc)
        assert len(chunks) >= 1
        # 验证分块被正确生成
        total_len = sum(len(c.text) for c in chunks)
        assert total_len > 0
        print(f"[PASS] test_fixed_chunking (生成 {len(chunks)} 个分块)")

    def test_sentence_chunking(self):
        """测试按句子分块"""
        from services.chunker import DocumentChunker, ChunkConfig
        from services.file_parser import ParsedDocument

        text = "这是第一句话。这是第二句话。这是第三句话。这是第四句话。这是第五句话。"
        chunker = DocumentChunker(ChunkConfig(
            chunk_size=20,
            chunk_overlap=5,
            strategy="sentence",
        ))
        doc = ParsedDocument(content=text, metadata={"source": "test"})
        chunks = chunker.chunk_document(doc)
        assert len(chunks) >= 1
        print(f"[PASS] test_sentence_chunking (生成 {len(chunks)} 个分块)")

    def test_paragraph_chunking(self):
        """测试按段落分块"""
        from services.chunker import DocumentChunker, ChunkConfig
        from services.file_parser import ParsedDocument

        text = "第一段落的内容比较长，这里有很多文字。\n\n第二段落的内容也很长，同样有很多文字。\n\n第三段落是最后一段。"
        chunker = DocumentChunker(ChunkConfig(
            chunk_size=50,
            chunk_overlap=10,
            strategy="paragraph",
        ))
        doc = ParsedDocument(content=text, metadata={"source": "test"})
        chunks = chunker.chunk_document(doc)
        assert len(chunks) >= 1
        print(f"[PASS] test_paragraph_chunking (生成 {len(chunks)} 个分块)")

    def test_sliding_window_chunking(self):
        """测试滑动窗口分块"""
        from services.chunker import DocumentChunker, ChunkConfig
        from services.file_parser import ParsedDocument

        text = "这是一段用于测试滑动窗口分块的文本。" * 10
        chunker = DocumentChunker(ChunkConfig(
            chunk_size=50,
            chunk_overlap=20,
            strategy="sliding_window",
        ))
        doc = ParsedDocument(content=text, metadata={"source": "test"})
        chunks = chunker.chunk_document(doc)
        assert len(chunks) > 1
        # 滑动窗口应该产生更多分块
        print(f"[PASS] test_sliding_window_chunking (生成 {len(chunks)} 个分块)")

    def test_empty_text(self):
        """测试空文本"""
        from services.chunker import DocumentChunker, ChunkConfig
        from services.file_parser import ParsedDocument

        chunker = DocumentChunker(ChunkConfig(strategy="fixed"))
        doc = ParsedDocument(content="", metadata={"source": "test"})
        chunks = chunker.chunk_document(doc)
        assert len(chunks) == 0
        print("[PASS] test_empty_text")

    def test_unsupported_strategy(self):
        """测试不支持的策略"""
        from services.chunker import DocumentChunker, ChunkConfig

        try:
            DocumentChunker(ChunkConfig(strategy="unknown"))
            assert False, "应该抛出异常"
        except ValueError:
            print("[PASS] test_unsupported_strategy")


class TestEmbeddingManager:
    """Embedding 模型管理器测试（仅测试接口，不加载实际模型）"""

    def test_list_empty_models(self):
        """测试列出空模型列表"""
        from embeddings.manager import EmbeddingManager
        manager = EmbeddingManager()
        models = manager.list_loaded_models()
        assert models == []
        print("[PASS] test_list_empty_models")

    def test_no_current_model(self):
        """测试未加载模型时获取当前模型"""
        from embeddings.manager import EmbeddingManager
        manager = EmbeddingManager()
        try:
            manager.get_current_provider()
            assert False, "应该抛出异常"
        except RuntimeError as e:
            assert "未加载" in str(e)
            print("[PASS] test_no_current_model")


class TestConfig:
    """配置模块测试"""

    def test_settings_defaults(self):
        """测试配置默认值"""
        from app.config import Settings
        s = Settings()
        assert s.app_host == "0.0.0.0"
        assert s.app_port == 8000
        assert s.default_chunk_size == 500
        assert s.default_chunk_overlap == 50
        print("[PASS] test_settings_defaults")

    def test_settings_paths(self):
        """测试路径创建方法"""
        from app.config import Settings
        s = Settings()
        # 这些方法应该返回 Path 对象
        assert s.get_chroma_persist_path() is not None
        assert s.get_model_cache_path() is not None
        assert s.get_upload_path() is not None
        assert s.get_data_path() is not None
        print("[PASS] test_settings_paths")


def run_tests():
    """运行所有测试"""
    print("\n" + "=" * 60)
    print("向量数据库管理平台 - 运行单元测试")
    print("=" * 60 + "\n")

    test_classes = [
        TestFileParser,
        TestDocumentChunker,
        TestEmbeddingManager,
        TestConfig,
    ]

    passed = 0
    failed = 0

    for test_class in test_classes:
        instance = test_class()
        test_methods = [m for m in dir(instance) if m.startswith("test_")]
        for method_name in test_methods:
            try:
                # 执行 setup
                if hasattr(instance, "setup_method"):
                    instance.setup_method()
                # 执行测试
                getattr(instance, method_name)()
                # 执行 teardown
                if hasattr(instance, "teardown_method"):
                    instance.teardown_method()
                passed += 1
            except Exception as e:
                print(f"[FAIL] {test_class.__name__}.{method_name}: {e}")
                failed += 1

    print("\n" + "=" * 60)
    print(f"测试完成: {passed} 通过, {failed} 失败")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
