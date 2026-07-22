"""
输出守卫测试 - OutputGuard 单元测试

测试覆盖：
1. 有害内容检测（暴力 / 色情 / 政治敏感 / 自残）
2. JSON 格式校验
3. JSON Schema 校验
4. 输出长度限制
5. 安全输出通过
"""

import sys
import os
import json
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from guards.output_guard import OutputGuard

# 加载敏感词库
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
SENSITIVE_WORDS_PATH = os.path.join(DATA_DIR, "sensitive_words.json")

def load_sensitive_words():
    """加载敏感词库"""
    if os.path.exists(SENSITIVE_WORDS_PATH):
        with open(SENSITIVE_WORDS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


class TestOutputGuardHarmfulContent(unittest.TestCase):
    """测试有害内容检测"""

    def setUp(self):
        words = load_sensitive_words()
        self.guard = OutputGuard(sensitive_words=words)

    def test_violence_detection(self):
        """测试暴力内容检测"""
        result = self.guard.check("这是一个关于杀戮和炸弹的暴力描述")
        self.assertFalse(result.is_safe)
        self.assertTrue(len(result.harmful_content) > 0)

    def test_pornography_detection(self):
        """测试色情内容检测"""
        result = self.guard.check("这个网站包含色情和淫秽内容")
        self.assertFalse(result.is_safe)
        self.assertTrue(len(result.harmful_content) > 0)

    def test_politics_detection(self):
        """测试政治敏感内容检测"""
        result = self.guard.check("这是一个关于颠覆政权和分裂国家的讨论")
        self.assertFalse(result.is_safe)

    def test_self_harm_detection(self):
        """测试自残内容检测"""
        result = self.guard.check("我最近一直有自杀的想法，不想活了")
        self.assertFalse(result.is_safe)

    def test_safe_output(self):
        """测试安全输出通过"""
        result = self.guard.check("Python是一种非常流行的编程语言，广泛用于Web开发。")
        self.assertTrue(result.is_safe)
        self.assertTrue(len(result.harmful_content) == 0)

    def test_harmful_content_sanitization(self):
        """测试有害内容脱敏"""
        result = self.guard.check("这段话包含杀人和炸弹等暴力词汇")
        if not result.is_safe:
            # 确认敏感词被打码
            self.assertIn("*", result.sanitized_output)


class TestOutputGuardJSONValidation(unittest.TestCase):
    """测试 JSON 格式校验"""

    def setUp(self):
        words = load_sensitive_words()
        self.guard = OutputGuard(
            sensitive_words=words,
            enable_json_validation=True,
        )

    def test_valid_json(self):
        """测试有效 JSON"""
        result = self.guard.check('{"name": "张三", "age": 25}')
        self.assertTrue(result.is_valid_json)
        self.assertIsNone(result.json_error)

    def test_invalid_json(self):
        """测试无效 JSON"""
        result = self.guard.check('{"name": "张三", "age": }')
        self.assertFalse(result.is_valid_json)
        self.assertIsNotNone(result.json_error)

    def test_json_in_markdown_block(self):
        """测试 markdown 代码块中的 JSON"""
        text = '```json\n{"name": "test"}\n```'
        result = self.guard.check(text)
        self.assertTrue(result.is_valid_json)

    def test_json_schema_validation(self):
        """测试 JSON Schema 校验"""
        schema = {
            "type": "object",
            "required": ["name", "age"],
            "properties": {
                "name": {"type": "string", "minLength": 1},
                "age": {"type": "integer", "minimum": 0},
            },
        }
        self.guard.json_schema = schema

        # 缺少必填字段
        result = self.guard.check('{"name": "test"}')
        self.assertFalse(result.is_valid_json)
        self.assertIsNotNone(result.json_error)

        # 类型错误
        result = self.guard.check('{"name": "test", "age": "二十五"}')
        self.assertFalse(result.is_valid_json)

        # 有效数据
        result = self.guard.check('{"name": "test", "age": 25}')
        self.assertTrue(result.is_valid_json)


class TestOutputGuardLengthLimit(unittest.TestCase):
    """测试输出长度限制"""

    def setUp(self):
        words = load_sensitive_words()
        self.guard = OutputGuard(
            sensitive_words=words,
            max_output_length=50,
        )

    def test_normal_length(self):
        """测试正常长度"""
        result = self.guard.check("这是一条正常的短回复")
        self.assertTrue(result.is_safe)

    def test_exceeded_length(self):
        """测试超长输出"""
        long_text = "这是一条很长的" + "内容" * 100
        result = self.guard.check(long_text)
        self.assertFalse(result.is_safe)
        self.assertIn("长度", result.blocked_reason)


if __name__ == "__main__":
    unittest.main()
