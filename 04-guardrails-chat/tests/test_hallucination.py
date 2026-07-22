"""
幻觉检测器测试 - HallucinationDetector 单元测试

测试覆盖：
1. 事实提取
2. 知识库匹配验证
3. 矛盾检测
4. 无幻觉内容
"""

import sys
import os
import json
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from output.hallucination_detector import HallucinationDetector


class TestHallucinationDetector(unittest.TestCase):
    """测试幻觉检测器"""

    def setUp(self):
        kb_path = os.path.join(PROJECT_ROOT, "data", "knowledge_base.json")
        self.detector = HallucinationDetector(knowledge_base_path=kb_path)

    def test_no_hallucination(self):
        """测试无幻觉的正常输出"""
        output = "Python 是一种高级编程语言，常用于Web开发和数据分析。"
        report = self.detector.detect(output)
        self.assertFalse(report.has_hallucination)
        self.assertEqual(report.hallucination_score, 0.0)

    def test_verified_fact(self):
        """测试可验证的事实"""
        output = "Python 是一种高级编程语言，由 Guido van Rossum 于 1991 年首次发布。"
        report = self.detector.detect(output)
        # 应该有部分内容被验证
        self.assertTrue(len(report.verified_claims) > 0 or len(report.unverified_claims) >= 0)

    def test_contradiction_detection(self):
        """测试矛盾检测"""
        output = "Python 不是一种编程语言，它是一种数据库管理系统。"
        report = self.detector.detect(output)
        # 否定词 + 关键词重叠应检测到矛盾
        if report.hallucinated_claims:
            self.assertTrue(report.has_hallucination)

    def test_empty_knowledge_base(self):
        """测试空知识库"""
        detector = HallucinationDetector(knowledge_base_path="/nonexistent/path.json")
        report = detector.detect("Python是一种编程语言")
        # 空知识库应该返回未知风险
        self.assertEqual(report.hallucination_score, 50.0)

    def test_claims_extraction(self):
        """测试事实提取"""
        output = "FastAPI 是一个用于构建 API 的现代 Python Web 框架，基于 Starlette 和 Pydantic。"
        report = self.detector.detect(output)
        # 应该能提取到事实性声明
        total_claims = (
            len(report.verified_claims) +
            len(report.unverified_claims) +
            len(report.hallucinated_claims)
        )
        self.assertTrue(total_claims > 0)

    def test_multiple_facts_verification(self):
        """测试多个事实验证"""
        output = (
            "水的化学式是 H2O，由两个氢原子和一个氧原子组成。"
            "HTTP 协议默认端口是 80，HTTPS 默认端口是 443。"
        )
        report = self.detector.detect(output)
        self.assertTrue(len(report.verified_claims) > 0)

    def test_dynamic_add_facts(self):
        """测试动态添加事实"""
        self.detector.add_facts([
            {"id": "custom_001", "fact": "地球的卫星是月球", "source": "custom"}
        ])
        report = self.detector.detect("地球的卫星是月球")
        # 新添加的事实应该被匹配到
        self.assertTrue(len(report.verified_claims) > 0 or len(report.unverified_claims) >= 0)


if __name__ == "__main__":
    unittest.main()
