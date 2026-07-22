"""
内容安全检查器测试 - ContentSafetyChecker 单元测试

测试覆盖：
1. 安全评分计算
2. 风险等级判定
3. 分类检测
4. 中文敏感词匹配
5. 英文敏感词匹配
"""

import sys
import os
import json
import unittest

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from safety.content_safety import ContentSafetyChecker, SafetyScorer


class TestSafetyScorer(unittest.TestCase):
    """测试安全评分计算器"""

    def setUp(self):
        self.scorer = SafetyScorer()

    def test_safe_text_high_score(self):
        """安全文本应获得高评分"""
        report = self.scorer.compute_score([])
        self.assertEqual(report.overall_score, 100.0)
        self.assertEqual(report.risk_level, "safe")

    def test_single_violence_hit(self):
        """单个暴力词命中应降低评分"""
        report = self.scorer.compute_score([
            {"category": "violence", "keyword": "kill", "position": 0}
        ])
        self.assertLess(report.overall_score, 100)
        self.assertIn("violence", report.risk_categories)

    def test_self_harm_highest_weight(self):
        """自残类关键词应产生最大的扣分"""
        violence_report = self.scorer.compute_score([
            {"category": "violence", "keyword": "kill", "position": 0}
        ])
        self_harm_report = self.scorer.compute_score([
            {"category": "self_harm", "keyword": "suicide", "position": 0}
        ])
        # 自残扣分更多
        self.assertLess(self_harm_report.overall_score, violence_report.overall_score)

    def test_risk_level_classification(self):
        """测试风险等级分类"""
        # 安全
        report = self.scorer.compute_score([])
        self.assertEqual(report.risk_level, "safe")

        # 高风险
        many_hits = [{"category": "self_harm", "keyword": f"word{i}", "position": i} for i in range(10)]
        report = self.scorer.compute_score(many_hits)
        self.assertIn(report.risk_level, ["high", "critical"])

    def test_recommendations_generated(self):
        """应生成对应的建议"""
        report = self.scorer.compute_score([
            {"category": "violence", "keyword": "kill", "position": 0}
        ])
        self.assertTrue(len(report.recommendations) > 0)


class TestContentSafetyChecker(unittest.TestCase):
    """测试内容安全检查器"""

    def setUp(self):
        data_dir = os.path.join(PROJECT_ROOT, "data")
        words_path = os.path.join(data_dir, "sensitive_words.json")
        self.checker = ContentSafetyChecker(sensitive_words_path=words_path)

    def test_safe_content(self):
        """测试安全内容"""
        report = self.checker.check("Python是一种编程语言，FastAPI是Web框架。")
        self.assertEqual(report.overall_score, 100.0)
        self.assertEqual(report.risk_level, "safe")
        self.assertEqual(len(report.matched_keywords), 0)

    def test_violence_content(self):
        """测试暴力内容"""
        report = self.checker.check("这篇文章描述了杀戮和恐怖袭击的过程")
        self.assertLess(report.overall_score, 100)
        self.assertIn("violence", report.risk_categories)

    def test_pornography_content(self):
        """测试色情内容"""
        report = self.checker.check("这个网站包含色情和淫秽内容")
        self.assertLess(report.overall_score, 100)
        self.assertIn("pornography", report.risk_categories)

    def test_self_harm_content(self):
        """测试自残内容"""
        report = self.checker.check("我想自杀，不想活了")
        self.assertLess(report.overall_score, 50)
        self.assertIn("self_harm", report.risk_categories)

    def test_english_sensitive_words(self):
        """测试英文敏感词"""
        report = self.checker.check("This content discusses murder and terrorism")
        self.assertLess(report.overall_score, 100)
        self.assertTrue(len(report.matched_keywords) > 0)

    def test_mixed_content(self):
        """测试混合内容（多种类别）"""
        report = self.checker.check("包含暴力杀人、色情内容和歧视言论的混合文本")
        self.assertLess(report.overall_score, 50)
        self.assertTrue(len(report.risk_categories) >= 2)

    def test_check_specific_categories(self):
        """测试只检查特定分类"""
        report, matches = self.checker.check_categories(
            "这篇文章提到了杀人和色情内容",
            categories=["violence"]
        )
        # 应该只检测暴力分类
        for match in matches:
            self.assertEqual(match["category"], "violence")

    def test_add_category_dynamically(self):
        """测试动态添加敏感词分类"""
        self.checker.add_category("custom", ["自定义敏感词A", "自定义敏感词B"])
        report = self.checker.check("这段话包含自定义敏感词A")
        self.assertLess(report.overall_score, 100)


if __name__ == "__main__":
    unittest.main()
