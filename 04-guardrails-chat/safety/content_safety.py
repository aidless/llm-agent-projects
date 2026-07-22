"""
内容安全检测模块 - ContentSafetyChecker 和 SafetyScorer

提供内容安全评分（0-100）和分类检测功能：
- 多维度关键词匹配
- 加权评分系统
- 实时风险评估
"""

import json
import re
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class SafetyReport:
    """安全评估报告"""
    overall_score: float = 100.0                   # 综合安全评分 0-100（100=最安全）
    category_scores: dict = field(default_factory=dict)  # 各分类安全评分
    risk_categories: List[str] = field(default_factory=list)  # 存在风险的分类
    matched_keywords: List[dict] = field(default_factory=list)  # 命中的关键词详情
    risk_level: str = "safe"                       # 风险等级: safe / low / medium / high / critical
    recommendations: List[str] = field(default_factory=list)  # 建议


class SafetyScorer:
    """安全评分计算器"""

    # 各风险类别的权重（影响综合评分的权重）
    CATEGORY_WEIGHTS = {
        "self_harm": 1.5,      # 自残类权重最高
        "violence": 1.3,        # 暴力类
        "pornography": 1.3,     # 色情类
        "politics": 1.0,        # 政治敏感
        "illegal": 1.1,         # 违法类
        "discrimination": 0.9,  # 歧视类
        "prompt_injection": 1.2,  # Prompt 注入
    }

    # 风险等级阈值
    RISK_THRESHOLDS = {
        "safe": 90,
        "low": 70,
        "medium": 50,
        "high": 30,
        "critical": 0,
    }

    def compute_score(
        self,
        matched_keywords: List[dict],
        text_length: int = 0
    ) -> SafetyReport:
        """
        根据命中的关键词计算综合安全评分

        评分逻辑：
        - 基础分 100 分
        - 每个命中的关键词按其类别权重扣分
        - 同一类别多次命中会有递增扣分效果
        - 最终评分取 0-100 范围

        Args:
            matched_keywords: 命中的关键词列表 [{category, keyword, position}]
            text_length: 文本长度（用于归一化）

        Returns:
            SafetyReport: 安全评估报告
        """
        report = SafetyReport()
        base_score = 100.0

        # 按分类统计命中数
        category_hits: Dict[str, int] = {}
        for match in matched_keywords:
            cat = match.get("category", "unknown")
            category_hits[cat] = category_hits.get(cat, 0) + 1
            report.matched_keywords.append(match)

        # 计算各分类扣分
        total_deduction = 0.0
        for category, hits in category_hits.items():
            weight = self.CATEGORY_WEIGHTS.get(category, 1.0)
            # 每次命中的扣分：基础15分 * 权重，同一类多次命中有累加效果（衰减）
            deduction = 0
            for i in range(hits):
                # 递减扣分：第一次扣分最多，后续逐渐减少
                deduction += 15 * weight * (0.8 ** i)
            total_deduction += deduction
            category_score = max(0, base_score - deduction)
            report.category_scores[category] = round(category_score, 1)
            if category_score < 85:
                report.risk_categories.append(category)

        # 计算综合评分
        report.overall_score = max(0, base_score - total_deduction)

        # 确定风险等级
        for level, threshold in sorted(
            self.RISK_THRESHOLDS.items(), key=lambda x: x[1], reverse=True
        ):
            if report.overall_score >= threshold:
                report.risk_level = level
                break

        # 生成建议
        report.recommendations = self._generate_recommendations(report)

        return report

    def _generate_recommendations(self, report: SafetyReport) -> List[str]:
        """根据安全报告生成改进建议"""
        recommendations = []
        if "self_harm" in report.risk_categories:
            recommendations.append("检测到自残相关内容，建议立即进行人工审核并提供心理援助信息")
        if "violence" in report.risk_categories:
            recommendations.append("检测到暴力相关内容，建议进行内容审核或过滤")
        if "pornography" in report.risk_categories:
            recommendations.append("检测到色情相关内容，建议进行内容过滤")
        if "politics" in report.risk_categories:
            recommendations.append("检测到政治敏感内容，建议进行合规审查")
        if "illegal" in report.risk_categories:
            recommendations.append("检测到违法相关内容，建议进行安全审核")
        if "discrimination" in report.risk_categories:
            recommendations.append("检测到歧视相关内容，建议进行内容审核")
        if "prompt_injection" in report.risk_categories:
            recommendations.append("检测到 Prompt 注入尝试，建议加强输入过滤")
        if not recommendations and report.overall_score < 100:
            recommendations.append("内容基本安全，但存在少量风险词汇，建议持续监控")
        return recommendations


class ContentSafetyChecker:
    """内容安全检查器：基于关键词库进行多维度内容安全检测"""

    def __init__(self, sensitive_words_path: Optional[str] = None):
        """
        初始化内容安全检查器

        Args:
            sensitive_words_path: 敏感词库 JSON 文件路径。
                                 如果为 None，将尝试从默认路径加载。
        """
        self.sensitive_words: Dict[str, List[str]] = {}
        self._compiled_patterns: Dict[str, List[re.Pattern]] = {}
        self.scorer = SafetyScorer()

        # 加载敏感词库
        if sensitive_words_path is None:
            # 默认路径
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            sensitive_words_path = os.path.join(base_dir, "data", "sensitive_words.json")

        if os.path.exists(sensitive_words_path):
            self._load_sensitive_words(sensitive_words_path)

    def _load_sensitive_words(self, filepath: str) -> None:
        """
        从 JSON 文件加载敏感词库

        文件格式: { "分类名": ["词1", "词2", ...] }

        Args:
            filepath: JSON 文件路径
        """
        with open(filepath, "r", encoding="utf-8") as f:
            self.sensitive_words = json.load(f)

        # 预编译正则表达式，提高匹配效率
        for category, words in self.sensitive_words.items():
            patterns = []
            for word in words:
                escaped = re.escape(word)
                patterns.append(re.compile(escaped, re.IGNORECASE))
            self._compiled_patterns[category] = patterns

    def add_category(self, category: str, words: List[str]) -> None:
        """
        动态添加敏感词分类

        Args:
            category: 分类名称
            words: 敏感词列表
        """
        self.sensitive_words[category] = words
        patterns = []
        for word in words:
            escaped = re.escape(word)
            patterns.append(re.compile(escaped, re.IGNORECASE))
        self._compiled_patterns[category] = patterns

    def check(self, text: str) -> SafetyReport:
        """
        对文本进行完整的内容安全检查

        Args:
            text: 待检查的文本

        Returns:
            SafetyReport: 安全评估报告，包含评分、风险等级和建议
        """
        matched_keywords = []

        for category, patterns in self._compiled_patterns.items():
            text_lower = text.lower()
            for pattern in patterns:
                for match in pattern.finditer(text_lower):
                    matched_keywords.append({
                        "category": category,
                        "keyword": match.group(),
                        "position": match.start(),
                    })

        return self.scorer.compute_score(matched_keywords, len(text))

    def check_categories(self, text: str, categories: List[str]) -> Tuple[SafetyReport, List[dict]]:
        """
        只检查指定分类的内容安全

        Args:
            text: 待检查的文本
            categories: 要检查的分类名称列表

        Returns:
            Tuple[SafetyReport, List[dict]]: (安全报告, 命中的关键词列表)
        """
        matched_keywords = []

        for category in categories:
            if category not in self._compiled_patterns:
                continue
            text_lower = text.lower()
            for pattern in self._compiled_patterns[category]:
                for match in pattern.finditer(text_lower):
                    matched_keywords.append({
                        "category": category,
                        "keyword": match.group(),
                        "position": match.start(),
                    })

        report = self.scorer.compute_score(matched_keywords, len(text))
        return report, matched_keywords
