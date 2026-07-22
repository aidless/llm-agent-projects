"""
幻觉检测模块 - HallucinationDetector

检测 LLM 输出中可能存在的幻觉（与预设知识库不符的事实声明）。

实现原理：
1. 从 LLM 输出中提取事实性陈述
2. 与知识库中的已知事实进行对比
3. 计算相似度和一致性分数
4. 标记可能的幻觉内容
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class HallucinationReport:
    """幻觉检测报告"""
    has_hallucination: bool = False                # 是否检测到幻觉
    hallucination_score: float = 0.0               # 幻觉评分 0-100（越高越可能存在幻觉）
    hallucinated_claims: List[dict] = field(default_factory=list)  # 疑似幻觉的声明
    verified_claims: List[dict] = field(default_factory=list)      # 已验证的声明
    unverified_claims: List[dict] = field(default_factory=list)    # 未能验证的声明
    knowledge_base_matched: int = 0                # 匹配到的知识库条目数


class HallucinationDetector:
    """
    幻觉检测器

    通过将 LLM 输出中的事实性陈述与预设知识库对比，检测可能的幻觉。
    """

    # 事实性陈述的模式（用于从文本中提取可能的事实）
    FACT_PATTERNS = [
        # "X 是 Y" 格式
        r"(.{2,30})\s*(?:是|为|属于|等于)\s*(.{2,50})",
        # "X 成立于 YYYY" / "X 创建于 YYYY"
        r"(.{2,30})\s*(?:成立于|创建于|诞生于|建于|始建|出现于)\s*(.{2,30})",
        # "X 的首都是 Y" / "X 位于 Y"
        r"(.{2,30})\s*(?:的首都是|位于|在|地处)\s*(.{2,30})",
        # 数字相关事实
        r"(.{2,30})\s*(?:约|大约|约为|大约为|有|有约|人口)\s*(\d[\d,.]*)\s*(.{2,20})",
        # "X 由 Y 开发/发明/发现"
        r"(.{2,30})\s*(?:由|被)\s*(.{2,30})\s*(?:开发|发明|发现|创造|设计|编写|创建)",
        # 英文事实模式
        r"(.{2,40})\s+(?:is|are|was|were)\s+(?:a|an|the)\s+(.{2,50})",
        r"(.{2,40})\s+(?:has|have|had)\s+(?:a|an|the|about|around)\s+(.{2,50})",
    ]

    def __init__(self, knowledge_base_path: Optional[str] = None):
        """
        初始化幻觉检测器

        Args:
            knowledge_base_path: 知识库 JSON 文件路径。
                               如果为 None，将尝试从默认路径加载。
        """
        self.knowledge_base: List[dict] = []

        if knowledge_base_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            knowledge_base_path = os.path.join(base_dir, "data", "knowledge_base.json")

        if os.path.exists(knowledge_base_path):
            self._load_knowledge_base(knowledge_base_path)

    def _load_knowledge_base(self, filepath: str) -> None:
        """从 JSON 文件加载知识库"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.knowledge_base = data.get("facts", [])

    def add_facts(self, facts: List[dict]) -> None:
        """
        动态添加事实到知识库

        Args:
            facts: 事实列表，每个事实包含 "id", "fact", "source" 字段
        """
        self.knowledge_base.extend(facts)

    def detect(self, output_text: str) -> HallucinationReport:
        """
        对 LLM 输出进行幻觉检测

        流程：
        1. 从输出中提取事实性声明
        2. 与知识库中的每条事实计算相似度
        3. 判断是否存在矛盾或无法验证的声明

        Args:
            output_text: LLM 输出的文本

        Returns:
            HallucinationReport: 幻觉检测报告
        """
        report = HallucinationReport()

        if not self.knowledge_base:
            # 没有知识库，无法进行幻觉检测
            report.hallucination_score = 50.0  # 未知风险
            return report

        # 1. 提取事实性声明
        claims = self._extract_claims(output_text)

        # 2. 对每个声明进行验证
        for claim in claims:
            verification = self._verify_claim(claim, output_text)
            if verification["status"] == "contradicted":
                report.hallucinated_claims.append({
                    "claim": claim,
                    "details": verification["details"],
                    "contradicted_by": verification.get("fact_id"),
                    "confidence": verification.get("confidence", 0.5),
                })
            elif verification["status"] == "verified":
                report.verified_claims.append({
                    "claim": claim,
                    "verified_by": verification.get("fact_id"),
                    "confidence": verification.get("confidence", 0.8),
                })
            else:
                report.unverified_claims.append({
                    "claim": claim,
                    "reason": verification.get("details", "未找到对应知识库条目"),
                })

        # 3. 计算幻觉评分
        total_claims = len(claims)
        if total_claims > 0:
            hallucinated_ratio = len(report.hallucinated_claims) / total_claims
            report.hallucination_score = round(hallucinated_ratio * 100, 1)
            report.has_hallucination = len(report.hallucinated_claims) > 0
        else:
            # 没有提取到事实性声明
            report.hallucination_score = 0.0

        report.knowledge_base_matched = len(report.verified_claims)
        return report

    def _extract_claims(self, text: str) -> List[str]:
        """
        从文本中提取事实性声明

        使用正则模式匹配常见的事实陈述格式。

        Args:
            text: 输入文本

        Returns:
            List[str]: 提取到的事实性声明列表
        """
        claims = []
        seen = set()  # 去重

        for pattern in self.FACT_PATTERNS:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                claim_text = match.group(0).strip()
                # 过滤掉过短或过长的声明
                if 10 <= len(claim_text) <= 100 and claim_text not in seen:
                    claims.append(claim_text)
                    seen.add(claim_text)

        return claims

    def _verify_claim(self, claim: str, full_output: str) -> dict:
        """
        验证单个声明是否与知识库一致

        验证策略：
        1. 对声明和知识库条目计算关键词重叠度
        2. 检查是否包含矛盾关键词（如"不是"、"并非"）
        3. 综合判断一致性

        Args:
            claim: 待验证的声明
            full_output: 完整输出文本（用于上下文分析）

        Returns:
            dict: 验证结果 {status, details, fact_id, confidence}
        """
        best_match = None
        best_similarity = 0.0

        claim_keywords = self._extract_keywords(claim)

        for fact in self.knowledge_base:
            fact_keywords = self._extract_keywords(fact["fact"])

            # 计算关键词重叠度
            overlap = len(claim_keywords & fact_keywords)
            if overlap == 0:
                continue

            # Jaccard 相似度
            union = len(claim_keywords | fact_keywords)
            similarity = overlap / union if union > 0 else 0

            if similarity > best_similarity:
                best_similarity = similarity
                best_match = fact

        if best_match and best_similarity >= 0.3:
            # 有较好的匹配，进一步检查是否矛盾
            is_contradicted = self._check_contradiction(claim, best_match["fact"])
            confidence = round(best_similarity, 2)

            if is_contradicted:
                return {
                    "status": "contradicted",
                    "details": f"声明与知识库事实矛盾",
                    "fact_id": best_match.get("id"),
                    "confidence": confidence,
                }
            else:
                return {
                    "status": "verified",
                    "details": f"声明与知识库事实一致",
                    "fact_id": best_match.get("id"),
                    "confidence": confidence,
                }
        elif best_match and best_similarity >= 0.15:
            # 弱匹配
            return {
                "status": "unverified",
                "details": "声明与知识库条目部分匹配，无法确认",
                "fact_id": best_match.get("id"),
                "confidence": round(best_similarity, 2),
            }
        else:
            return {
                "status": "unverified",
                "details": "未找到对应知识库条目",
            }

    def _extract_keywords(self, text: str) -> set:
        """
        从文本中提取关键词

        简单实现：去除停用词后，取长度>=2的中文字符和英文单词

        Args:
            text: 输入文本

        Returns:
            set: 关键词集合
        """
        # 中文停用词
        chinese_stopwords = {
            "的", "了", "在", "是", "我", "有", "和", "就", "不", "人", "都",
            "一", "一个", "上", "也", "很", "到", "说", "要", "去", "你",
            "会", "着", "没有", "看", "好", "自己", "这", "他", "她", "它",
            "为", "与", "及", "等", "被", "把", "从", "但", "而", "对",
        }

        # 英文停用词
        english_stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "shall", "can",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "about", "it", "its", "this", "that", "and",
            "or", "not", "but", "if", "so", "no", "than", "then",
        }

        keywords = set()

        # 提取中文词（简单按字符拆分，2字以上组合）
        chinese_chars = re.findall(r"[\u4e00-\u9fff]+", text)
        for word in chinese_chars:
            if word not in chinese_stopwords and len(word) >= 2:
                keywords.add(word)

        # 提取英文词
        english_words = re.findall(r"[a-zA-Z]+", text)
        for word in english_words:
            word_lower = word.lower()
            if word_lower not in english_stopwords and len(word_lower) >= 2:
                keywords.add(word_lower)

        # 提取数字
        numbers = re.findall(r"\d+[\d,.]*\d*|\d+", text)
        for num in numbers:
            keywords.add(num)

        return keywords

    def _check_contradiction(self, claim: str, fact: str) -> bool:
        """
        检查声明是否与事实矛盾

        简单实现：检查是否包含否定词且关键词与事实匹配

        Args:
            claim: 声明
            fact: 知识库事实

        Returns:
            bool: 是否矛盾
        """
        # 否定词列表
        negation_words = [
            "不是", "并非", "不是", "错误", "不对", "没有",
            "不正确", "不", "非", "无", "未",
            "is not", "are not", "was not", "were not",
            "isn't", "aren't", "wasn't", "weren't",
            "not", "no", "never", "none", "nothing",
            "incorrect", "wrong", "false",
        ]

        claim_lower = claim.lower()

        # 检查声明中是否包含否定词
        has_negation = any(neg in claim_lower for neg in negation_words)

        if has_negation:
            # 提取关键词检查是否与事实相关
            claim_keywords = self._extract_keywords(claim)
            fact_keywords = self._extract_keywords(fact)
            overlap = len(claim_keywords & fact_keywords)
            # 如果有否定词且与事实有关键词重叠，可能是矛盾的
            return overlap >= 2

        # 检查数字是否一致
        claim_numbers = re.findall(r"\d+[\d,.]*\d*|\d+", claim)
        fact_numbers = re.findall(r"\d+[\d,.]*\d*|\d+", fact)
        if claim_numbers and fact_numbers:
            # 简单比较：如果事实中有数字而声明中也有数字但不一致
            for cn in claim_numbers:
                cn_clean = cn.replace(",", "").replace(".", "")
                for fn in fact_numbers:
                    fn_clean = fn.replace(",", "").replace(".", "")
                    if cn_clean == fn_clean:
                        return False  # 数字一致，不矛盾
            # 数字不一致且关键词有重叠
            claim_keywords = self._extract_keywords(claim)
            fact_keywords = self._extract_keywords(fact)
            if len(claim_keywords & fact_keywords) >= 3:
                return True

        return False
