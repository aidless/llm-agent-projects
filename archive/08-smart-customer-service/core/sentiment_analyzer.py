"""情感分析模块 - 规则 + LLM 双重方案"""

from __future__ import annotations

import re
from typing import Optional

from app.models import SentimentLabel, SentimentResult


# 正面词库
_POSITIVE_WORDS: list[str] = [
    "好", "不错", "满意", "喜欢", "棒", "优秀", "感谢", "谢谢", "赞",
    "nice", "good", "great", "excellent", "thanks", "thank you", "love", "happy",
    "perfect", "amazing", "wonderful", "awesome", "best", "helpful", "recommended",
    "好评", "推荐", "很赞", "到位", "专业", "靠谱",
]

# 负面词库
_NEGATIVE_WORDS: list[str] = [
    "差", "烂", "坏", "垃圾", "骗", "不满", "失望", "讨厌", "恶心",
    "bad", "terrible", "horrible", "awful", "worst", "angry", "upset", "disappointed",
    "poor", "useless", "broken", "defective", "fraud", "scam",
    "太差了", "什么玩意", "再也不", "后悔", "坑", "忽悠",
]

# 愤怒/极端负面词库
_ANGER_WORDS: list[str] = [
    "投诉", "举报", "投诉你们", "消协", "12315", "曝光",
    "你们这些骗子", "去死", "滚", "笨", "蠢",
    "complain", "report", "lawsuit", "sue", "fraud", "scam",
    "我要投诉", "我要举报", "你们太不像话了", "什么破东西",
    "能不能做事", "废物", "垃圾公司",
]

# 否定词
_NEGATION_WORDS: list[str] = ["不", "没", "没有", "无", "别", "未", "not", "no", "never", "don't", "doesn't"]

# 强化词
_INTENSIFIERS: list[str] = ["很", "非常", "特别", "太", "极", "超级", "really", "very", "extremely", "so"]


class SentimentAnalyzer:
    """情感分析器 - 规则 + LLM 双方案"""

    def __init__(self) -> None:
        self._llm_func: Optional[callable] = None
        self._history: list[SentimentResult] = []

    def set_llm_analyzer(self, func: callable) -> None:
        """设置可选的 LLM 情感分析回调"""
        self._llm_func = func

    def analyze(self, text: str) -> SentimentResult:
        """分析文本情感"""
        text_lower = text.lower().strip()

        # 1. 规则分析
        rule_result = self._rule_based_analyze(text_lower)

        # 2. 如果有 LLM 回调，综合 LLM 结果
        if self._llm_func:
            try:
                llm_result = self._llm_func(text)
                if llm_result is not None:
                    # 加权融合
                    final_label = llm_result.label
                    final_score = 0.6 * llm_result.score + 0.4 * rule_result.score
                    rule_result = SentimentResult(
                        label=final_label,
                        score=min(max(final_score, 0.0), 1.0),
                        is_angry=rule_result.is_angry or llm_result.is_angry,
                        anger_score=max(rule_result.anger_score, llm_result.anger_score),
                    )
            except Exception:
                pass  # LLM 失败回退到规则结果

        self._history.append(rule_result)
        return rule_result

    def _rule_based_analyze(self, text: str) -> SentimentResult:
        """基于规则的情感分析"""
        # 检测愤怒
        anger_count = sum(1 for w in _ANGER_WORDS if w in text)
        is_angry = anger_count >= 1
        anger_score = min(anger_count * 0.4, 1.0)

        # 计算正负面得分
        pos_score = 0
        neg_score = 0

        has_negation = any(nw in text for nw in _NEGATION_WORDS)
        has_intensifier = any(iw in text for iw in _INTENSIFIERS)

        for word in _POSITIVE_WORDS:
            if word in text:
                w = 1.0
                if has_intensifier:
                    w *= 1.5
                if has_negation:
                    neg_score += w
                else:
                    pos_score += w

        for word in _NEGATIVE_WORDS:
            if word in text:
                w = 1.0
                if has_intensifier:
                    w *= 1.5
                if has_negation:
                    pos_score += w * 0.5  # 否定负面 = 偏正面但不完全翻转
                else:
                    neg_score += w

        # 愤怒词额外增加负面分
        neg_score += anger_count * 2.0

        total = pos_score + neg_score
        if total == 0:
            return SentimentResult(
                label=SentimentLabel.NEUTRAL,
                score=0.5,
                is_angry=is_angry,
                anger_score=anger_score,
            )

        neg_ratio = neg_score / total

        if neg_ratio > 0.6:
            label = SentimentLabel.NEGATIVE
            score = neg_ratio
        elif neg_ratio < 0.3:
            label = SentimentLabel.POSITIVE
            score = 1.0 - neg_ratio
        else:
            label = SentimentLabel.NEUTRAL
            score = 0.5

        return SentimentResult(
            label=label,
            score=score,
            is_angry=is_angry,
            anger_score=anger_score,
        )

    def get_sentiment_trend(self, last_n: int = 10) -> str:
        """获取情感趋势"""
        if not self._history:
            return "stable"
        recent = self._history[-last_n:]
        scores = [self._label_to_score(r.label) for r in recent]

        if len(scores) < 2:
            return "stable"

        # 简单线性趋势
        first_half = sum(scores[: len(scores) // 2]) / max(len(scores[: len(scores) // 2]), 1)
        second_half = sum(scores[len(scores) // 2 :]) / max(len(scores[len(scores) // 2 :]), 1)

        diff = second_half - first_half
        if diff > 0.1:
            return "improving"
        elif diff < -0.1:
            return "declining"
        return "stable"

    @staticmethod
    def _label_to_score(label: SentimentLabel) -> float:
        mapping = {
            SentimentLabel.POSITIVE: 1.0,
            SentimentLabel.NEUTRAL: 0.5,
            SentimentLabel.NEGATIVE: 0.0,
        }
        return mapping.get(label, 0.5)

    def get_history(self) -> list[SentimentResult]:
        """获取情感历史"""
        return self._history.copy()

    def reset_history(self) -> None:
        """重置历史"""
        self._history.clear()
