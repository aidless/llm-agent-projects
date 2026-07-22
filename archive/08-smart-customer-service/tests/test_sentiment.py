"""情感分析测试"""

import pytest
from core.sentiment_analyzer import SentimentAnalyzer
from app.models import SentimentLabel


@pytest.fixture
def analyzer():
    return SentimentAnalyzer()


class TestSentimentAnalyzer:
    """情感分析器测试"""

    def test_positive_chinese(self, analyzer):
        """测试中文正面情感"""
        result = analyzer.analyze("你们的服务很好")
        assert result.label == SentimentLabel.POSITIVE

    def test_positive_english(self, analyzer):
        """测试英文正面情感"""
        result = analyzer.analyze("great service")
        assert result.label == SentimentLabel.POSITIVE

    def test_negative_chinese(self, analyzer):
        """测试中文负面情感"""
        result = analyzer.analyze("你们的产品太差了")
        assert result.label == SentimentLabel.NEGATIVE

    def test_negative_english(self, analyzer):
        """测试英文负面情感"""
        result = analyzer.analyze("terrible quality")
        assert result.label == SentimentLabel.NEGATIVE

    def test_neutral(self, analyzer):
        """测试中性情感"""
        result = analyzer.analyze("我想了解一下退货流程")
        assert result.label == SentimentLabel.NEUTRAL

    def test_anger_detection(self, analyzer):
        """测试愤怒检测"""
        result = analyzer.analyze("我要投诉你们")
        assert result.is_angry is True
        assert result.anger_score > 0

    def test_no_anger(self, analyzer):
        """测试非愤怒"""
        result = analyzer.analyze("你好")
        assert result.is_angry is False

    def test_score_range(self, analyzer):
        """测试分数范围"""
        result = analyzer.analyze("测试")
        assert 0.0 <= result.score <= 1.0

    def test_history_tracking(self, analyzer):
        """测试历史跟踪"""
        analyzer.analyze("好评")
        analyzer.analyze("不错")
        analyzer.analyze("太差了")
        history = analyzer.get_history()
        assert len(history) == 3

    def test_history_reset(self, analyzer):
        """测试历史重置"""
        analyzer.analyze("好评")
        analyzer.reset_history()
        assert len(analyzer.get_history()) == 0

    def test_sentiment_trend_improving(self, analyzer):
        """测试情感趋势上升"""
        analyzer.reset_history()
        analyzer.analyze("太差了")
        analyzer.analyze("太差了")
        analyzer.analyze("不错")
        analyzer.analyze("很好")
        trend = analyzer.get_sentiment_trend()
        assert trend == "improving"

    def test_sentiment_trend_stable(self, analyzer):
        """测试情感趋势稳定"""
        analyzer.reset_history()
        trend = analyzer.get_sentiment_trend()
        assert trend == "stable"

    def test_intensified_negative(self, analyzer):
        """测试强化负面"""
        result = analyzer.analyze("非常不满意")
        assert result.label == SentimentLabel.NEGATIVE

    def test_llm_integration(self, analyzer):
        """测试 LLM 集成"""
        from app.models import SentimentResult

        def mock_llm(text):
            return SentimentResult(label=SentimentLabel.POSITIVE, score=0.9)

        analyzer.set_llm_analyzer(mock_llm)
        result = analyzer.analyze("太差了")
        # LLM 结果应融合
        assert result.label in [SentimentLabel.POSITIVE, SentimentLabel.NEGATIVE]
