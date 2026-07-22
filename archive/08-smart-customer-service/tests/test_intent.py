"""意图识别测试"""

import pytest
from core.intent_classifier import IntentClassifier
from app.models import IntentCategory, SubIntent


@pytest.fixture
def classifier():
    return IntentClassifier()


class TestIntentClassifier:
    """意图分类器测试"""

    def test_greeting_chinese(self, classifier):
        """测试中文问候识别"""
        result = classifier.classify("你好")
        assert result.intent == IntentCategory.GREETING

    def test_greeting_english(self, classifier):
        """测试英文问候识别"""
        result = classifier.classify("hello")
        assert result.intent == IntentCategory.GREETING

    def test_consultation_price(self, classifier):
        """测试价格咨询识别"""
        result = classifier.classify("这个商品多少钱")
        assert result.intent == IntentCategory.CONSULTATION
        assert result.sub_intent == SubIntent.PRICE_INQUIRY

    def test_consultation_stock(self, classifier):
        """测试库存咨询识别"""
        result = classifier.classify("还有库存吗")
        assert result.intent == IntentCategory.CONSULTATION

    def test_complaint(self, classifier):
        """测试投诉识别"""
        result = classifier.classify("你们的产品太差了")
        assert result.intent == IntentCategory.COMPLAINT

    def test_after_sale_return(self, classifier):
        """测试退货意图识别"""
        result = classifier.classify("我要退货")
        assert result.intent == IntentCategory.AFTER_SALE
        assert result.sub_intent == SubIntent.RETURN_REQUEST

    def test_transfer_human(self, classifier):
        """测试转人工意图识别"""
        result = classifier.classify("转人工")
        assert result.intent == IntentCategory.TRANSFER_HUMAN

    def test_feedback_positive(self, classifier):
        """测试正面反馈识别"""
        result = classifier.classify("你们服务不错")
        assert result.intent == IntentCategory.FEEDBACK

    def test_order_query(self, classifier):
        """测试订单查询识别"""
        result = classifier.classify("我的订单到哪了")
        assert result.intent == IntentCategory.ORDER_QUERY

    def test_logistics(self, classifier):
        """测试物流识别"""
        result = classifier.classify("快递单号多少")
        assert result.intent == IntentCategory.LOGISTICS

    def test_confidence_range(self, classifier):
        """测试置信度范围"""
        result = classifier.classify("我要退货")
        assert 0.0 <= result.confidence <= 1.0

    def test_raw_scores_not_empty(self, classifier):
        """测试原始分数不为空"""
        result = classifier.classify("退货")
        assert len(result.raw_scores) == len(IntentCategory)

    def test_intent_switch_detection(self, classifier):
        """测试意图切换检测"""
        switched = classifier.detect_intent_switch(IntentCategory.GREETING, "我要退货")
        assert switched is True

    def test_intent_no_switch(self, classifier):
        """测试意图未切换"""
        switched = classifier.detect_intent_switch(IntentCategory.AFTER_SALE, "可以换货吗")
        assert switched is False

    def test_llm_fallback(self, classifier):
        """测试 LLM 回调回退"""

        def mock_llm(text):
            return None

        classifier.set_llm_classifier(mock_llm)
        result = classifier.classify("你好")
        assert result.intent == IntentCategory.GREETING
