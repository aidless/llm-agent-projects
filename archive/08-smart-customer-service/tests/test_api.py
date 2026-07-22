"""API 测试"""

import os
import pytest
from fastapi.testclient import TestClient

# 确保可以导入 app
os.chdir(os.path.join(os.path.dirname(__file__), ".."))

from app.main import app

client = TestClient(app)


class TestHealthAPI:
    """健康检查 API 测试"""

    def test_health_check(self):
        """测试健康检查"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["faq_count"] >= 50

    def test_root(self):
        """测试根路由"""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data


class TestSessionAPI:
    """会话 API 测试"""

    def test_create_session(self):
        """测试创建会话"""
        response = client.post("/api/v1/sessions")
        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data

    def test_get_session_not_found(self):
        """测试获取不存在的会话"""
        response = client.get("/api/v1/sessions/nonexistent")
        assert response.status_code == 404


class TestKnowledgeAPI:
    """知识库 API 测试"""

    def test_get_stats(self):
        """测试获取知识库统计"""
        response = client.get("/api/v1/knowledge/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["total_count"] >= 50

    def test_search_knowledge(self):
        """测试搜索知识库"""
        response = client.post(
            "/api/v1/knowledge/search",
            json={"query": "退货", "top_k": 3},
        )
        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)
        if data:
            assert "answer" in data[0]
            assert "score" in data[0]

    def test_search_knowledge_english(self):
        """测试英文搜索"""
        response = client.post(
            "/api/v1/knowledge/search",
            json={"query": "refund", "top_k": 3},
        )
        assert response.status_code == 200

    def test_add_knowledge(self):
        """测试添加知识"""
        response = client.post(
            "/api/v1/knowledge/add",
            json={
                "question": "测试问题",
                "answer": "测试答案",
                "category": "测试",
                "keywords": ["测试"],
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_delete_knowledge_not_found(self):
        """测试删除不存在的知识"""
        response = client.delete("/api/v1/knowledge/nonexistent_id")
        assert response.status_code == 404

    def test_get_knowledge_item(self):
        """测试获取知识条目"""
        response = client.get("/api/v1/knowledge/item/faq_001")
        assert response.status_code == 200
        data = response.json()
        assert data["id"] == "faq_001"


class TestChatAPI:
    """聊天 API 测试"""

    def test_chat_greeting(self):
        """测试问候聊天"""
        response = client.post(
            "/api/v1/chat",
            json={"message": "你好", "session_id": "test_greeting"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["intent"] == "greeting"
        assert data["reply"] is not None
        assert len(data["reply"]) > 0

    def test_chat_transfer_human(self):
        """测试转人工聊天"""
        response = client.post(
            "/api/v1/chat",
            json={"message": "转人工", "session_id": "test_transfer"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["need_transfer"] is True

    def test_chat_after_sale(self):
        """测试售后聊天"""
        response = client.post(
            "/api/v1/chat",
            json={"message": "我要退货", "session_id": "test_after_sale"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["intent"] in ["after_sale", "consultation"]

    def test_chat_with_sources(self):
        """测试带来源的回复"""
        response = client.post(
            "/api/v1/chat",
            json={"message": "如何退款", "session_id": "test_source"},
        )
        assert response.status_code == 200
        data = response.json()
        # 应该有回复
        assert len(data["reply"]) > 0

    def test_chat_confidence(self):
        """测试置信度字段"""
        response = client.post(
            "/api/v1/chat",
            json={"message": "快递几天能到", "session_id": "test_conf"},
        )
        assert response.status_code == 200
        data = response.json()
        assert 0.0 <= data["confidence"] <= 1.0

    def test_chat_sentiment(self):
        """测试情感字段"""
        response = client.post(
            "/api/v1/chat",
            json={"message": "你们服务太差了", "session_id": "test_sent"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["sentiment"] in ["positive", "negative", "neutral"]


class TestWebSocket:
    """WebSocket 测试"""

    def test_websocket_connect(self):
        """测试 WebSocket 连接"""
        with client.websocket_connect("/ws/test_ws_connect") as ws:
            # 接收连接确认消息
            data = ws.receive_json()
            assert data["type"] == "system_notice"
            assert "连接成功" in data["content"] or "Connected" in data["content"]

    def test_websocket_chat(self):
        """测试 WebSocket 聊天"""
        with client.websocket_connect("/ws/test_ws_chat") as ws:
            # 接收连接确认
            ws.receive_json()
            # 发送消息
            ws.send_json({"type": "chat_message", "content": "你好"})
            # 接收打字状态
            typing = ws.receive_json()
            assert typing["type"] == "typing_indicator"
            # 接收回复
            reply = ws.receive_json()
            assert reply["type"] == "chat_message"
            # 接收停止打字
            stop_typing = ws.receive_json()
            assert stop_typing["type"] == "typing_indicator"

    def test_websocket_heartbeat(self):
        """测试 WebSocket 心跳"""
        with client.websocket_connect("/ws/test_ws_heartbeat") as ws:
            ws.receive_json()  # 连接确认
            ws.send_json({"type": "heartbeat", "content": "ping"})
            response = ws.receive_json()
            assert response["type"] == "heartbeat"

    def test_websocket_disconnect(self):
        """测试 WebSocket 断开"""
        with client.websocket_connect("/ws/test_ws_disconnect") as ws:
            ws.receive_json()  # 连接确认
        # 退出后连接应已断开
