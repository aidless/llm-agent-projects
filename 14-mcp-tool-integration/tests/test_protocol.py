"""JSON-RPC 2.0 协议测试"""

import json
import pytest

from protocol.jsonrpc import (
    JSONRPCError,
    JSONRPCNotification,
    JSONRPCRequest,
    JSONRPCResponse,
    create_batch_request,
    create_batch_response,
    create_error_response,
    create_notification,
    create_request,
    create_response,
    encode_batch,
    encode_message,
    parse_message,
    INVALID_PARAMS,
    METHOD_NOT_FOUND,
)


class TestJSONRPCRequest:
    """JSON-RPC 请求测试"""

    def test_create_request_with_params(self):
        req = create_request("tools/list", {"cursor": "abc"}, req_id="test-1")
        assert req.method == "tools/list"
        assert req.params == {"cursor": "abc"}
        assert req.id == "test-1"

    def test_create_request_auto_id(self):
        req = create_request("ping")
        assert req.method == "ping"
        assert req.id != ""
        assert len(req.id) == 8

    def test_request_to_dict(self):
        req = create_request("test", {"key": "val"}, req_id="123")
        d = req.to_dict()
        assert d["jsonrpc"] == "2.0"
        assert d["method"] == "test"
        assert d["params"] == {"key": "val"}
        assert d["id"] == "123"

    def test_request_is_notification(self):
        req = create_request("test", req_id="123")
        assert not req.is_notification()

    def test_notification_no_id(self):
        notif = create_notification("test", {"key": "val"})
        d = notif.to_dict()
        assert "id" not in d
        assert d["method"] == "test"
        assert d["params"] == {"key": "val"}


class TestJSONRPCResponse:
    """JSON-RPC 响应测试"""

    def test_success_response(self):
        resp = create_response({"tools": []}, req_id="r1")
        assert resp.id == "r1"
        assert resp.result == {"tools": []}
        assert resp.error is None

    def test_error_response(self):
        resp = create_error_response(-32601, "Method not found", req_id="r2")
        assert resp.error is not None
        assert resp.error.code == -32601
        assert resp.error.message == "Method not found"

    def test_response_to_dict_success(self):
        resp = create_response("ok", req_id="1")
        d = resp.to_dict()
        assert d["result"] == "ok"
        assert "error" not in d

    def test_response_to_dict_error(self):
        resp = create_error_response(-32602, "Bad params", req_id="2")
        d = resp.to_dict()
        assert "result" not in d
        assert d["error"]["code"] == -32602

    def test_error_from_dict(self):
        err = JSONRPCError.from_dict({"code": -32700, "message": "Parse error", "data": "extra"})
        assert err.code == -32700
        assert err.message == "Parse error"
        assert err.data == "extra"


class TestJSONRPCBatch:
    """JSON-RPC 批量消息测试"""

    def test_batch_request(self):
        reqs = [
            create_request("tools/list", req_id="1"),
            create_notification("ping"),
        ]
        batch = create_batch_request(reqs)
        assert len(batch) == 2
        assert batch[0]["id"] == "1"
        assert "id" not in batch[1]

    def test_batch_response(self):
        resps = [
            create_response("ok", req_id="1"),
            create_error_response(-32601, "Not found", req_id="2"),
        ]
        batch = create_batch_response(resps)
        assert len(batch) == 2


class TestParseMessage:
    """消息解析测试"""

    def test_parse_request_string(self):
        raw = json.dumps({"jsonrpc": "2.0", "method": "ping", "params": {}, "id": "p1"})
        msg = parse_message(raw)
        assert isinstance(msg, JSONRPCRequest)
        assert msg.method == "ping"

    def test_parse_notification(self):
        msg = parse_message({"jsonrpc": "2.0", "method": "log", "params": {"msg": "hi"}})
        assert isinstance(msg, JSONRPCNotification)
        assert msg.method == "log"

    def test_parse_response(self):
        msg = parse_message({"jsonrpc": "2.0", "id": "r1", "result": {}})
        assert isinstance(msg, JSONRPCResponse)
        assert msg.result == {}

    def test_parse_error_response(self):
        msg = parse_message({"jsonrpc": "2.0", "id": "r2", "error": {"code": -32601, "message": "Not found"}})
        assert isinstance(msg, JSONRPCResponse)
        assert msg.error.code == -32601

    def test_parse_batch(self):
        batch = [
            {"jsonrpc": "2.0", "method": "ping", "id": "1"},
            {"jsonrpc": "2.0", "id": "2", "result": "ok"},
        ]
        msgs = parse_message(batch)
        assert len(msgs) == 2
        assert isinstance(msgs[0], JSONRPCRequest)
        assert isinstance(msgs[1], JSONRPCResponse)

    def test_parse_invalid_json(self):
        with pytest.raises(ValueError, match="Invalid JSON"):
            parse_message("not json")

    def test_parse_invalid_version(self):
        with pytest.raises(ValueError, match="jsonrpc"):
            parse_message({"jsonrpc": "1.0", "method": "test"})

    def test_parse_empty_batch(self):
        with pytest.raises(ValueError, match="Empty batch"):
            parse_message([])


class TestEncodeMessage:
    """消息编码测试"""

    def test_encode_request(self):
        req = create_request("test", req_id="e1")
        encoded = encode_message(req)
        parsed = json.loads(encoded)
        assert parsed["method"] == "test"

    def test_encode_batch(self):
        msgs = [
            create_request("a", req_id="1"),
            create_response("ok", req_id="2"),
        ]
        encoded = encode_batch(msgs)
        parsed = json.loads(encoded)
        assert len(parsed) == 2