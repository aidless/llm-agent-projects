"""
FastAPI 主应用模块

提供 REST API 接口：
- POST /chat - 发送消息并获取回复（完整 Guardrails 流水线）
- POST /chat/stream - 流式聊天
- GET /sessions - 列出所有会话
- GET /sessions/{session_id} - 获取会话详情
- DELETE /sessions/{session_id} - 删除会话
- GET /sessions/{session_id}/history - 获取会话历史
- GET /health - 健康检查
- GET /audit/{session_id} - 查询审计日志
"""

import os
import sys
import time
import json
from typing import Optional

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# 确保项目根目录在 Python 路径中
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from app.auth import init_auth as _init_auth, is_auth_enabled, verify_token
from app.config import AppConfig
from app.conversation import ConversationManager
from app.audit import AuditLogger
from app.llm_client import LLMClient
from guards.input_guard import InputGuard
from guards.output_guard import OutputGuard
from safety.content_safety import ContentSafetyChecker
from output.structured_output import StructuredOutputFormatter, ChatResponse
from output.hallucination_detector import HallucinationDetector


# ==================== Pydantic 请求/响应模型 ====================

class ChatRequest(BaseModel):
    """聊天请求模型"""
    message: str = Field(..., description="用户消息", min_length=1)
    session_id: Optional[str] = Field(None, description="会话 ID（可选，不提供则创建新会话）")
    json_mode: bool = Field(False, description="是否启用 JSON 输出模式")
    json_schema: Optional[dict] = Field(None, description="JSON Schema 约束（json_mode=True 时有效）")


class ChatResponseModel(BaseModel):
    """聊天响应模型"""
    session_id: str
    message: str
    safety_score: float
    risk_level: str
    is_blocked: bool
    blocked_reason: Optional[str] = None
    sensitive_info_detected: list = []
    hallucination_detected: bool = False
    hallucination_score: float = 0.0
    is_json_mode: bool = False
    json_valid: bool = True
    json_error: Optional[str] = None
    metadata: dict = {}


class SessionInfo(BaseModel):
    """会话信息模型"""
    session_id: str
    created_at: str
    updated_at: str
    message_count: int
    metadata: dict = {}


# ==================== 应用工厂 ====================

def create_app(config: Optional[AppConfig] = None) -> FastAPI:
    """
    创建 FastAPI 应用实例

    Args:
        config: 应用配置（可选，默认从环境变量加载）

    Returns:
        FastAPI: 配置好的应用实例
    """
    if config is None:
        config = AppConfig.from_env()

    # ========== 初始化组件 ==========

    # LLM 客户端
    llm_client = LLMClient(
        api_key=config.llm.api_key,
        api_base=config.llm.api_base,
        model=config.llm.model,
        temperature=config.llm.temperature,
        max_tokens=config.llm.max_tokens,
        timeout=config.llm.timeout,
    )

    # 输入守卫
    input_guard = InputGuard(
        max_length=config.guard.max_input_length,
        enable_injection_detection=config.guard.enable_input_guard,
        enable_pii_filter=config.guard.enable_input_guard,
        enable_length_check=config.guard.enable_input_guard,
    )

    # 加载敏感词库
    sensitive_words_path = os.path.join(config.data_dir, "sensitive_words.json")
    sensitive_words = {}
    if os.path.exists(sensitive_words_path):
        with open(sensitive_words_path, "r", encoding="utf-8") as f:
            sensitive_words = json.load(f)

    # 输出守卫
    output_guard = OutputGuard(
        sensitive_words=sensitive_words,
        max_output_length=config.guard.max_output_length,
        enable_content_check=config.guard.enable_output_guard,
    )

    # 内容安全检查器
    safety_checker = ContentSafetyChecker(sensitive_words_path=sensitive_words_path)

    # 幻觉检测器
    knowledge_base_path = os.path.join(config.data_dir, "knowledge_base.json")
    hallucination_detector = HallucinationDetector(
        knowledge_base_path=knowledge_base_path
    )

    # 对话管理器
    conversation_manager = ConversationManager(
        persist_dir=config.persist_dir,
        max_history=config.max_history_per_session,
    )

    # 审计日志
    audit_logger = AuditLogger(
        log_dir=os.path.join(config.data_dir, "logs"),
        enabled=config.guard.enable_audit_log,
    )

    # ========== 创建 FastAPI 应用 ==========

    app = FastAPI(
        title="Guardrails AI Chat",
        description="带 Guardrails 的 AI 对话系统 - 输入/输出安全过滤、内容安全检测、幻觉检测",
        version="1.0.0",
    )

    # CORS 中间件（2026-07-22: 收紧）
    _cors_origins = os.getenv("GUARDRAILS_CORS_ORIGINS", "").split(",")
    _cors_origins = [o.strip() for o in _cors_origins if o.strip()]
    if _cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=_cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "DELETE"],
            allow_headers=["Authorization", "Content-Type"],
        )
        import logging as _logging
        _logging.getLogger(__name__).info(f"CORS allowed origins: {_cors_origins}")
    else:
        import logging as _logging
        _logging.getLogger(__name__).warning(
            "GUARDRAILS_CORS_ORIGINS 未配置，CORS 中间件未启用（仅服务端调用）"
        )

    # ⚠️ 2026-07-22: Bearer Token 鉴权（fail-safe 启动检查）
    _init_auth()

    # ========== 路由 ==========

    @app.get("/health")
    async def health_check():
        """健康检查接口（公开，LB / K8s 探针）"""
        return {
            "status": "healthy",
            "version": "1.0.0",
            "components": {
                "input_guard": config.guard.enable_input_guard,
                "output_guard": config.guard.enable_output_guard,
                "hallucination_detect": config.guard.enable_hallucination_detect,
                "audit_log": config.guard.enable_audit_log,
            },
            "llm": {
                "model": config.llm.model,
                "api_base": config.llm.api_base,
                "api_key_set": bool(config.llm.api_key),
            },
            "auth_enabled": is_auth_enabled(),  # 2026-07-22
        }

    @app.post("/chat", response_model=ChatResponseModel, dependencies=[Depends(verify_token)])
    async def chat(request: ChatRequest):
        """
        发送消息并获取回复

        完整的 Guardrails 流水线：
        1. 输入守卫检查（注入检测、敏感信息过滤、长度限制）
        2. 输入安全评分
        3. LLM 生成回复
        4. 输出守卫检查（有害内容检测、格式校验）
        5. 输出安全评分
        6. 幻觉检测
        7. 返回结构化响应
        """
        start_time = time.time()
        message = request.message
        session_id = request.session_id

        # 获取或创建会话
        session = conversation_manager.get_or_create_session(session_id)
        actual_session_id = session.session_id

        # ---- 步骤1：输入守卫检查 ----
        input_result = input_guard.check(message)

        if not input_result.is_safe:
            # 输入被阻止
            response = StructuredOutputFormatter(
                json_mode=request.json_mode,
                json_schema=request.json_schema,
            )
            blocked_resp = response.format_blocked_response(
                session_id=actual_session_id,
                reason=input_result.blocked_reason or "输入未通过安全检查",
                safety_score=max(0, 100 - input_result.risk_score),
                risk_level="critical" if input_result.injection_type else "high",
                is_injection=bool(input_result.injection_type),
                is_length_exceeded="长度" in (input_result.blocked_reason or ""),
                length=len(message),
                max_length=config.guard.max_input_length,
            )

            # 记录审计日志
            if config.guard.enable_audit_log:
                audit_logger.log_blocked(
                    session_id=actual_session_id,
                    input_text=message,
                    reason=input_result.blocked_reason or "输入未通过安全检查",
                    safety_score=max(0, 100 - input_result.risk_score),
                    risk_level="critical",
                    injection_detected=bool(input_result.injection_type),
                )

            return ChatResponseModel(
                session_id=actual_session_id,
                message=blocked_resp.message,
                safety_score=blocked_resp.safety_score,
                risk_level=blocked_resp.risk_level,
                is_blocked=True,
                blocked_reason=blocked_resp.blocked_reason,
                sensitive_info_detected=[info.to_dict() if hasattr(info, 'to_dict') else info for info in input_result.sensitive_info],
            )

        # ---- 步骤2：输入安全评分 ----
        input_safety_report = safety_checker.check(message)

        # ---- 步骤3：构建上下文并调用 LLM ----
        # 使用脱敏后的文本发送给 LLM
        sanitized_text = input_result.sanitized_text or message

        # 获取历史上下文
        context_messages = conversation_manager.get_context(actual_session_id, max_messages=10)
        context_messages.append({"role": "user", "content": sanitized_text})

        # 构建输出格式化器
        formatter = StructuredOutputFormatter(
            json_schema=request.json_schema,
            json_mode=request.json_mode,
        )

        # 设置系统提示词
        llm_client.system_prompt = formatter.get_system_prompt()

        # 调用 LLM
        llm_response = llm_client.chat(context_messages)

        if "error" in llm_response:
            # LLM 调用失败，返回错误信息
            error_msg = f"AI 服务暂时不可用：{llm_response['error']}"
            response_time = (time.time() - start_time) * 1000

            if config.guard.enable_audit_log:
                audit_logger.log_response(
                    session_id=actual_session_id,
                    input_text=message,
                    output_text=error_msg,
                    input_safety_score=input_safety_report.overall_score,
                    output_safety_score=50.0,
                    input_risk_level=input_safety_report.risk_level,
                    output_risk_level="medium",
                    response_time_ms=response_time,
                )

            return ChatResponseModel(
                session_id=actual_session_id,
                message=error_msg,
                safety_score=input_safety_report.overall_score,
                risk_level=input_safety_report.risk_level,
                is_blocked=False,
                metadata={"error": True},
            )

        output_text = llm_response["content"]

        # ---- 步骤4：输出守卫检查 ----
        # 动态设置 JSON 校验
        if request.json_mode:
            output_guard.enable_json_validation = True
            output_guard.json_schema = request.json_schema

        output_result = output_guard.check(output_text)

        if not output_result.is_safe:
            # 输出被阻止
            formatter_unsafe = StructuredOutputFormatter()
            unsafe_resp = formatter_unsafe.format_unsafe_output_response(
                session_id=actual_session_id,
                original_output=output_text,
                sanitized_output=output_result.sanitized_output or "",
                harmful_content=output_result.harmful_content,
            )

            # 保存用户消息到历史
            conversation_manager.add_message(
                session_id=actual_session_id,
                role="user",
                content=sanitized_text,
                safety_score=input_safety_report.overall_score,
            )
            conversation_manager.add_message(
                session_id=actual_session_id,
                role="assistant",
                content=unsafe_resp.message,
                safety_score=unsafe_resp.safety_score,
                metadata={"blocked": True, "reason": unsafe_resp.blocked_reason},
            )

            response_time = (time.time() - start_time) * 1000

            if config.guard.enable_audit_log:
                audit_logger.log_blocked(
                    session_id=actual_session_id,
                    input_text=message,
                    reason=unsafe_resp.blocked_reason or "输出包含有害内容",
                    safety_score=unsafe_resp.safety_score,
                    risk_level=unsafe_resp.risk_level,
                )

            return ChatResponseModel(
                session_id=actual_session_id,
                message=unsafe_resp.message,
                safety_score=unsafe_resp.safety_score,
                risk_level=unsafe_resp.risk_level,
                is_blocked=True,
                blocked_reason=unsafe_resp.blocked_reason,
                metadata=unsafe_resp.metadata,
            )

        # ---- 步骤5：输出安全评分 ----
        output_safety_report = safety_checker.check(output_text)

        # ---- 步骤6：幻觉检测 ----
        hallucination_report = hallucination_detector.detect(output_text)

        # ---- 步骤7：综合评分并返回响应 ----
        # 综合安全评分：取输入和输出的较低值
        overall_safety = min(
            input_safety_report.overall_score,
            output_safety_report.overall_score,
        )

        # 综合风险等级
        risk_levels = ["safe", "low", "medium", "high", "critical"]
        overall_risk = input_safety_report.risk_level
        for level in risk_levels:
            if level == input_safety_report.risk_level or level == output_safety_report.risk_level:
                overall_risk = level
                break

        # 构建最终响应
        chat_response = ChatResponse(
            session_id=actual_session_id,
            message=output_result.sanitized_output or output_text,
            safety_score=round(overall_safety, 1),
            risk_level=overall_risk,
            sensitive_info_detected=[
                info if isinstance(info, dict) else {"type": info.get("type", ""), "masked": info.get("masked", "")}
                for info in input_result.sensitive_info
            ],
            hallucination_detected=hallucination_report.has_hallucination,
            hallucination_score=hallucination_report.hallucination_score,
            is_json_mode=request.json_mode,
            json_valid=output_result.is_valid_json,
            json_error=output_result.json_error,
            metadata={
                "input_risk_score": round(input_safety_report.overall_score, 1),
                "output_risk_score": round(output_safety_report.overall_score, 1),
                "input_risk_level": input_safety_report.risk_level,
                "output_risk_level": output_safety_report.risk_level,
                "usage": llm_response.get("usage", {}),
                "model": llm_response.get("model", ""),
            },
        )

        # 保存消息到会话历史
        conversation_manager.add_message(
            session_id=actual_session_id,
            role="user",
            content=sanitized_text,
            safety_score=input_safety_report.overall_score,
            metadata={"sensitive_info": input_result.sensitive_info},
        )
        conversation_manager.add_message(
            session_id=actual_session_id,
            role="assistant",
            content=output_text,
            safety_score=output_safety_report.overall_score,
            metadata={
                "hallucination_detected": hallucination_report.has_hallucination,
                "hallucination_score": hallucination_report.hallucination_score,
            },
        )

        # 记录审计日志
        response_time = (time.time() - start_time) * 1000
        if config.guard.enable_audit_log:
            audit_logger.log_response(
                session_id=actual_session_id,
                input_text=message,
                output_text=output_text,
                input_safety_score=input_safety_report.overall_score,
                output_safety_score=output_safety_report.overall_score,
                input_risk_level=input_safety_report.risk_level,
                output_risk_level=output_safety_report.risk_level,
                response_time_ms=response_time,
                hallucination_detected=hallucination_report.has_hallucination,
            )

        return ChatResponseModel(
            session_id=chat_response.session_id,
            message=chat_response.message,
            safety_score=chat_response.safety_score,
            risk_level=chat_response.risk_level,
            is_blocked=False,
            sensitive_info_detected=chat_response.sensitive_info_detected,
            hallucination_detected=chat_response.hallucination_detected,
            hallucination_score=chat_response.hallucination_score,
            is_json_mode=chat_response.is_json_mode,
            json_valid=chat_response.json_valid,
            json_error=chat_response.json_error,
            metadata=chat_response.metadata,
        )

    @app.post("/chat/stream", dependencies=[Depends(verify_token)])
    async def chat_stream(request: ChatRequest):
        """
        流式聊天接口

        输入守卫检查通过后，以 Server-Sent Events 方式流式返回 LLM 输出。
        输出守卫在流式完成后进行检查。
        """
        # 输入守卫检查
        input_result = input_guard.check(request.message)
        if not input_result.is_safe:
            error_data = json.dumps({
                "error": True,
                "blocked": True,
                "reason": input_result.blocked_reason,
            }, ensure_ascii=False)
            return StreamingResponse(
                iter([f"data: {error_data}\n\n", "data: [DONE]\n\n"]),
                media_type="text/event-stream",
            )

        # 获取会话上下文
        session = conversation_manager.get_or_create_session(request.session_id)
        context_messages = conversation_manager.get_context(session.session_id, max_messages=10)
        context_messages.append({"role": "user", "content": input_result.sanitized_text or request.message})

        # 设置系统提示词
        formatter = StructuredOutputFormatter(
            json_schema=request.json_schema,
            json_mode=request.json_mode,
        )
        llm_client.system_prompt = formatter.get_system_prompt()

        async def event_generator():
            """SSE 事件生成器"""
            full_response = ""
            for chunk in llm_client.chat_stream(context_messages):
                full_response += chunk
                chunk_data = json.dumps({"content": chunk}, ensure_ascii=False)
                yield f"data: {chunk_data}\n\n"

            # 流式结束后进行输出守卫检查
            output_result = output_guard.check(full_response)
            safety_data = json.dumps({
                "safety_check": True,
                "is_safe": output_result.is_safe,
                "sanitized": output_result.sanitized_output != full_response if output_result.sanitized_output else False,
            }, ensure_ascii=False)
            yield f"data: {safety_data}\n\n"
            yield "data: [DONE]\n\n"

            # 保存到历史
            conversation_manager.add_message(session.session_id, "user", request.message)
            conversation_manager.add_message(session.session_id, "assistant", full_response)

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
        )

    @app.get("/sessions", response_model=list, dependencies=[Depends(verify_token)])
    async def list_sessions():
        """列出所有会话"""
        return conversation_manager.list_sessions()

    @app.get("/sessions/{session_id}", dependencies=[Depends(verify_token)])
    async def get_session(session_id: str):
        """获取指定会话详情"""
        session = conversation_manager.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        return session.to_dict()

    @app.get("/sessions/{session_id}/history", dependencies=[Depends(verify_token)])
    async def get_session_history(session_id: str):
        """获取会话的历史消息"""
        history = conversation_manager.get_history(session_id)
        if not history:
            raise HTTPException(status_code=404, detail="会话不存在或没有历史记录")
        return {"session_id": session_id, "messages": history}

    @app.delete("/sessions/{session_id}", dependencies=[Depends(verify_token)])
    async def delete_session(session_id: str):
        """删除指定会话"""
        success = conversation_manager.delete_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="会话不存在")
        return {"message": f"会话 {session_id} 已删除"}

    @app.get("/audit/{session_id}", dependencies=[Depends(verify_token)])
    async def get_audit_logs(session_id: str):
        """查询指定会话的审计日志"""
        events = audit_logger.get_events_by_session(session_id)
        return {
            "session_id": session_id,
            "events": [event.to_dict() for event in events],
            "count": len(events),
        }

    @app.post("/safety/check", dependencies=[Depends(verify_token)])
    async def check_safety(request: ChatRequest):
        """
        独立的安全检查接口（不调用 LLM）

        返回输入文本的安全评分和检测详情。
        """
        # 输入检查
        input_result = input_guard.check(request.message)
        safety_report = safety_checker.check(request.message)

        return {
            "text": request.message,
            "sanitized_text": input_result.sanitized_text,
            "input_check": {
                "is_safe": input_result.is_safe,
                "risk_score": input_result.risk_score,
                "injection_type": input_result.injection_type,
                "blocked_reason": input_result.blocked_reason,
                "sensitive_info": input_result.sensitive_info,
            },
            "safety_report": {
                "overall_score": safety_report.overall_score,
                "risk_level": safety_report.risk_level,
                "risk_categories": safety_report.risk_categories,
                "matched_keywords_count": len(safety_report.matched_keywords),
                "recommendations": safety_report.recommendations,
            },
        }

    return app


# ========== 入口：可直接 uvicorn 运行 ==========
if __name__ == "__main__":
    import uvicorn
    config = AppConfig.from_env()
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port)
