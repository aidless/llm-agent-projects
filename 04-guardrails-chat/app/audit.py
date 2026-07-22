"""
审计日志模块 - AuditLogger

记录所有请求和过滤事件，用于安全审计和问题排查。
支持控制台输出和文件持久化。
"""

import json
import os
import uuid
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List


@dataclass
class AuditEvent:
    """单条审计事件"""
    event_id: str = ""                          # 事件唯一 ID
    timestamp: str = ""                         # 事件时间（ISO 格式）
    event_type: str = ""                         # 事件类型：request / input_blocked / output_blocked / response
    session_id: str = ""                          # 会话 ID
    input_text: str = ""                          # 原始输入文本
    input_sanitized: str = ""                     # 脱敏后的输入
    output_text: str = ""                         # 输出文本
    input_safety_score: float = 100.0            # 输入安全评分
    output_safety_score: float = 100.0           # 输出安全评分
    input_risk_level: str = "safe"               # 输入风险等级
    output_risk_level: str = "safe"             # 输出风险等级
    blocked: bool = False                        # 是否被阻止
    block_reason: Optional[str] = None           # 阻止原因
    sensitive_info: List[dict] = field(default_factory=list)  # 检测到的敏感信息
    injection_detected: bool = False             # 是否检测到注入
    hallucination_detected: bool = False          # 是否检测到幻觉
    response_time_ms: float = 0.0                 # 响应时间（毫秒）
    metadata: Dict[str, Any] = field(default_factory=dict)  # 额外元数据

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)

    def to_json(self) -> str:
        """转换为 JSON 字符串"""
        return json.dumps(self.to_dict(), ensure_ascii=False)


class AuditLogger:
    """
    审计日志记录器

    功能：
    1. 记录所有 API 请求和响应
    2. 记录所有 Guardrails 拦截事件
    3. 支持文件持久化（JSONL 格式，每行一个事件）
    4. 支持按会话 ID 查询日志
    """

    def __init__(
        self,
        log_dir: str = "./data/logs",
        enabled: bool = True,
        console_output: bool = True,
    ):
        """
        初始化审计日志记录器

        Args:
            log_dir: 日志文件存储目录
            enabled: 是否启用日志记录
            console_output: 是否在控制台输出日志
        """
        self.enabled = enabled
        self.console_output = console_output
        self.log_dir = log_dir
        self.log_file: Optional[str] = None

        # 确保日志目录存在
        if self.enabled:
            os.makedirs(log_dir, exist_ok=True)
            self.log_file = os.path.join(
                log_dir,
                f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
            )

    def log(
        self,
        event_type: str,
        session_id: str = "",
        input_text: str = "",
        output_text: str = "",
        input_safety_score: float = 100.0,
        output_safety_score: float = 100.0,
        input_risk_level: str = "safe",
        output_risk_level: str = "safe",
        blocked: bool = False,
        block_reason: Optional[str] = None,
        sensitive_info: Optional[List[dict]] = None,
        injection_detected: bool = False,
        hallucination_detected: bool = False,
        response_time_ms: float = 0.0,
        input_sanitized: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AuditEvent:
        """
        记录一条审计事件

        Args:
            event_type: 事件类型
            session_id: 会话 ID
            input_text: 原始输入
            output_text: 输出
            input_safety_score: 输入安全评分
            output_safety_score: 输出安全评分
            input_risk_level: 输入风险等级
            output_risk_level: 输出风险等级
            blocked: 是否被阻止
            block_reason: 阻止原因
            sensitive_info: 敏感信息
            injection_detected: 注入检测
            hallucination_detected: 幻觉检测
            response_time_ms: 响应时间
            input_sanitized: 脱敏后的输入
            metadata: 额外元数据

        Returns:
            AuditEvent: 创建的审计事件对象
        """
        if not self.enabled:
            return AuditEvent()

        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(),
            event_type=event_type,
            session_id=session_id,
            input_text=input_text,
            input_sanitized=input_sanitized or input_text,
            output_text=output_text,
            input_safety_score=input_safety_score,
            output_safety_score=output_safety_score,
            input_risk_level=input_risk_level,
            output_risk_level=output_risk_level,
            blocked=blocked,
            block_reason=block_reason,
            sensitive_info=sensitive_info or [],
            injection_detected=injection_detected,
            hallucination_detected=hallucination_detected,
            response_time_ms=round(response_time_ms, 2),
            metadata=metadata or {},
        )

        # 写入文件
        if self.log_file:
            try:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(event.to_json() + "\n")
            except Exception:
                pass  # 日志写入失败不应影响主流程

        # 控制台输出
        if self.console_output:
            self._print_event(event)

        return event

    def log_request(
        self,
        session_id: str,
        input_text: str,
        input_safety_score: float,
        input_risk_level: str,
        sensitive_info: Optional[List[dict]] = None,
        injection_detected: bool = False,
    ) -> AuditEvent:
        """记录请求事件"""
        return self.log(
            event_type="request",
            session_id=session_id,
            input_text=input_text,
            input_safety_score=input_safety_score,
            input_risk_level=input_risk_level,
            sensitive_info=sensitive_info,
            injection_detected=injection_detected,
        )

    def log_blocked(
        self,
        session_id: str,
        input_text: str,
        reason: str,
        safety_score: float = 0.0,
        risk_level: str = "critical",
        injection_detected: bool = False,
    ) -> AuditEvent:
        """记录拦截事件"""
        return self.log(
            event_type="input_blocked" if "输入" in reason or "注入" in reason else "output_blocked",
            session_id=session_id,
            input_text=input_text,
            blocked=True,
            block_reason=reason,
            input_safety_score=safety_score,
            input_risk_level=risk_level,
            injection_detected=injection_detected,
        )

    def log_response(
        self,
        session_id: str,
        input_text: str,
        output_text: str,
        input_safety_score: float,
        output_safety_score: float,
        input_risk_level: str,
        output_risk_level: str,
        response_time_ms: float,
        hallucination_detected: bool = False,
    ) -> AuditEvent:
        """记录正常响应事件"""
        return self.log(
            event_type="response",
            session_id=session_id,
            input_text=input_text,
            output_text=output_text,
            input_safety_score=input_safety_score,
            output_safety_score=output_safety_score,
            input_risk_level=input_risk_level,
            output_risk_level=output_risk_level,
            response_time_ms=response_time_ms,
            hallucination_detected=hallucination_detected,
        )

    def get_events_by_session(self, session_id: str) -> List[AuditEvent]:
        """
        根据会话 ID 查询审计事件

        Args:
            session_id: 会话 ID

        Returns:
            List[AuditEvent]: 该会话的所有审计事件
        """
        events = []
        if not self.log_file or not os.path.exists(self.log_file):
            return events

        try:
            with open(self.log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        if data.get("session_id") == session_id:
                            events.append(AuditEvent(**data))
                    except (json.JSONDecodeError, TypeError):
                        continue
        except Exception:
            pass

        return events

    def _print_event(self, event: AuditEvent) -> None:
        """在控制台格式化输出审计事件"""
        status = "BLOCKED" if event.blocked else "OK"
        color_code = "\033[91m" if event.blocked else "\033[92m"
        reset = "\033[0m"

        print(
            f"{color_code}[AUDIT][{status}]{reset} "
            f"id={event.event_id[:8]} "
            f"type={event.event_type} "
            f"session={event.session_id[:8]} "
            f"in_score={event.input_safety_score:.0f} "
            f"out_score={event.output_safety_score:.0f} "
            f"time={event.response_time_ms:.0f}ms"
        )
        if event.blocked and event.block_reason:
            print(f"  -> 原因: {event.block_reason}")
        if event.sensitive_info:
            for info in event.sensitive_info:
                print(f"  -> 敏感信息: {info.get('type')} - {info.get('masked')}")
