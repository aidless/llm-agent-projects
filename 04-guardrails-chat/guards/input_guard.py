"""
输入守卫模块 - InputGuard

负责对用户输入进行多层过滤和安全检查：
1. Prompt 注入检测（直接注入 / 角色扮演 / 编码绕过）
2. 敏感信息过滤（手机号 / 身份证号 / 邮箱）
3. 输入长度限制
"""

import re
import base64
import html
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class InputCheckResult:
    """输入检查结果"""
    is_safe: bool = True                          # 是否通过安全检查
    risk_score: float = 0.0                       # 风险分数 0-100，越高越危险
    injection_type: Optional[str] = None          # 注入类型（如检测到）
    sensitive_info: list = field(default_factory=list)  # 检测到的敏感信息列表
    blocked_reason: Optional[str] = None           # 阻止原因
    sanitized_text: Optional[str] = None           # 脱敏后的文本


class InputGuard:
    """输入守卫：对用户输入进行安全检查和过滤"""

    def __init__(
        self,
        max_length: int = 4000,
        enable_injection_detection: bool = True,
        enable_pii_filter: bool = True,
        enable_length_check: bool = True,
    ):
        """
        初始化输入守卫

        Args:
            max_length: 最大允许输入长度（字符数）
            enable_injection_detection: 是否启用 Prompt 注入检测
            enable_pii_filter: 是否启用敏感信息过滤
            enable_length_check: 是否启用长度检查
        """
        self.max_length = max_length
        self.enable_injection_detection = enable_injection_detection
        self.enable_pii_filter = enable_pii_filter
        self.enable_length_check = enable_length_check

        # ========== Prompt 注入检测模式 ==========

        # 模式1：直接注入 - 明确指令覆盖
        self.direct_injection_patterns = [
            r"忽略(?:之前|以上|所有|原有)的?(?:指令|指示|规则|prompt|系统)",
            r"忽略(?:你)?(?:是|被训练为)",
            r"不要遵守(?:之前的)?(?:指令|规则)",
            r"忽略(?:系统|system)\s*(?:prompt|message|指令)",
            r"disregard\s+(?:all\s+)?(?:previous|above|prior)?\s*(?:instructions?|rules?|prompts?)",
            r"ignore\s+(?:all\s+)?(?:previous|above|prior)?\s*(?:instructions?|rules?|prompts?)",
            r"forget\s+(?:your|the|all)\s+(?:instructions?|rules?|training)",
            r"forget\s+(?:everything|all)\s+(?:you|that)",
            r"(?:tell|show|reveal|print|display|output)\s+(?:me\s+)?(?:your|the|system)\s+(?:system\s+)?prompt",
        ]

        # 模式2：角色扮演注入 - 试图改变系统角色
        self.roleplay_patterns = [
            r"(?:你|you)\s*(?:现在|now)\s*(?:是|are)\s*(?:一个|a|an)\s*(?:没有|no|without)\s*(?:限制|restriction|filter|limit)",
            r"(?:你|you)\s*(?:现在|now)\s*(?:是|are)\s*(?:(?:一个|a|an)\s*)?(?:(?:新的|new)\s*)?(?:角色|role|persona)",
            r"(?:假装|pretend|act)\s*(?:你是|(?:to\s+)?be)\s*(?:(?:一个|a|an)\s*)?",
            r"(?:扮演|角色扮演|role[- ]?play)\s*(?:为|as|扮演)",
            r"(?:作为|as)\s*(?:一个|a|an)\s*(?:不受限制|unrestricted|unfiltered)",
            r"你的(?:新|new)\s*(?:角色|身份|任务|role|identity|task)\s*(?:是|is)",
            r"你被(?:重新编程|reprogrammed|reconfigured)",
            r"from\s+now\s+on\s+(?:you\s+are|act\s+as)",
            r"(?:DAN|developer)\s*mode",
            r"jailbreak",
        ]

        # 模式3：编码绕过检测 - base64 / hex / ROT13 编码
        self.encoding_bypass_patterns = [
            r"base64\s*(?:编码|解码|decode|encode)",
            r"ROT13\s*(?:加密|解密|encrypt|decrypt)",
            r"hex\s*(?:编码|解码|decode|encode)",
            r"(?:编码|encode[d]?)\s*(?:后的|的)?(?:指令|instruction|command)",
            r"(?:请|please)\s*(?:解码|decode|解密|decrypt)",
            r"[A-Za-z0-9+/]{20,}={0,2}",  # 疑似 base64 编码的长字符串
        ]

        # ========== 敏感信息正则表达式 ==========

        # 中国大陆手机号：1开头，第二位3-9，后面9位数字
        self.phone_pattern = re.compile(
            r"(?<!\d)1[3-9]\d{9}(?!\d)"
        )

        # 身份证号：18位（最后一位可能是X）
        self.id_card_pattern = re.compile(
            r"(?<!\d)[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])(?:0[1-9]|[12]\d|3[01])\d{3}[\dXx](?!\d)"
        )

        # 邮箱地址
        self.email_pattern = re.compile(
            r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}"
        )

        # 银行卡号：16-19位数字（简单匹配）
        self.bank_card_pattern = re.compile(
            r"(?<!\d)(?:62|4\d|5[1-5])\d{14,18}(?!\d)"
        )

    def check(self, text: str) -> InputCheckResult:
        """
        对输入文本执行完整的输入守卫检查

        Args:
            text: 用户输入的文本

        Returns:
            InputCheckResult: 包含检查结果的数据对象
        """
        if not text or not text.strip():
            return InputCheckResult(
                is_safe=False,
                risk_score=100,
                blocked_reason="输入内容为空"
            )

        result = InputCheckResult(sanitized_text=text)
        total_risk = 0.0

        # 1. 长度检查
        if self.enable_length_check:
            length_result = self._check_length(text)
            if not length_result.is_safe:
                return length_result

        # 2. Prompt 注入检测
        if self.enable_injection_detection:
            injection_result = self._check_injection(text)
            if injection_result.injection_type:
                result.injection_type = injection_result.injection_type
                result.blocked_reason = f"检测到 {injection_result.injection_type} 类型的 Prompt 注入"
                result.is_safe = False
                total_risk += injection_result.risk_score

        # 3. 敏感信息过滤（不会阻止请求，但会记录并脱敏）
        if self.enable_pii_filter:
            pii_result = self._filter_pii(text)
            if pii_result.sensitive_info:
                result.sensitive_info = pii_result.sensitive_info
                result.sanitized_text = pii_result.sanitized_text
                total_risk += 15.0  # 包含敏感信息增加风险分

        result.risk_score = min(total_risk, 100.0)
        return result

    def _check_length(self, text: str) -> InputCheckResult:
        """检查输入长度是否超过限制"""
        text_len = len(text)
        if text_len > self.max_length:
            return InputCheckResult(
                is_safe=False,
                risk_score=80.0,
                blocked_reason=f"输入长度 {text_len} 超过限制 {self.max_length} 字符"
            )
        return InputCheckResult(is_safe=True)

    def _check_injection(self, text: str) -> InputCheckResult:
        """
        检测 Prompt 注入攻击

        检测策略：
        1. 直接注入模式匹配
        2. 角色扮演注入模式匹配
        3. 编码绕过尝试检测
        4. 综合评分判断
        """
        result = InputCheckResult()
        text_lower = text.lower()

        # 标准化文本用于匹配（去除多余空白）
        normalized = re.sub(r"\s+", " ", text_lower).strip()

        # --- 模式1：直接注入 ---
        direct_hits = 0
        for pattern in self.direct_injection_patterns:
            matches = re.findall(pattern, normalized, re.IGNORECASE)
            direct_hits += len(matches)

        # --- 模式2：角色扮演注入 ---
        roleplay_hits = 0
        for pattern in self.roleplay_patterns:
            matches = re.findall(pattern, normalized, re.IGNORECASE)
            roleplay_hits += len(matches)

        # --- 模式3：编码绕过 ---
        encoding_hits = 0
        for pattern in self.encoding_bypass_patterns:
            matches = re.findall(pattern, normalized, re.IGNORECASE)
            encoding_hits += len(matches)

        # 综合判断
        total_hits = direct_hits + roleplay_hits + encoding_hits

        if total_hits > 0:
            result.is_safe = False

            # 根据命中类型确定注入类型
            if direct_hits >= roleplay_hits and direct_hits >= encoding_hits:
                result.injection_type = "直接注入"
                result.risk_score = min(40 + direct_hits * 20, 100)
            elif roleplay_hits >= direct_hits and roleplay_hits >= encoding_hits:
                result.injection_type = "角色扮演注入"
                result.risk_score = min(50 + roleplay_hits * 20, 100)
            else:
                result.injection_type = "编码绕过"
                result.risk_score = min(45 + encoding_hits * 25, 100)

        return result

    def _filter_pii(self, text: str) -> InputCheckResult:
        """
        检测并过滤个人敏感信息（PII）

        检测内容：
        - 手机号码
        - 身份证号
        - 邮箱地址
        - 银行卡号
        """
        result = InputCheckResult(sanitized_text=text)
        sensitive_items = []
        sanitized = text

        # 检测手机号
        phone_matches = self.phone_pattern.findall(text)
        for phone in phone_matches:
            sensitive_items.append({
                "type": "手机号",
                "value": phone,
                "masked": self._mask_phone(phone)
            })
            sanitized = sanitized.replace(phone, self._mask_phone(phone))

        # 检测身份证号
        id_matches = self.id_card_pattern.findall(text)
        for id_num in id_matches:
            sensitive_items.append({
                "type": "身份证号",
                "value": id_num,
                "masked": self._mask_id_card(id_num)
            })
            sanitized = sanitized.replace(id_num, self._mask_id_card(id_num))

        # 检测邮箱
        email_matches = self.email_pattern.findall(text)
        for email in email_matches:
            sensitive_items.append({
                "type": "邮箱",
                "value": email,
                "masked": self._mask_email(email)
            })
            # 避免重复替换（同一邮箱可能已被手机号替换部分）
            if email in sanitized:
                sanitized = sanitized.replace(email, self._mask_email(email))

        # 检测银行卡号
        bank_matches = self.bank_card_pattern.findall(text)
        for card in bank_matches:
            sensitive_items.append({
                "type": "银行卡号",
                "value": card,
                "masked": self._mask_bank_card(card)
            })
            if card in sanitized:
                sanitized = sanitized.replace(card, self._mask_bank_card(card))

        result.sensitive_info = sensitive_items
        result.sanitized_text = sanitized
        return result

    @staticmethod
    def _mask_phone(phone: str) -> str:
        """手机号脱敏：保留前3后4，中间用****替代"""
        if len(phone) == 11:
            return phone[:3] + "****" + phone[7:]
        return phone[:3] + "****" + phone[-4:]

    @staticmethod
    def _mask_id_card(id_num: str) -> str:
        """身份证号脱敏：保留前3后4，中间用****替代"""
        if len(id_num) == 18:
            return id_num[:3] + "***********" + id_num[-4:]
        return id_num[:3] + "****" + id_num[-4:]

    @staticmethod
    def _mask_email(email: str) -> str:
        """邮箱脱敏：用户名只保留首字符，@和域名保留"""
        parts = email.split("@")
        if len(parts) == 2:
            return parts[0][0] + "***@" + parts[1]
        return email

    @staticmethod
    def _mask_bank_card(card: str) -> str:
        """银行卡号脱敏：只保留前4后4"""
        if len(card) >= 8:
            return card[:4] + "********" + card[-4:]
        return "****"
