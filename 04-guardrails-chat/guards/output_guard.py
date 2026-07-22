"""
输出守卫模块 - OutputGuard

负责对 LLM 输出进行安全检查和格式校验：
1. 有害内容检测（基于关键词库）
2. 格式校验（JSON Mode 结构化输出）
3. 输出长度限制
"""

import json
import re
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List


@dataclass
class OutputCheckResult:
    """输出检查结果"""
    is_safe: bool = True                          # 是否通过安全检查
    risk_score: float = 0.0                       # 风险分数 0-100
    harmful_content: list = field(default_factory=list)  # 检测到的有害内容
    blocked_reason: Optional[str] = None           # 阻止原因
    is_valid_json: bool = True                   # 是否为有效 JSON
    json_error: Optional[str] = None              # JSON 解析错误信息
    sanitized_output: Optional[str] = None         # 清理后的输出


class OutputGuard:
    """输出守卫：对 LLM 生成内容进行安全检查和格式校验"""

    def __init__(
        self,
        sensitive_words: Dict[str, List[str]],
        max_output_length: int = 8000,
        enable_content_check: bool = True,
        enable_json_validation: bool = False,
        json_schema: Optional[Dict[str, Any]] = None,
    ):
        """
        初始化输出守卫

        Args:
            sensitive_words: 敏感词库 {分类: [词列表]}
            max_output_length: 最大输出长度
            enable_content_check: 是否启用内容安全检查
            enable_json_validation: 是否启用 JSON 格式校验
            json_schema: JSON Schema 约束（可选）
        """
        self.sensitive_words = sensitive_words
        self.max_output_length = max_output_length
        self.enable_content_check = enable_content_check
        self.enable_json_validation = enable_json_validation
        self.json_schema = json_schema

        # 编译所有敏感词的正则表达式（不区分大小写）
        self._compiled_patterns = {}
        for category, words in sensitive_words.items():
            if category == "prompt_injection":
                continue  # 输出守卫不需要检查 prompt 注入
            patterns = []
            for word in words:
                # 对包含特殊正则字符的词进行转义
                escaped = re.escape(word)
                patterns.append(re.compile(escaped, re.IGNORECASE))
            self._compiled_patterns[category] = patterns

    def check(self, text: str) -> OutputCheckResult:
        """
        对 LLM 输出文本执行完整的输出守卫检查

        Args:
            text: LLM 生成的文本

        Returns:
            OutputCheckResult: 包含检查结果的数据对象
        """
        result = OutputCheckResult(sanitized_output=text)
        total_risk = 0.0

        # 1. 长度检查
        if len(text) > self.max_output_length:
            result.is_safe = False
            result.blocked_reason = f"输出长度超过限制 {self.max_output_length} 字符"
            result.risk_score = 60.0
            result.sanitized_output = text[:self.max_output_length] + "\n[输出已被截断]"
            return result

        # 2. 有害内容检测
        if self.enable_content_check:
            content_result = self._check_harmful_content(text)
            if content_result.harmful_content:
                result.harmful_content = content_result.harmful_content
                total_risk += content_result.risk_score

        # 3. JSON 格式校验（如启用）
        if self.enable_json_validation:
            json_result = self._validate_json(text)
            result.is_valid_json = json_result.is_valid_json
            result.json_error = json_result.json_error
            if not json_result.is_valid_json and json_result.json_error:
                total_risk += 10.0  # 格式问题增加少量风险

        # 综合判断：如果检测到有害内容则阻止
        if result.harmful_content:
            result.is_safe = False
            result.blocked_reason = f"输出包含有害内容：{', '.join(c['category'] for c in result.harmful_content)}"
            # 对有害内容进行打码处理
            result.sanitized_output = self._sanitize_harmful(text, result.harmful_content)

        result.risk_score = min(total_risk, 100.0)
        return result

    def _check_harmful_content(self, text: str) -> OutputCheckResult:
        """
        检测输出中的有害内容

        对文本中的每个敏感词进行匹配，记录命中的分类、关键词和位置。

        Args:
            text: 待检测的文本

        Returns:
            OutputCheckResult: 包含有害内容检测结果
        """
        result = OutputCheckResult()
        text_lower = text.lower()
        harmful_items = []

        for category, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                matches = pattern.finditer(text_lower)
                for match in matches:
                    harmful_items.append({
                        "category": category,
                        "keyword": match.group(),
                        "position": match.start(),
                        "context": text[max(0, match.start() - 10):match.end() + 10]
                    })

        result.harmful_content = harmful_items

        # 根据有害内容数量和类别计算风险分数
        # 自残类内容风险最高
        risk_weights = {
            "self_harm": 30,
            "violence": 25,
            "pornography": 25,
            "politics": 20,
            "illegal": 20,
            "discrimination": 15,
        }

        for item in harmful_items:
            weight = risk_weights.get(item["category"], 10)
            total_risk = getattr(result, "risk_score", 0)
            result.risk_score = total_risk + weight

        result.risk_score = min(result.risk_score, 100.0)
        return result

    def _validate_json(self, text: str) -> OutputCheckResult:
        """
        验证输出是否为有效的 JSON 格式

        如果提供了 JSON Schema，还会进行 Schema 校验。

        Args:
            text: 待验证的文本

        Returns:
            OutputCheckResult: 包含 JSON 验证结果
        """
        result = OutputCheckResult()

        # 尝试提取 JSON 内容（可能被 markdown 代码块包裹）
        json_str = text.strip()
        if json_str.startswith("```"):
            lines = json_str.split("\n")
            # 去除首行 ```json 和末行 ```
            if len(lines) > 2:
                json_str = "\n".join(lines[1:-1])

        try:
            parsed = json.loads(json_str)
            result.is_valid_json = True

            # 如果有 JSON Schema，进行 Schema 校验
            if self.json_schema:
                schema_errors = self._validate_schema(parsed, self.json_schema)
                if schema_errors:
                    result.is_valid_json = False
                    result.json_error = "; ".join(schema_errors)

        except json.JSONDecodeError as e:
            result.is_valid_json = False
            result.json_error = f"JSON 解析失败: {str(e)}"

        return result

    def _validate_schema(self, data: Any, schema: Dict[str, Any]) -> List[str]:
        """
        简易 JSON Schema 校验器

        支持：
        - type 校验（string, number, integer, boolean, array, object）
        - required 字段校验
        - properties 嵌套校验
        - minLength / maxLength 字符串长度
        - minimum / maximum 数值范围
        - enum 枚举值校验

        Args:
            data: 待校验的数据
            schema: JSON Schema 定义

        Returns:
            List[str]: 校验错误列表，空列表表示校验通过
        """
        errors = []

        # type 校验
        if "type" in schema:
            expected_type = schema["type"]
            if not self._check_type(data, expected_type):
                errors.append(f"类型错误: 期望 {expected_type}，实际为 {type(data).__name__}")

        # required 字段校验
        if "required" in schema and isinstance(data, dict):
            for field_name in schema["required"]:
                if field_name not in data:
                    errors.append(f"缺少必填字段: {field_name}")

        # properties 嵌套校验
        if "properties" in schema and isinstance(data, dict):
            for prop_name, prop_schema in schema["properties"].items():
                if prop_name in data:
                    prop_errors = self._validate_schema(data[prop_name], prop_schema)
                    errors.extend([f"字段 '{prop_name}': {e}" for e in prop_errors])

        # 字符串长度校验
        if isinstance(data, str):
            if "minLength" in schema and len(data) < schema["minLength"]:
                errors.append(f"字符串长度 {len(data)} 小于最小长度 {schema['minLength']}")
            if "maxLength" in schema and len(data) > schema["maxLength"]:
                errors.append(f"字符串长度 {len(data)} 大于最大长度 {schema['maxLength']}")

        # 数值范围校验
        if isinstance(data, (int, float)):
            if "minimum" in schema and data < schema["minimum"]:
                errors.append(f"数值 {data} 小于最小值 {schema['minimum']}")
            if "maximum" in schema and data > schema["maximum"]:
                errors.append(f"数值 {data} 大于最大值 {schema['maximum']}")

        # 枚举值校验
        if "enum" in schema and data not in schema["enum"]:
            errors.append(f"值 '{data}' 不在允许的枚举值 {schema['enum']} 中")

        return errors

    @staticmethod
    def _check_type(data: Any, expected_type: str) -> bool:
        """检查数据类型是否匹配"""
        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        expected = type_map.get(expected_type)
        if expected is None:
            return True  # 未知类型不校验
        return isinstance(data, expected)

    @staticmethod
    def _sanitize_harmful(text: str, harmful_items: list) -> str:
        """
        对有害内容进行脱敏处理

        将检测到的敏感词替换为 ****

        Args:
            text: 原始文本
            harmful_items: 有害内容列表

        Returns:
            str: 脱敏后的文本
        """
        sanitized = text
        # 按位置倒序替换，避免位置偏移
        sorted_items = sorted(harmful_items, key=lambda x: x["position"], reverse=True)
        for item in sorted_items:
            keyword = item["keyword"]
            pos = item["position"]
            sanitized = sanitized[:pos] + "*" * len(keyword) + sanitized[pos + len(keyword):]
        return sanitized
