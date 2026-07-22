"""Prompt 优化器。"""

import re
from typing import Any, Dict, List, Optional


class PromptOptimizer:
    """基于评估结果的 Prompt 优化器。"""

    # 常见 Prompt 问题模式
    WEAK_PATTERNS = [
        (r"\bplease\b", "避免使用 'please'，直接给出指令更有效"),
        (r"\bcan you\b", "避免使用 'can you'，直接要求执行任务"),
        (r"\?+$", "以问号结尾可能产生开放性回答，考虑更明确的指令"),
        (r"^(hi|hello|hey)", "避免使用问候语开头，直接进入主题"),
        (r"\.{3,}", "避免使用省略号，使用明确的语言"),
    ]

    def analyze_prompt(self, prompt: str) -> Dict[str, Any]:
        """分析 Prompt 并提供改进建议。

        Args:
            prompt: 待分析的 Prompt

        Returns:
            分析结果和改进建议
        """
        issues = []
        suggestions = []

        # 检查长度
        word_count = len(prompt.split())
        if word_count < 10:
            issues.append("Prompt 过短，可能缺乏足够的上下文信息")
            suggestions.append("增加更多上下文描述和具体要求")
        elif word_count > 500:
            issues.append("Prompt 过长，可能影响模型注意力")
            suggestions.append("精简 Prompt，聚焦核心指令")

        # 检查是否有明确的输出格式要求
        if not re.search(r"(格式|format|输出|output|返回|return|json|list|步骤|step)", prompt, re.IGNORECASE):
            issues.append("未指定输出格式")
            suggestions.append("添加明确的输出格式要求，如 '以 JSON 格式输出' 或 '分步骤回答'")

        # 检查是否有示例
        if not re.search(r"(例如|比如|example|for instance|如：|如下)", prompt, re.IGNORECASE):
            suggestions.append("考虑添加 few-shot 示例以提高输出质量")

        # 检查弱模式
        for pattern, suggestion in self.WEAK_PATTERNS:
            if re.search(pattern, prompt, re.IGNORECASE):
                issues.append(f"检测到弱表达模式: '{pattern}'")
                suggestions.append(suggestion)

        # 检查角色设定
        has_role = bool(re.search(r"(你是|你是一个|act as|you are|as a)", prompt, re.IGNORECASE))
        if not has_role:
            suggestions.append("考虑添加角色设定，如 '你是一个专业的...'")

        # 评估清晰度
        clarity_score = self._compute_clarity(prompt)

        return {
            "prompt": prompt,
            "word_count": word_count,
            "char_count": len(prompt),
            "has_role_setting": has_role,
            "issues": issues,
            "suggestions": suggestions,
            "clarity_score": clarity_score,
            "improvement_priority": "high" if len(issues) >= 3 else "medium" if len(issues) >= 1 else "low",
        }

    def select_few_shot_examples(
        self,
        dataset: List[Dict[str, str]],
        input_key: str = "input",
        output_key: str = "output",
        query: str = "",
        n: int = 3,
    ) -> List[Dict[str, str]]:
        """自动选择 few-shot 示例。

        基于简单词汇重叠度选择最相关的示例。

        Args:
            dataset: 数据集
            input_key: 输入字段名
            output_key: 输出字段名
            query: 当前查询
            n: 选择示例数量

        Returns:
            选中的示例列表
        """
        if not query or not dataset:
            return dataset[:n] if len(dataset) >= n else dataset

        query_tokens = set(query.lower().split())
        scored = []

        for item in dataset:
            input_text = item.get(input_key, "")
            item_tokens = set(input_text.lower().split())
            if not item_tokens or not query_tokens:
                overlap = 0.0
            else:
                overlap = len(query_tokens & item_tokens) / len(query_tokens | item_tokens)
            scored.append((overlap, item))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [item for _, item in scored[:n]]

    def parameterize_template(self, template: str) -> Dict[str, Any]:
        """将 Prompt 模板变量化。

        识别 {variable} 格式的变量并提取。

        Args:
            template: Prompt 模板

        Returns:
            变量化结果
        """
        variables = re.findall(r"\{(\w+)\}", template)

        return {
            "template": template,
            "variables": variables,
            "num_variables": len(variables),
            "parameterized": True if variables else False,
            "render_example": template.format(**{v: f"<{v}>" for v in variables}) if variables else template,
        }

    def _compute_clarity(self, prompt: str) -> float:
        """计算 Prompt 清晰度分数 (0-1)。"""
        score = 0.5  # 基础分

        # 有明确指令 +0.1
        if re.search(r"(请|please|你需要|you should|must|确保|make sure)", prompt, re.IGNORECASE):
            score += 0.1

        # 有输出格式 +0.1
        if re.search(r"(格式|format|输出|output|步骤|step)", prompt, re.IGNORECASE):
            score += 0.1

        # 有角色设定 +0.1
        if re.search(r"(你是|you are|as a|act as)", prompt, re.IGNORECASE):
            score += 0.1

        # 长度适中 +0.1
        words = len(prompt.split())
        if 20 <= words <= 300:
            score += 0.1

        # 没有弱模式 +0.1
        has_weak = any(re.search(p, prompt, re.IGNORECASE) for p, _ in self.WEAK_PATTERNS)
        if not has_weak:
            score += 0.1

        return round(min(score, 1.0), 2)