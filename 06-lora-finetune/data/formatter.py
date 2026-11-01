"""
数据格式化模块
将原始数据转换为 SFT 训练所需的格式（instruction + input + output → prompt）
"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


# 预定义的提示模板
TEMPLATES = {
    "alpaca": (
        "以下是一个描述任务的指令，以及一个提供上下文的输入。"
        "请编写一个能恰当完成该任务的回答。\n\n"
        "### 指令:\n{instruction}\n\n"
        "### 输入:\n{input}\n\n"
        "### 回答:\n{output}"
    ),
    "alpaca_no_input": (
        "以下是一个描述任务的指令。"
        "请编写一个能恰当完成该任务的回答。\n\n"
        "### 指令:\n{instruction}\n\n"
        "### 回答:\n{output}"
    ),
    "chatml": (
        "<|im_start|>system\n你是一个有帮助的助手。<|im_end|>\n"
        "<|im_start|>user\n{instruction}{input_part}<|im_end|>\n"
        "<|im_start|>assistant\n{output}<|im_end|>"
    ),
    "simple": "{instruction}\n{output}",
    "qwen": (
        "<|im_start|>system\n你是一个有帮助的助手。<|im_end|>\n"
        "<|im_start|>user\n{instruction}{input_part}<|im_end|>\n"
        "<|im_start|>assistant\n{output}<|im_end|>"
    ),
}


@dataclass
class InstructionFormatter:
    """
    指令数据格式化器

    将原始的 instruction/input/output 数据格式化为模型可接受的 prompt 格式。
    支持多种预定义模板和自定义模板。
    """
    template_name: str = "alpaca"  # 模板名称
    custom_template: Optional[str] = None  # 自定义模板（优先级最高）
    system_prompt: Optional[str] = None  # 系统提示词
    max_input_length: int = 1024  # 输入最大长度
    max_output_length: int = 512  # 输出最大长度

    def __post_init__(self):
        """初始化后验证模板是否存在"""
        if self.custom_template is None and self.template_name not in TEMPLATES:
            raise ValueError(
                f"未知的模板名称: '{self.template_name}'。"
                f"可用模板: {list(TEMPLATES.keys())}"
            )

    def format_single(self, item: Dict[str, Any]) -> str:
        """
        格式化单条数据

        Args:
            item: 包含 instruction/output 以及可选 input 的数据字典

        Returns:
            格式化后的完整 prompt 字符串
        """
        instruction = item.get("instruction", "")
        output = item.get("output", "")
        input_text = item.get("input", "")

        # 截断过长输入/输出
        if len(instruction) > self.max_input_length:
            instruction = instruction[:self.max_input_length]
        if len(output) > self.max_output_length:
            output = output[:self.max_output_length]

        # 使用自定义模板或预定义模板
        if self.custom_template is not None:
            template = self.custom_template
        else:
            template = TEMPLATES[self.template_name]

        # 根据模板类型处理 input 部分
        if self.template_name in ("alpaca",):
            if input_text:
                formatted = template.format(
                    instruction=instruction,
                    input=input_text,
                    output=output,
                )
            else:
                formatted = TEMPLATES["alpaca_no_input"].format(
                    instruction=instruction,
                    output=output,
                )
        elif self.template_name in ("chatml", "qwen"):
            input_part = f"\n{input_text}" if input_text else ""
            formatted = template.format(
                instruction=instruction,
                input_part=input_part,
                output=output,
            )
        else:
            formatted = template.format(
                instruction=instruction,
                input=input_text,
                output=output,
            )

        return formatted

    def format_batch(self, items: List[Dict[str, Any]]) -> List[str]:
        """
        批量格式化数据

        Args:
            items: 数据字典列表

        Returns:
            格式化后的 prompt 字符串列表
        """
        formatted = []
        for item in items:
            try:
                formatted.append(self.format_single(item))
            except Exception as e:
                logger.warning(f"格式化数据失败: {e}, 跳过该条数据: {item.get('instruction', '')[:50]}")
                continue
        logger.info(f"批量格式化完成: {len(formatted)}/{len(items)} 条成功")
        return formatted

    def get_prompt_only(self, item: Dict[str, Any]) -> str:
        """
        仅获取 prompt 部分（不包含 output），用于推理

        Args:
            item: 包含 instruction 以及可选 input 的数据字典

        Returns:
            仅包含 prompt 的字符串
        """
        instruction = item.get("instruction", "")
        input_text = item.get("input", "")

        if self.custom_template is not None:
            template = self.custom_template
        else:
            template = TEMPLATES[self.template_name]

        if self.template_name == "alpaca":
            if input_text:
                prompt = template.format(
                    instruction=instruction,
                    input=input_text,
                    output="",  # 推理时不需要 output
                ).rsplit("### 回答:\n", 1)[0] + "### 回答:\n"
            else:
                prompt = TEMPLATES["alpaca_no_input"].format(
                    instruction=instruction,
                    output="",
                ).rsplit("### 回答:\n", 1)[0] + "### 回答:\n"
        elif self.template_name in ("chatml", "qwen"):
            input_part = f"\n{input_text}" if input_text else ""
            prompt = template.format(
                instruction=instruction,
                input_part=input_part,
                output="",
            ).rsplit("<|im_start|>assistant\n", 1)[0] + "<|im_start|>assistant\n"
        else:
            prompt = f"{instruction}\n"

        return prompt
