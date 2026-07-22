"""
推理接口模块
提供加载微调后模型并进行推理的简单 API
"""
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

import torch

logger = logging.getLogger(__name__)


@dataclass
class GenerationConfig:
    """
    生成配置

    控制文本生成的各项参数。
    """
    max_new_tokens: int = 512  # 最大生成 token 数
    temperature: float = 0.7  # 温度（越低越确定性）
    top_p: float = 0.9  # nucleus sampling 的概率阈值
    top_k: int = 50  # top-k sampling 的 k 值
    repetition_penalty: float = 1.0  # 重复惩罚
    do_sample: bool = True  # 是否使用采样（False 则为贪心解码）
    num_beams: int = 1  # beam search 的 beam 数量
    length_penalty: float = 1.0  # beam search 的长度惩罚


class InferenceEngine:
    """
    推理引擎

    加载微调后的模型（支持 LoRA 适配器和合并后的模型），
    提供便捷的文本生成接口。

    使用示例:
        engine = InferenceEngine(model_path="./output/merged_model")
        result = engine.generate("请介绍一下深度学习")
        print(result)

        # 批量推理
        results = engine.generate_batch(["问题1", "问题2"])
    """

    def __init__(
        self,
        model_path: str,
        adapter_path: Optional[str] = None,
        device: Optional[str] = None,
        torch_dtype: str = "auto",
    ):
        """
        初始化推理引擎

        Args:
            model_path: 基础模型路径或合并后的模型路径
            adapter_path: LoRA 适配器路径（可选，如果提供则加载适配器）
            device: 推理设备
            torch_dtype: 模型权重数据类型
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_dtype = torch_dtype

        logger.info(f"正在加载模型: {model_path}")

        # 设置数据类型
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(torch_dtype) if torch_dtype != "auto" else "auto"

        # 加载模型
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            device_map="auto" if self.device == "cuda" else None,
            trust_remote_code=True,
        )

        # 加载 LoRA 适配器（可选）
        if adapter_path is not None:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, adapter_path)
            logger.info(f"已加载 LoRA 适配器: {adapter_path}")

        # 加载分词器
        tokenizer_path = adapter_path or model_path
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path,
            trust_remote_code=True,
            use_fast=True,
        )

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model.eval()
        logger.info("模型加载完成，已设置为评估模式")

    def generate(
        self,
        prompt: str,
        generation_config: Optional[GenerationConfig] = None,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> str:
        """
        单条文本生成

        Args:
            prompt: 输入提示文本
            generation_config: 生成配置（可选）
            max_new_tokens: 覆盖配置中的 max_new_tokens
            temperature: 覆盖配置中的 temperature
            **kwargs: 传递给 model.generate() 的其他参数

        Returns:
            生成的文本
        """
        gen_config = generation_config or GenerationConfig()

        # 构建 generate 参数
        gen_kwargs = {
            "max_new_tokens": max_new_tokens or gen_config.max_new_tokens,
            "temperature": temperature or gen_config.temperature,
            "top_p": gen_config.top_p,
            "top_k": gen_config.top_k,
            "repetition_penalty": gen_config.repetition_penalty,
            "do_sample": gen_config.do_sample,
            "num_beams": gen_config.num_beams,
            "length_penalty": gen_config.length_penalty,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        gen_kwargs.update(kwargs)

        # 编码输入
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        )

        if self.device != "auto":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # 生成
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **gen_kwargs,
            )

        # 解码（仅取生成的部分）
        generated_ids = outputs[0][inputs["input_ids"].size(1):]
        response = self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        return response.strip()

    def generate_batch(
        self,
        prompts: List[str],
        generation_config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> List[str]:
        """
        批量文本生成

        Args:
            prompts: 输入提示文本列表
            generation_config: 生成配置
            **kwargs: 传递给 generate() 的其他参数

        Returns:
            生成的文本列表
        """
        results = []
        for prompt in prompts:
            try:
                result = self.generate(prompt, generation_config, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"生成失败: {e}")
                results.append(f"[生成失败: {e}]")

        return results

    def chat(
        self,
        messages: List[Dict[str, str]],
        generation_config: Optional[GenerationConfig] = None,
        **kwargs,
    ) -> str:
        """
        对话模式推理

        Args:
            messages: 消息列表，每条消息包含 "role" 和 "content"
                例如: [{"role": "user", "content": "你好"}]
            generation_config: 生成配置
            **kwargs: 其他生成参数

        Returns:
            模型的回复文本
        """
        # 构建 chat template 格式
        if hasattr(self.tokenizer, "apply_chat_template"):
            prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            # 简单回退：将消息拼接
            parts = []
            for msg in messages:
                parts.append(f"{msg['role']}: {msg['content']}")
            prompt = "\n".join(parts) + "\nassistant:"

        return self.generate(prompt, generation_config, **kwargs)
