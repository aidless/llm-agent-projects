"""
分词器工具模块
封装 HuggingFace Tokenizer，提供统一的分词接口
"""
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class TokenizerWrapper:
    """
    分词器封装类

    封装 HuggingFace transformers 的 PreTrainedTokenizer，
    提供统一的分词和编码接口。
    """
    tokenizer: Any = None  # HuggingFace PreTrainedTokenizer 实例
    max_length: int = 2048  # 最大序列长度
    truncation: bool = True  # 是否截断
    padding: str = "max_length"  # 填充策略: "max_length" 或 "longest"

    def load_tokenizer(self, model_name_or_path: str) -> None:
        """
        加载预训练分词器

        Args:
            model_name_or_path: 模型名称或本地路径
        """
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
            use_fast=True,
        )

        # 如果没有 pad_token，使用 eos_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info("未找到 pad_token，已设置 pad_token = eos_token")

        logger.info(
            f"分词器加载完成: vocab_size={self.tokenizer.vocab_size}, "
            f"pad_token={self.tokenizer.pad_token}"
        )

    def encode_text(
        self,
        text: str,
        max_length: Optional[int] = None,
    ) -> Dict[str, List[int]]:
        """
        对单条文本进行编码

        Args:
            text: 输入文本
            max_length: 覆盖默认最大长度

        Returns:
            包含 input_ids 和 attention_mask 的字典
        """
        if self.tokenizer is None:
            raise RuntimeError("分词器未加载，请先调用 load_tokenizer()")

        effective_max_length = max_length or self.max_length

        result = self.tokenizer(
            text,
            max_length=effective_max_length,
            truncation=self.truncation,
            padding=self.padding if self.padding != "longest" else False,
            return_tensors=None,
        )

        return {
            "input_ids": result["input_ids"],
            "attention_mask": result["attention_mask"],
        }

    def encode_batch(
        self,
        texts: List[str],
        max_length: Optional[int] = None,
    ) -> Dict[str, List[List[int]]]:
        """
        批量编码文本

        Args:
            texts: 文本列表
            max_length: 覆盖默认最大长度

        Returns:
            包含 input_ids 和 attention_mask 的字典
        """
        if self.tokenizer is None:
            raise RuntimeError("分词器未加载，请先调用 load_tokenizer()")

        effective_max_length = max_length or self.max_length

        results = self.tokenizer(
            texts,
            max_length=effective_max_length,
            truncation=self.truncation,
            padding=self.padding,
            return_tensors=None,
        )

        return {
            "input_ids": results["input_ids"],
            "attention_mask": results["attention_mask"],
        }

    def decode(self, token_ids: List[int], skip_special_tokens: bool = True) -> str:
        """
        将 token IDs 解码为文本

        Args:
            token_ids: token ID 列表
            skip_special_tokens: 是否跳过特殊 token

        Returns:
            解码后的文本
        """
        if self.tokenizer is None:
            raise RuntimeError("分词器未加载，请先调用 load_tokenizer()")

        return self.tokenizer.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def get_token_count(self, text: str) -> int:
        """
        获取文本的 token 数量

        Args:
            text: 输入文本

        Returns:
            token 数量
        """
        if self.tokenizer is None:
            raise RuntimeError("分词器未加载，请先调用 load_tokenizer()")

        return len(self.tokenizer.encode(text, add_special_tokens=True))
