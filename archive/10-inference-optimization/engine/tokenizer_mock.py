"""模拟分词器 - 不依赖真实模型，模拟分词行为"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class TokenizerConfig:
    """分词器配置"""
    vocab_size: int = 32000
    max_seq_length: int = 4096
    bos_token_id: int = 1
    eos_token_id: int = 2
    pad_token_id: int = 0
    name: str = "mock_tokenizer"


@dataclass
class EncodeResult:
    """编码结果"""
    token_ids: List[int]
    num_tokens: int
    encode_latency_ms: float


class MockTokenizer:
    """模拟分词器，通过确定性哈希将文本映射为 token ids"""

    def __init__(self, config: Optional[TokenizerConfig] = None):
        self.config = config or TokenizerConfig()
        self._vocab_size = self.config.vocab_size
        self._id_to_token: Dict[int, str] = {}
        self._token_to_id: Dict[str, int] = {}
        self._build_vocab()

    def _build_vocab(self) -> None:
        """构建模拟词表"""
        # 常用 token
        common_tokens = [
            "<unk>", "<s>", "</s>", "<pad>", "the", " a ", " is ", " in ",
            " to ", " and ", " of ", " for ", " that ", " with ", " this ",
            " from ", " are ", " was ", " it ", " at ", " be ", " have ",
            " as ", " not ", " but ", " by ", " on ", " or ", " an ",
            " so ", " if ", " we ", " they ", " he ", " she ", " you ",
        ]
        for idx, token in enumerate(common_tokens):
            if idx < self._vocab_size:
                self._id_to_token[idx] = token
                self._token_to_id[token] = idx

    def encode(self, text: str) -> EncodeResult:
        """将文本编码为 token ids"""
        start = time.perf_counter()
        token_ids = [self.config.bos_token_id]

        if text:
            # 将文本分块并哈希为 token id
            chunk_size = 3
            for i in range(0, len(text), chunk_size):
                chunk = text[i:i + chunk_size]
                token_id = self._hash_to_token(chunk)
                token_ids.append(token_id)

        token_ids.append(self.config.eos_token_id)

        # 截断到最大长度
        token_ids = token_ids[:self.config.max_seq_length]

        elapsed = (time.perf_counter() - start) * 1000
        return EncodeResult(
            token_ids=token_ids,
            num_tokens=len(token_ids),
            encode_latency_ms=elapsed,
        )

    def decode(self, token_ids: List[int]) -> str:
        """将 token ids 解码为文本"""
        tokens = []
        for tid in token_ids:
            if tid == self.config.bos_token_id:
                tokens.append("<s>")
            elif tid == self.config.eos_token_id:
                tokens.append("</s>")
            elif tid == self.config.pad_token_id:
                continue
            elif tid in self._id_to_token:
                tokens.append(self._id_to_token[tid])
            else:
                tokens.append(f"<tok_{tid}>")
        return " ".join(tokens)

    def _hash_to_token(self, text: str) -> int:
        """将文本通过哈希映射到 token id"""
        h = hashlib.md5(text.encode()).hexdigest()
        return int(h, 16) % self._vocab_size

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    @property
    def pad_token_id(self) -> int:
        return self.config.pad_token_id

    @property
    def eos_token_id(self) -> int:
        return self.config.eos_token_id
