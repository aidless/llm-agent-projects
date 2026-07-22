"""
Perplexity 评估器
计算模型在给定文本上的困惑度
"""
import math
from typing import List, Dict, Any, Optional
import logging

import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


class PerplexityEvaluator:
    """
    困惑度（Perplexity）评估器

    通过计算模型在给定数据集上的交叉熵损失来评估困惑度。
    PPL = exp(avg_loss)

    使用示例:
        evaluator = PerplexityEvaluator(model, tokenizer)
        ppl = evaluator.evaluate(dataset)
        print(f"Perplexity: {ppl:.2f}")
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        max_length: int = 2048,
        stride: int = 512,
        batch_size: int = 8,
        device: Optional[str] = None,
    ):
        """
        初始化评估器

        Args:
            model: 语言模型（可以是 PEFT 模型）
            tokenizer: 分词器
            max_length: 最大序列长度
            stride: 滑动窗口步长（用于长文本分块评估）
            batch_size: 评估 batch size
            device: 计算设备，自动检测
        """
        self.model = model
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.batch_size = batch_size

        # 设置设备
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.model.eval()

    def evaluate(self, texts: List[str]) -> Dict[str, float]:
        """
        计算给定文本列表的困惑度

        Args:
            texts: 待评估的文本列表

        Returns:
            包含 perplexity、avg_loss 等指标的字典
        """
        total_loss = 0.0
        total_tokens = 0
        num_batches = 0

        with torch.no_grad():
            for text in texts:
                # 编码文本
                encodings = self.tokenizer(
                    text,
                    return_tensors="pt",
                    truncation=True,
                    max_length=self.max_length,
                )
                input_ids = encodings.input_ids.to(self.device)
                attention_mask = encodings.attention_mask.to(self.device)

                # 使用滑动窗口处理长文本
                seq_len = input_ids.size(1)
                if seq_len <= self.max_length:
                    # 短文本直接评估
                    loss, nlls = self._compute_loss(input_ids, attention_mask)
                    total_loss += loss
                    total_tokens += nlls
                else:
                    # 长文本使用滑动窗口
                    loss, nlls = self._compute_sliding_window_loss(
                        input_ids, attention_mask
                    )
                    total_loss += loss
                    total_tokens += nlls

                num_batches += 1

        # 计算最终指标
        avg_loss = total_loss / total_tokens if total_tokens > 0 else float("inf")
        perplexity = math.exp(avg_loss) if avg_loss < 100 else float("inf")

        results = {
            "perplexity": perplexity,
            "avg_loss": avg_loss,
            "total_tokens": total_tokens,
            "num_texts": len(texts),
        }

        logger.info(
            f"Perplexity 评估完成: PPL={perplexity:.2f}, "
            f"avg_loss={avg_loss:.4f}, tokens={total_tokens}"
        )

        return results

    def _compute_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple:
        """
        计算单个序列的 loss

        Returns:
            (total_neg_log_likelihood, num_tokens) 元组
        """
        # 将目标设置为输入的平移版本（因果语言模型）
        target_ids = input_ids.clone()

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=target_ids,
        )

        loss = outputs.loss.item()
        # 计算 token 数量（排除 padding）
        num_tokens = attention_mask.sum().item() - 1  # 减去第一个 token（没有前文）

        return loss * num_tokens, num_tokens

    def _compute_sliding_window_loss(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple:
        """
        使用滑动窗口计算长文本的 loss

        Returns:
            (total_neg_log_likelihood, num_tokens) 元组
        """
        total_nll = 0.0
        total_tokens = 0
        seq_len = input_ids.size(1)

        # 第一个窗口
        prev_end = 0
        for start in range(0, seq_len, self.stride):
            end = min(start + self.max_length, seq_len)
            chunk_ids = input_ids[:, start:end]
            chunk_mask = attention_mask[:, start:end]

            if chunk_ids.size(1) < 2:
                continue

            # 确保有上下文（除了第一个窗口）
            if start > 0:
                context_start = max(0, start - self.stride)
                context_ids = input_ids[:, context_start:start]
                chunk_ids = torch.cat([context_ids, chunk_ids], dim=1)
                context_mask = attention_mask[:, context_start:start]
                chunk_mask = torch.cat([context_mask, chunk_mask], dim=1)
                target_ids = chunk_ids.clone()
                # 只计算新 chunk 部分的 loss
                target_ids[:, :-chunk_ids.size(1) + (end - start)] = -100

            outputs = self.model(
                input_ids=chunk_ids,
                attention_mask=chunk_mask,
            )

            # 手动计算 loss
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = input_ids[:, start:end][..., 1:].contiguous().to(self.device)

            loss_fct = torch.nn.CrossEntropyLoss(reduction="sum")
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            total_nll += loss.item()
            total_tokens += shift_labels.numel()

            prev_end = end

            if end >= seq_len:
                break

        return total_nll, total_tokens
