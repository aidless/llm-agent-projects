"""
任务评估器
提供自定义任务的评估指标：准确率、F1 分数等
"""
import re
from typing import List, Dict, Any, Optional, Callable
from collections import Counter
import logging

logger = logging.getLogger(__name__)


class TaskEvaluator:
    """
    任务评估器

    支持多种评估指标的生成式任务评估。
    通过对比模型生成文本和参考答案来计算准确率、F1 等指标。

    使用示例:
        evaluator = TaskEvaluator()
        predictions = ["北京", "上海", "广州"]
        references = ["北京", "上海", "深圳"]
        results = evaluator.evaluate(predictions, references)
        print(f"准确率: {results['accuracy']}")
    """

    def __init__(
        self,
        metrics: Optional[List[str]] = None,
        exact_match_fn: Optional[Callable] = None,
    ):
        """
        初始化评估器

        Args:
            metrics: 要计算的指标列表，支持: "accuracy", "f1", "precision", "recall"
            exact_match_fn: 自定义精确匹配函数，接收 (prediction, reference) 返回 bool
        """
        self.metrics = metrics or ["accuracy", "f1"]
        self.exact_match_fn = exact_match_fn

    def evaluate(
        self,
        predictions: List[str],
        references: List[str],
    ) -> Dict[str, float]:
        """
        评估预测结果

        Args:
            predictions: 模型预测的文本列表
            references: 参考答案文本列表

        Returns:
            包含各项评估指标的字典
        """
        if len(predictions) != len(references):
            raise ValueError(
                f"预测数量 ({len(predictions)}) 和参考数量 ({len(references)}) 不匹配"
            )

        if not predictions:
            return {metric: 0.0 for metric in self.metrics}

        results = {}

        for metric in self.metrics:
            if metric == "accuracy":
                results["accuracy"] = self._compute_accuracy(predictions, references)
            elif metric == "f1":
                precision, recall, f1 = self._compute_f1(predictions, references)
                results["precision"] = precision
                results["recall"] = recall
                results["f1"] = f1
            else:
                logger.warning(f"未知的评估指标: {metric}")

        logger.info(f"评估结果: {results}")
        return results

    def _compute_accuracy(
        self,
        predictions: List[str],
        references: List[str],
    ) -> float:
        """
        计算精确匹配准确率

        支持自定义匹配函数或默认的字符串匹配（忽略大小写和首尾空格）
        """
        correct = 0
        for pred, ref in zip(predictions, references):
            pred_clean = pred.strip()
            ref_clean = ref.strip()

            if self.exact_match_fn is not None:
                if self.exact_match_fn(pred, ref):
                    correct += 1
            else:
                # 默认匹配：忽略大小写
                if pred_clean.lower() == ref_clean.lower():
                    correct += 1

        return correct / len(predictions)

    def _compute_f1(
        self,
        predictions: List[str],
        references: List[str],
    ) -> tuple:
        """
        计算 Token 级别的 F1 分数

        Returns:
            (precision, recall, f1) 元组
        """
        total_precision = 0.0
        total_recall = 0.0
        total_f1 = 0.0

        for pred, ref in zip(predictions, references):
            pred_tokens = self._tokenize(pred)
            ref_tokens = self._tokenize(ref)

            common = Counter(pred_tokens) & Counter(ref_tokens)
            num_common = sum(common.values())

            if num_common == 0:
                continue

            precision = num_common / len(pred_tokens) if pred_tokens else 0
            recall = num_common / len(ref_tokens) if ref_tokens else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            total_precision += precision
            total_recall += recall
            total_f1 += f1

        n = len(predictions)
        return (
            total_precision / n,
            total_recall / n,
            total_f1 / n,
        )

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """
        简单的中文+英文分词

        将文本拆分为单词和中文字符的混合 token 列表
        """
        # 移除标点并转小写
        text = re.sub(r'[^\w\u4e00-\u9fff]', ' ', text.lower())
        text = text.strip()

        tokens = []
        for word in text.split():
            # 英文单词作为单个 token
            if word.isascii():
                tokens.append(word)
            else:
                # 中文字符逐个拆分
                for char in word:
                    if char.strip():
                        tokens.append(char)

        return tokens

    def evaluate_generation(
        self,
        prompts: List[str],
        predictions: List[str],
        references: List[str],
        extract_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        评估生成任务（含从生成文本中提取答案）

        Args:
            prompts: 输入提示列表
            predictions: 模型生成的原始文本列表
            references: 参考答案列表
            extract_fn: 从生成文本中提取答案的函数

        Returns:
            包含评估指标和详细结果的字典
        """
        if extract_fn is not None:
            extracted = [extract_fn(pred) for pred in predictions]
        else:
            extracted = predictions

        # 计算指标
        metrics = self.evaluate(extracted, references)

        # 生成详细结果
        details = []
        for i, (prompt, pred, ref, ext) in enumerate(zip(prompts, predictions, references, extracted)):
            details.append({
                "index": i,
                "prompt": prompt[:100],
                "prediction": pred[:200],
                "extracted": ext,
                "reference": ref,
                "match": ext.strip().lower() == ref.strip().lower(),
            })

        return {
            "metrics": metrics,
            "details": details,
            "total_samples": len(predictions),
            "correct_samples": sum(
                1 for d in details if d["match"]
            ),
        }
