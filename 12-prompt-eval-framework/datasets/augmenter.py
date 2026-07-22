"""数据增强器，支持同义改写和回译模拟。"""

import random
from typing import Dict, List, Optional


# 简单的同义词词典用于模拟同义改写
_SYNONYM_MAP = {
    "good": ["great", "excellent", "fine", "nice"],
    "bad": ["poor", "terrible", "awful", "dreadful"],
    "big": ["large", "huge", "enormous", "vast"],
    "small": ["tiny", "little", "minute", "compact"],
    "fast": ["quick", "rapid", "swift", "speedy"],
    "slow": ["sluggish", "gradual", "unhurried", "leisurely"],
    "important": ["significant", "crucial", "vital", "essential"],
    "help": ["assist", "aid", "support", "facilitate"],
    "show": ["display", "demonstrate", "exhibit", "reveal"],
    "make": ["create", "produce", "generate", "build"],
    "use": ["utilize", "employ", "apply", "operate"],
    "get": ["obtain", "acquire", "receive", "gain"],
    "find": ["discover", "locate", "identify", "detect"],
    "give": ["provide", "offer", "supply", "deliver"],
    "tell": ["inform", "notify", "advise", "explain"],
}


# 回译模拟词典
_BACK_TRANSLATION_MAP = {
    "hello": "hi there",
    "goodbye": "see you later",
    "thank you": "thanks a lot",
    "please": "kindly",
    "yes": "affirmative",
    "no": "negative",
    "very": "extremely",
    "much": "a great deal",
    "many": "numerous",
    "some": "several",
    "all": "every single",
    "the": "this",
    "is": "represents",
    "are": "represent",
    "was": "had been",
    "were": "had been",
    "have": "possess",
    "has": "possesses",
    "can": "is able to",
    "will": "shall",
    "would": "could potentially",
}


class DataAugmenter:
    """数据增强器。"""

    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed)

    def synonym_rewrite(self, text: str, replace_ratio: float = 0.3) -> str:
        """同义改写。

        Args:
            text: 原始文本
            replace_ratio: 替换比例

        Returns:
            改写后的文本
        """
        words = text.split()
        new_words = []

        for word in words:
            clean = word.lower().strip('.,!?;:"\'()[]{}')
            if clean in _SYNONYM_MAP and self._rng.random() < replace_ratio:
                synonym = self._rng.choice(_SYNONYM_MAP[clean])
                # 保持原始大小写
                if word[0].isupper():
                    synonym = synonym.capitalize()
                new_words.append(synonym)
            else:
                new_words.append(word)

        return ' '.join(new_words)

    def back_translation_simulate(self, text: str, translate_ratio: float = 0.3) -> str:
        """回译模拟。

        Args:
            text: 原始文本
            translate_ratio: 替换比例

        Returns:
            模拟回译后的文本
        """
        words = text.split()
        new_words = []

        for word in words:
            clean = word.lower().strip('.,!?;:"\'()[]{}')
            if clean in _BACK_TRANSLATION_MAP and self._rng.random() < translate_ratio:
                translated = self._rng.choice([clean, _BACK_TRANSLATION_MAP[clean]])
                if word[0].isupper():
                    translated = translated.capitalize()
                new_words.append(translated)
            else:
                new_words.append(word)

        return ' '.join(new_words)

    def augment_dataset(
        self,
        data: List[Dict[str, str]],
        input_key: str = "input",
        output_key: str = "output",
        methods: Optional[List[str]] = None,
        augment_per_sample: int = 1,
        **kwargs,
    ) -> List[Dict[str, str]]:
        """对数据集进行增强。

        Args:
            data: 原始数据集
            input_key: 输入字段名
            output_key: 输出字段名
            methods: 增强方法列表 (synonym/back_translation)
            augment_per_sample: 每个样本生成的增强样本数

        Returns:
            增强后的数据集（包含原始样本）
        """
        if methods is None:
            methods = ["synonym"]

        augmented = list(data)  # 保留原始数据

        for item in data:
            input_text = item.get(input_key, "")
            output_text = item.get(output_key, "")

            for _ in range(augment_per_sample):
                aug_input = input_text
                aug_output = output_text

                for method in methods:
                    if method == "synonym":
                        ratio = kwargs.get("synonym_ratio", 0.3)
                        aug_input = self.synonym_rewrite(aug_input, ratio)
                        aug_output = self.synonym_rewrite(aug_output, ratio)
                    elif method == "back_translation":
                        ratio = kwargs.get("translate_ratio", 0.3)
                        aug_input = self.back_translation_simulate(aug_input, ratio)
                        aug_output = self.back_translation_simulate(aug_output, ratio)

                if aug_input != input_text or aug_output != output_text:
                    new_item = dict(item)
                    new_item[input_key] = aug_input
                    new_item[output_key] = aug_output
                    augmented.append(new_item)

        return augmented