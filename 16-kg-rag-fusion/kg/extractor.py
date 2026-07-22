"""Entity and Relation Extractor - Rule-based + LLM mock.

⚠️ **2026-07-22**：原版只用硬编码 regex 字典（OpenAI/Hinton/Transformer 等），
对未登录词无效。本版本新增 `provider="spacy"` 选项，使用 `zh_core_web_trf`
做真实 NER。默认仍为 `regex`（向后兼容 + 测试不需重跑）。

用法：
    # 默认 regex 字典（仅匹配 ~30 个已知实体）
    ext = Extractor()

    # spacy 中文 NER（推荐，需安装：pip install spacy && python -m spacy download zh_core_web_trf）
    ext = Extractor(provider="spacy", spacy_model="zh_core_web_trf")

    # 自动回退：spacy 不可用 → regex
    ext = Extractor(provider="spacy", spacy_fallback_to_regex=True)
"""

from __future__ import annotations

import logging
import re
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Predefined rule patterns for NER ────────────────────────────────

_TECH_PATTERNS: List[Dict[str, str]] = [
    {"pattern": r"[A-Z][a-z]*(?:\s+[A-Z][a-z]*)*\s+(?:模型|网络|算法|架构|框架)", "type": "Technology"},
    {"pattern": r"(?:深度学习|机器学习|强化学习|自然语言处理|计算机视觉|知识图谱|大语言模型|生成式AI|推荐系统|语音识别|图神经网络)", "type": "Technology"},
]

_ORG_PATTERNS: List[Dict[str, str]] = [
    {"pattern": r"(?:OpenAI|Google|Meta|Microsoft|Baidu|Alibaba|ByteDance|Tsinghua|Stanford|MIT|DeepMind|Anthropic|NVIDIA|Hugging Face)", "type": "Organization"},
]

_PERSON_PATTERNS: List[Dict[str, str]] = [
    {"pattern": r"(?:Geoffrey Hinton|Yann LeCun|Yoshua Bengio|Andrew Ng|Ilya Sutskever|Sam Altman|Demis Hassabis|Andrej Karpathy)", "type": "Person"},
]

_CONCEPT_PATTERNS: List[Dict[str, str]] = [
    {"pattern": r"(?:注意力机制|Transformer|RNN|LSTM|CNN|GAN|自监督学习|迁移学习|多模态|提示工程|RAG|微调|预训练)", "type": "Concept"},
]

_ALL_PATTERNS = _TECH_PATTERNS + _ORG_PATTERNS + _PERSON_PATTERNS + _CONCEPT_PATTERNS

# spacy 中文 NER 标签 → 我们内部类型映射
_SPACY_LABEL_MAP = {
    "PERSON": "Person",
    "ORG": "Organization",
    "GPE": "Organization",      # 地缘政治实体（国家/城市）按 Org 处理
    "LOC": "Organization",
    "PRODUCT": "Technology",
    "EVENT": "Concept",
    "WORK_OF_ART": "Concept",
    "LAW": "Concept",
    "LANGUAGE": "Concept",
}


class Extractor:
    """Extract entities and relations from text using rules + LLM mock."""

    def __init__(
        self,
        use_llm_mock: bool = True,
        provider: str = "regex",
        spacy_model: str = "zh_core_web_trf",
        spacy_fallback_to_regex: bool = True,
    ) -> None:
        """初始化 Extractor。

        Args:
            use_llm_mock: 是否启用 LLM mock（仅对 regex provider 有效）
            provider: "regex"（默认，硬编码字典）或 "spacy"（真 NER）
            spacy_model: spacy 模型名，默认 "zh_core_web_trf"（~500MB）
            spacy_fallback_to_regex: spacy 加载失败时是否回退到 regex
        """
        self.use_llm_mock = use_llm_mock
        self.provider = provider
        self.spacy_fallback_to_regex = spacy_fallback_to_regex
        self._spacy_nlp = None

        if provider == "spacy":
            self._load_spacy(spacy_model)

    def _load_spacy(self, model_name: str) -> None:
        """懒加载 spacy 模型；失败时按 spacy_fallback_to_regex 处理。"""
        try:
            import spacy  # noqa: F401
            import spacy.cli  # noqa: F401  (ensure CLI is available for model download)

            self._spacy_nlp = spacy.load(model_name)
            logger.info(f"spaCy NER 加载成功: {model_name}")
        except Exception as e:
            msg = f"spaCy 模型 '{model_name}' 加载失败: {e}"
            if self.spacy_fallback_to_regex:
                logger.warning(f"{msg}。回退到 regex provider。")
                self.provider = "regex"
            else:
                raise RuntimeError(
                    f"{msg}。请先下载模型：\n"
                    f"  python -m spacy download {model_name}\n"
                    "或设置 spacy_fallback_to_regex=True。"
                )

    # ── Entity extraction ───────────────────────────────────────────

    def extract_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract named entities from text.

        Returns list of {"name": ..., "type": ..., "entity_id": ...}.
        """
        if self.provider == "spacy" and self._spacy_nlp is not None:
            return self._spacy_extract_entities(text)
        return self._regex_extract_entities(text)

    def _regex_extract_entities(self, text: str) -> List[Dict[str, str]]:
        entities: List[Dict[str, str]] = []
        seen: set = set()

        # Rule-based extraction
        for entry in _ALL_PATTERNS:
            for match in re.finditer(entry["pattern"], text):
                name = match.group(0).strip()
                if name not in seen:
                    seen.add(name)
                    entity_id = self._make_entity_id(name)
                    entities.append(
                        {"name": name, "type": entry["type"], "entity_id": entity_id}
                    )

        # LLM mock enrichment (adds a couple extra entities if enabled)
        if self.use_llm_mock:
            mock_entities = self._llm_mock_extract(text, seen)
            entities.extend(mock_entities)

        return entities

    def _spacy_extract_entities(self, text: str) -> List[Dict[str, str]]:
        """用 spaCy 中文模型做真实 NER。"""
        assert self._spacy_nlp is not None, "spaCy model not loaded"
        # spacy 中文模型对长度有限制，超长文本分段处理
        MAX_CHARS = 100_000
        chunks = [text[i : i + MAX_CHARS] for i in range(0, len(text), MAX_CHARS)]

        entities: List[Dict[str, str]] = []
        seen: set = set()

        for chunk in chunks:
            doc = self._spacy_nlp(chunk)
            for ent in doc.ents:
                label = _SPACY_LABEL_MAP.get(ent.label_, "Concept")
                name = ent.text.strip()
                if name and name not in seen and len(name) >= 2:
                    seen.add(name)
                    entities.append(
                        {
                            "name": name,
                            "type": label,
                            "entity_id": self._make_entity_id(name),
                            "spacy_label": ent.label_,
                        }
                    )
        return entities

    def _llm_mock_extract(
        self, text: str, already_seen: set
    ) -> List[Dict[str, str]]:
        """Simulate LLM entity extraction (regex provider only)."""
        additional: List[Dict[str, str]] = []
        # Simple heuristic: capitalized words not yet extracted
        words = re.findall(r"\b[A-Z][a-zA-Z]{1,}(?:\s+[A-Z][a-zA-Z]+)*\b", text)
        for w in words:
            if w not in already_seen and len(w) >= 2:
                additional.append(
                    {
                        "name": w,
                        "type": "Unknown",
                        "entity_id": self._make_entity_id(w),
                    }
                )
                already_seen.add(w)
        return additional[:5]  # limit mock results

    # ── Relation extraction ─────────────────────────────────────────

    def extract_relations(
        self, text: str, entities: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Extract relations between entities from text.

        Returns list of {"source": entity_id, "target": entity_id, "relation": ...}.
        """
        relations: List[Dict[str, str]] = []

        # Chinese verb patterns for relation extraction
        # We look for entity_name + verb + entity_name patterns
        entity_name_map = {e["name"]: e["entity_id"] for e in entities}

        if not entities:
            return relations

        # Build regex patterns from known entity names
        verb_patterns = {
            "created": ["开发了", "发明了", "创建了", "提出"],
            "belongs_to": ["是", "属于", "隶属于"],
            "uses": ["基于", "使用", "采用"],
            "improved": ["改进了", "优化了", "扩展了"],
            "trained": ["训练了", "训练出"],
            "published": ["发表", "发布", "推出"],
            "applied_in": ["应用于", "用于"],
            "contains": ["包含", "包括"],
            "led_by": ["创始人", "CEO", "领导", "负责人"],
            "developed_by": ["的"],
        }

        # For each pair of entities, check if they appear with a verb between them
        all_verbs = []
        for rel_type, verbs in verb_patterns.items():
            all_verbs.extend([(v, rel_type) for v in verbs])

        for i, ent_a in enumerate(entities):
            for ent_b in entities[i + 1:]:
                name_a = ent_a["name"]
                name_b = ent_b["name"]
                for verb, rel_type in all_verbs:
                    # Check both orders: A verb B and B verb A
                    for src, tgt in [(name_a, name_b), (name_b, name_a)]:
                        # Check if text contains src...verb...tgt
                        pattern_src = re.escape(src)
                        pattern_tgt = re.escape(tgt)
                        pattern_verb = re.escape(verb)
                        # Allow optional whitespace between them
                        check_pattern = f"{pattern_src}\\s*{pattern_verb}\\s*{pattern_tgt}"
                        if re.search(check_pattern, text):
                            relations.append(
                                {
                                    "source": ent_a["entity_id"] if src == name_a else ent_b["entity_id"],
                                    "target": ent_b["entity_id"] if tgt == name_b else ent_a["entity_id"],
                                    "relation": rel_type,
                                }
                            )
                            break
                    # Avoid duplicate relations for same entity pair
                    if any(r["source"] in [ent_a["entity_id"], ent_b["entity_id"]] for r in relations):
                        break

        # LLM mock relations
        if self.use_llm_mock and len(relations) == 0 and len(entities) >= 2:
            # Generate a plausible relation between first two entities
            relations.append(
                {
                    "source": entities[0]["entity_id"],
                    "target": entities[1]["entity_id"],
                    "relation": "related_to",
                }
            )

        return relations

    # ── Entity disambiguation / merging ─────────────────────────────

    def disambiguate(
        self, entities: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        """Merge duplicate entities based on name similarity."""
        merged: List[Dict[str, str]] = []
        seen_names: Dict[str, str] = {}

        for ent in entities:
            name_lower = ent["name"].lower()
            matched = False
            for existing_name, existing_id in seen_names.items():
                if self._name_similarity(existing_name, name_lower) > 0.85:
                    matched = True
                    break
            if not matched:
                merged.append(ent)
                seen_names[ent["name"].lower()] = ent["entity_id"]

        return merged

    # ── Helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _make_entity_id(name: str) -> str:
        return re.sub(r"\s+", "_", name.strip().lower())

    @staticmethod
    def _fuzzy_match_entity(
        name: str, entity_name_map: Dict[str, str]
    ) -> Optional[str]:
        # Exact match
        if name in entity_name_map:
            return entity_name_map[name]
        # Substring match
        for ename, eid in entity_name_map.items():
            if ename in name or name in ename:
                return eid
        return None

    @staticmethod
    def _name_similarity(a: str, b: str) -> float:
        """Simple character-level similarity."""
        if a == b:
            return 1.0
        set_a = set(a)
        set_b = set(b)
        if not set_a or not set_b:
            return 0.0
        intersection = len(set_a & set_b)
        union = len(set_a | set_b)
        return intersection / union if union else 0.0
