"""Answer Generator - Generates answers from fused context."""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class AnswerGenerator:
    """Generates answers from fused KG+RAG context using template-based
    generation (LLM mock).
    """

    def __init__(self) -> None:
        pass

    def generate(
        self,
        query: str,
        context: str,
        linked_entities: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """Generate an answer based on the fused context.

        Args:
            query: User query.
            context: Fused context string from ResultFuser.
            linked_entities: Linked entities from EntityLinker.

        Returns:
            Generated answer with source attribution.
        """
        if not context.strip():
            return {
                "query": query,
                "answer": "抱歉，未能找到与您的问题相关的信息。请尝试换一种表述。",
                "confidence": 0.0,
                "sources": [],
            }

        # Build answer from context
        answer_parts = []
        sources = []

        entity_names = []
        if linked_entities:
            entity_names = [e["name"] for e in linked_entities]
            sources.append({"type": "knowledge_graph", "entities": entity_names})

        # Extract facts from context
        kg_facts = [
            line for line in context.split("\n") if line.startswith("[知识图谱]")
        ]
        doc_chunks = [
            line for line in context.split("\n") if line.startswith("[文档]")
        ]

        if entity_names:
            answer_parts.append(
                f"根据知识图谱分析，与 '{entity_names[0]}' 相关的信息如下:"
            )

        for fact in kg_facts:
            clean = fact.replace("[知识图谱] ", "")
            answer_parts.append(clean)
            sources.append({"type": "kg_fact", "content": clean})

        for chunk in doc_chunks[:3]:
            clean = chunk.replace("[文档] ", "")
            answer_parts.append(f"此外，相关文档指出: {clean}")
            sources.append({"type": "document", "content": clean[:100]})

        answer = " ".join(answer_parts) if answer_parts else context

        # Compute confidence
        confidence = min(0.5 + 0.1 * len(kg_facts) + 0.05 * len(doc_chunks), 1.0)

        return {
            "query": query,
            "answer": answer,
            "confidence": round(confidence, 2),
            "sources": sources,
            "linked_entities": entity_names,
            "kg_facts_count": len(kg_facts),
            "doc_chunks_count": len(doc_chunks),
        }
