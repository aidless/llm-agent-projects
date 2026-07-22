"""
引用溯源模块
从 LLM 生成结果中提取引用标注，关联到检索到的原始文档段落
"""
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

from loguru import logger
from core.config import settings
from core.retrieval.retriever import RetrievalResult


@dataclass
class CitationSource:
    """引用来源信息"""

    source_file: str  # 来源文件名
    doc_title: str  # 文档标题
    chunk_id: int = 0  # 文本块编号
    content: str = ""  # 引用的原始段落内容
    score: float = 0.0  # 相关性得分
    start_index: int = 0  # 在原文中的起始位置
    end_index: int = 0  # 在原文中的结束位置

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "source_file": self.source_file,
            "doc_title": self.doc_title,
            "chunk_id": self.chunk_id,
            "content": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "score": round(self.score, 4),
            "start_index": self.start_index,
            "end_index": self.end_index,
        }


class CitationExtractor:
    """
    引用提取器

    从检索结果中提取引用信息，并可选择性地在生成文本中添加引用标注

    使用示例:
        extractor = CitationExtractor()
        citations = extractor.extract_citations(retrieval_results, max_sources=3)
    """

    def __init__(self, max_sources: int = None, enabled: bool = None):
        """
        初始化引用提取器

        Args:
            max_sources: 最大引用来源数
            enabled: 是否启用引用溯源
        """
        self.max_sources = max_sources or settings.citation_max_sources
        self.enabled = enabled if enabled is not None else settings.citation_enabled

    def extract_citations(
        self,
        results: List[RetrievalResult],
        max_sources: int = None,
    ) -> List[CitationSource]:
        """
        从检索结果中提取引用来源

        Args:
            results: 检索结果列表
            max_sources: 最大引用数量

        Returns:
            List[CitationSource]: 引用来源列表
        """
        if not self.enabled:
            return []

        if not results:
            return []

        k = max_sources or self.max_sources

        # 按得分排序取 top_k
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)[:k]

        citations = []
        for i, result in enumerate(sorted_results):
            meta = result.metadata or {}
            citation = CitationSource(
                source_file=meta.get("source_file", "未知文件"),
                doc_title=meta.get("doc_title", "未知标题"),
                chunk_id=meta.get("chunk_id", i),
                content=result.content,
                score=result.score,
                start_index=result.metadata.get("start_index", 0),
                end_index=result.metadata.get("end_index", 0),
            )
            citations.append(citation)

        logger.debug(f"提取 {len(citations)} 个引用来源")
        return citations

    def build_context_with_sources(
        self,
        results: List[RetrievalResult],
        max_sources: int = None,
    ) -> str:
        """
        构建带来源标注的上下文文本

        将检索结果格式化为带编号引用的文本，供 LLM 使用

        Args:
            results: 检索结果列表
            max_sources: 最大引用数量

        Returns:
            str: 格式化的上下文文本
        """
        if not results:
            return "（未找到相关参考资料）"

        k = max_sources or self.max_sources
        sorted_results = sorted(results, key=lambda x: x.score, reverse=True)[:k]

        context_parts = []
        for i, result in enumerate(sorted_results, 1):
            source = result.metadata.get("source_file", "未知来源")
            title = result.metadata.get("doc_title", "")
            source_info = f"[来源{i}: {source}]"
            if title and title != source.replace(".", ""):
                source_info = f"[来源{i}: {title} ({source})]"

            context_parts.append(f"{source_info}\n{result.content}")

        return "\n\n---\n\n".join(context_parts)

    def add_citation_marks(
        self,
        answer: str,
        citations: List[CitationSource],
    ) -> str:
        """
        在回答文本中添加引用标注

        使用简单的关键词匹配方法，在回答中标注引用编号

        Args:
            answer: LLM 生成的回答文本
            citations: 引用来源列表

        Returns:
            str: 添加了引用标注的回答文本
        """
        if not self.enabled or not citations:
            return answer

        marked_answer = answer
        for i, citation in enumerate(citations, 1):
            # 取引用内容的前50个字符作为匹配关键词
            key_text = citation.content[:50].strip()
            if key_text and key_text in marked_answer:
                marked_answer = marked_answer.replace(
                    key_text,
                    f"{key_text} [{i}]",
                    1,  # 只替换第一个匹配
                )

        # 在回答末尾添加引用列表
        if citations:
            ref_list = "\n\n**参考来源：**\n"
            for i, c in enumerate(citations, 1):
                ref_list += f"[{i}] {c.doc_title} ({c.source_file})\n"
            marked_answer += ref_list

        return marked_answer

    def build_citation_prompt(self) -> str:
        """
        构建引用相关的 prompt 指令

        指导 LLM 在回答时引用参考资料的编号

        Returns:
            str: prompt 指令文本
        """
        if not self.enabled:
            return ""

        return (
            "\n\n引用要求："
            "请在回答中引用参考资料时，使用 [来源N] 的格式标注引用来源编号。"
            "如果某些内容没有在参考资料中出现，请明确说明这是你的补充知识。"
            "不要虚构引用。"
        )
