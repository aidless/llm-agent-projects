"""Structure extractor -- provides higher-level utilities for paper structure analysis."""

from __future__ import annotations

from app.models import ParsedPaper, PaperStructure


class StructureExtractor:
    """Analyse and enrich the structure of a parsed paper."""

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_section_completion(self, paper: ParsedPaper) -> dict[str, float]:
        """Return per-section completion ratio (0.0 -- 1.0).

        A section is considered "complete" if it has a reasonable amount of
        text.  We use simple heuristic character thresholds.
        """
        structure = paper.structure
        sections: dict[str, str] = {
            "title": structure.title,
            "abstract": structure.abstract,
            "introduction": structure.introduction,
            "method": structure.method,
            "experiments": structure.experiments,
            "conclusion": structure.conclusion,
            "references": structure.references,
        }
        thresholds = {
            "title": 10,
            "abstract": 100,
            "introduction": 200,
            "method": 300,
            "experiments": 300,
            "conclusion": 100,
            "references": 50,
        }
        result: dict[str, float] = {}
        for sec, content in sections.items():
            thr = thresholds.get(sec, 100)
            ratio = min(1.0, len(content.strip()) / thr) if thr > 0 else 0.0
            result[sec] = round(ratio, 2)
        return result

    def get_missing_sections(self, paper: ParsedPaper, min_chars: int = 50) -> list[str]:
        """Return list of section names that are missing or too short."""
        structure = paper.structure
        missing: list[str] = []
        mapping = {
            "Abstract": structure.abstract,
            "Introduction": structure.introduction,
            "Method": structure.method,
            "Experiments": structure.experiments,
            "Conclusion": structure.conclusion,
            "References": structure.references,
        }
        for name, content in mapping.items():
            if len(content.strip()) < min_chars:
                missing.append(name)
        return missing

    def get_structure_summary(self, paper: ParsedPaper) -> str:
        """Return a human-readable summary of the paper structure."""
        completion = self.get_section_completion(paper)
        lines: list[str] = [f"Title: {paper.structure.title or '(missing)'}"]
        lines.append(f"Authors: {', '.join(a.name for a in paper.authors)}")
        lines.append(f"Keywords: {', '.join(paper.keywords) or '(none)'}")
        lines.append(f"Datasets: {', '.join(paper.datasets) or '(none)'}")
        lines.append(f"Estimated pages: {paper.page_count}")
        lines.append("")
        for sec, ratio in completion.items():
            status = "OK" if ratio >= 0.8 else "partial" if ratio > 0 else "MISSING"
            lines.append(f"  {sec}: {status} (completeness {ratio:.0%})")
        return "\n".join(lines)

    def extract_key_sentences(self, paper: ParsedPaper, section: str, max_sentences: int = 3) -> list[str]:
        """Extract the first *max_sentences* non-trivial sentences from a section."""
        structure = paper.structure
        text = getattr(structure, section, "")
        if not text:
            return []
        import re
        sentences = re.split(r'(?<=[.!?])\s+', text)
        result = [s.strip() for s in sentences if len(s.strip()) > 20]
        return result[:max_sentences]