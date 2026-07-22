"""Paper text parser - extracts raw text into a ParsedPaper.

In production this would integrate PyMuPDF / pdfplumber.  For this project
the parser works on plain-text input (simulating PDF extraction output).
"""

from __future__ import annotations

import re
from typing import Optional

from app.models import AuthorInfo, ParsedPaper, PaperStructure


# ---------------------------------------------------------------------------
# Section heading patterns
# ---------------------------------------------------------------------------

_SECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("title", re.compile(r"^(?:Title|TITLE)\s*[:：]?\s*(.+)", re.MULTILINE)),
    ("abstract", re.compile(
        r"(?:Abstract|ABSTRACT)\s*[:：]?\s*\n?(.*?)(?=\n\s*(?:1[\.\s]|Introduction|INTRODUCTION)|$)",
        re.DOTALL,
    )),
    ("introduction", re.compile(
        r"(?:1[\.\s]+\s*Introduction|INTRODUCTION)\s*\n?(.*?)(?=\n\s*(?:2[\.\s]|Related|Background|Method|METHODOLOGY)|$)",
        re.DOTALL,
    )),
    ("method", re.compile(
        r"(?:2[\.\s]+\s*(?:Method|METHODOLOGY|Proposed|Approach|OUR APPROACH))\s*\n?(.*?)(?=\n\s*(?:3[\.\s]|Experiment|RESULT|Evaluation)|$)",
        re.DOTALL | re.IGNORECASE,
    )),
    ("experiments", re.compile(
        r"(?:3[\.\s]+\s*(?:Experiment|Experiments|RESULT|RESULTS|Evaluation|EXPERIMENTS))\s*\n?(.*?)(?=\n\s*(?:4[\.\s]|Conclusion|CONCLUSION|Discussion|Acknowledgment)|$)",
        re.DOTALL | re.IGNORECASE,
    )),
    ("conclusion", re.compile(
        r"(?:4[\.\s]+\s*(?:Conclusion|Conclusions|CONCLUSION))\s*\n?(.*?)(?=\n\s*(?:Reference|ACKNOWLEDGMENT|Appendix)|$)",
        re.DOTALL | re.IGNORECASE,
    )),
    ("references", re.compile(
        r"(?:References|REFERENCES|Bibliography)\s*\n?(.*)",
        re.DOTALL,
    )),
]

_AUTHOR_PATTERN = re.compile(
    r"(?:Author|AUTHOR[S]?)\s*[:：]?\s*(.+?)(?=\n\n|\n\s*[A-Z])",
    re.MULTILINE,
)

_AFFILIATION_PATTERN = re.compile(
    r"(?:Affiliation|AFFILIATION|University|Institute|Department)\s*[:：]?\s*(.+?)(?=\n|$)",
    re.MULTILINE,
)

_KEYWORDS_PATTERN = re.compile(
    r"(?:^Keywords|KEYWORDS)\s*[:\uff1a]?\s*(.+?)(?=\n|$)",
    re.MULTILINE,
)

_DATASET_PATTERN = re.compile(
    r"\b(?:dataset|benchmark|corpus)\b[\s:]+([A-Z][A-Za-z0-9\-]+)",
    re.IGNORECASE,
)


class PaperParser:
    """Parse plain-text paper content into a structured ParsedPaper."""

    def __init__(self) -> None:
        self._section_patterns = _SECTION_PATTERNS
        self._author_pattern = _AUTHOR_PATTERN
        self._affiliation_pattern = _AFFILIATION_PATTERN
        self._keywords_pattern = _KEYWORDS_PATTERN
        self._dataset_pattern = _DATASET_PATTERN

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, text: str) -> ParsedPaper:
        """Parse the full paper text and return a ParsedPaper."""
        structure = self._extract_structure(text)
        authors = self._extract_authors(text)
        keywords = self._extract_keywords(text)
        datasets = self._extract_datasets(text)
        page_count = self._estimate_page_count(text)

        return ParsedPaper(
            raw_text=text,
            structure=structure,
            authors=authors,
            keywords=keywords,
            datasets=datasets,
            page_count=page_count,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_structure(self, text: str) -> PaperStructure:
        """Try to identify standard academic sections in *text*."""
        fields: dict[str, str] = {}
        for key, pattern in self._section_patterns:
            match = pattern.search(text)
            fields[key] = match.group(1).strip() if match else ""

        # Fallback: if no title is found via pattern, use first non-empty line
        if not fields.get("title"):
            for line in text.strip().splitlines():
                line = line.strip()
                if line and not line.startswith("#") and len(line) > 5:
                    fields["title"] = line
                    break

        return PaperStructure(**fields)

    def _extract_authors(self, text: str) -> list[AuthorInfo]:
        """Heuristic author extraction."""
        authors: list[AuthorInfo] = []
        for match in self._author_pattern.finditer(text):
            name_str = match.group(1).strip()
            # Handle "Name1, Name2" or "Name1 and Name2"
            names = re.split(r"[,;]|\band\b", name_str)
            for name in names:
                name = name.strip().strip("*")
                if name and len(name) > 2:
                    authors.append(AuthorInfo(name=name))
        if not authors:
            authors.append(AuthorInfo(name="Unknown Author"))
        return authors

    def _extract_keywords(self, text: str) -> list[str]:
        """Extract keyword list from paper."""
        match = self._keywords_pattern.search(text)
        if not match:
            return []
        raw = match.group(1)
        # Split on commas, semicolons, or periods
        kws = re.split(r"[,;.\t]", raw)
        return [kw.strip().lower() for kw in kws if kw.strip() and len(kw.strip()) > 1]

    def _extract_datasets(self, text: str) -> list[str]:
        """Extract dataset/benchmark names mentioned in the paper."""
        matches = self._dataset_pattern.findall(text)
        seen: set[str] = set()
        result: list[str] = []
        for m in matches:
            m_lower = m.lower()
            if m_lower not in seen:
                seen.add(m_lower)
                result.append(m)
        return result

    @staticmethod
    def _estimate_page_count(text: str) -> int:
        """Rough page estimate (~3000 chars per page for double-column)."""
        return max(1, len(text) // 3000)