"""Vector Store - In-memory TF-IDF based vector storage with cosine similarity."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Optional, Tuple

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class VectorStore:
    """In-memory vector store using TF-IDF vectors and cosine similarity."""

    def __init__(self) -> None:
        self.documents: Dict[str, str] = {}  # doc_id -> raw text
        self.chunks: Dict[str, List[str]] = {}  # doc_id -> list of chunks
        self.chunk_ids: List[str] = []  # flat list of "doc_id:chunk_idx"
        self.chunk_texts: List[str] = []  # parallel to chunk_ids
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words="english",
            token_pattern=r"(?u)\b\w+\b",
        )
        self.vectors: Optional[np.ndarray] = None
        self._fitted = False

    def add_document(self, doc_id: str, text: str, chunk_size: int = 200, overlap: int = 50) -> int:
        """Add a document, chunk it, and vectorize.

        Returns the number of chunks created.
        """
        self.documents[doc_id] = text
        chunks = self._chunk_text(text, chunk_size, overlap)
        self.chunks[doc_id] = chunks

        for idx, chunk in enumerate(chunks):
            cid = f"{doc_id}:chunk_{idx}"
            self.chunk_ids.append(cid)
            self.chunk_texts.append(chunk)

        self._rebuild_vectors()
        return len(chunks)

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document and all its chunks."""
        if doc_id not in self.documents:
            return False
        del self.documents[doc_id]
        if doc_id in self.chunks:
            del self.chunks[doc_id]

        # Remove associated chunk_ids and chunk_texts
        keep_ids = []
        keep_texts = []
        for cid, ctext in zip(self.chunk_ids, self.chunk_texts):
            if not cid.startswith(f"{doc_id}:"):
                keep_ids.append(cid)
                keep_texts.append(ctext)
        self.chunk_ids = keep_ids
        self.chunk_texts = keep_texts

        self._rebuild_vectors()
        return True

    def search(
        self, query: str, top_k: int = 5, threshold: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Search for chunks similar to query using TF-IDF cosine similarity."""
        if not self._fitted or not self.chunk_texts:
            return []

        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.vectors).flatten()

        results = []
        for idx in np.argsort(similarities)[::-1]:
            if similarities[idx] < threshold:
                break
            cid = self.chunk_ids[idx]
            doc_id = cid.rsplit(":chunk_", 1)[0]
            results.append(
                {
                    "chunk_id": cid,
                    "doc_id": doc_id,
                    "text": self.chunk_texts[idx],
                    "score": float(similarities[idx]),
                }
            )
            if len(results) >= top_k:
                break

        return results

    def keyword_search(
        self, query: str, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """Simple keyword-based search using BM25-like scoring."""
        query_terms = self._tokenize(query)
        query_tf = Counter(query_terms)

        scores: List[Tuple[int, float]] = []
        for idx, chunk_text in enumerate(self.chunk_texts):
            chunk_terms = self._tokenize(chunk_text)
            chunk_tf = Counter(chunk_terms)
            chunk_len = len(chunk_terms)

            score = 0.0
            for term, qfreq in query_tf.items():
                if term in chunk_tf:
                    # Simplified BM25-like scoring
                    tf = chunk_tf[term]
                    idf = math.log(
                        (len(self.chunk_texts) + 1) / (sum(1 for c in self.chunk_texts if term in self._tokenize(c)) + 0.5) + 1
                    )
                    score += idf * tf * (3 * tf) / (tf + 1.5) / (chunk_len + 1)

            scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in scores[:top_k]:
            if score > 0:
                cid = self.chunk_ids[idx]
                doc_id = cid.rsplit(":chunk_", 1)[0]
                results.append(
                    {
                        "chunk_id": cid,
                        "doc_id": doc_id,
                        "text": self.chunk_texts[idx],
                        "score": score,
                    }
                )
        return results

    def get_document_count(self) -> int:
        return len(self.documents)

    def get_chunk_count(self) -> int:
        return len(self.chunk_ids)

    def get_all_documents(self) -> List[Dict[str, Any]]:
        return [
            {"doc_id": did, "text": text[:200] + "..." if len(text) > 200 else text}
            for did, text in self.documents.items()
        ]

    # ── Internal ────────────────────────────────────────────────────

    def _chunk_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split text into overlapping chunks."""
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return []
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            # Try to break at sentence boundary
            if end < len(text):
                last_period = text.rfind("。", start, end)
                if last_period == -1:
                    last_period = text.rfind(".", start, end)
                if last_period > start:
                    end = last_period + 1
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            # If we processed the whole text, stop
            if end >= len(text):
                break
            start = end - overlap
            if start >= len(text) or start < 0:
                break
        return chunks

    def _rebuild_vectors(self) -> None:
        """Rebuild TF-IDF vectors from all chunk texts."""
        if not self.chunk_texts:
            self.vectors = None
            self._fitted = False
            return
        self.vectors = self.vectorizer.fit_transform(self.chunk_texts)
        self._fitted = True

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple whitespace + punctuation tokenization."""
        return re.findall(r"\b\w+\b", text.lower())
