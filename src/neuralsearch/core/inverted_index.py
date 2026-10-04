"""
In-memory Inverted Index with posting lists, term frequencies, positions,
and dynamic document length tracking.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set
from .tokenizer import Tokenizer, Token


@dataclass(slots=True)
class Posting:
    """A single posting entry for a term in a document."""
    doc_id: int
    term_freq: int
    positions: List[int] = field(default_factory=list)


class InvertedIndex:
    """
    In-memory Inverted Index optimized for fast lexical postings traversal.
    """

    def __init__(self, tokenizer: Optional[Tokenizer] = None):
        self.tokenizer = tokenizer or Tokenizer()
        # term -> list of Postings
        self._index: Dict[str, List[Posting]] = {}
        # doc_id -> total token count in document
        self._doc_lengths: Dict[int, int] = {}
        # doc_id -> metadata dictionary (title, path, raw text, etc.)
        self._doc_metadata: Dict[int, dict] = {}
        # cached statistics
        self._total_terms: int = 0

    @property
    def total_docs(self) -> int:
        return len(self._doc_lengths)

    @property
    def avg_doc_length(self) -> float:
        if not self._doc_lengths:
            return 0.0
        return self._total_terms / len(self._doc_lengths)

    @property
    def vocabulary_size(self) -> int:
        return len(self._index)

    def get_doc_length(self, doc_id: int) -> int:
        return self._doc_lengths.get(doc_id, 0)

    def get_doc_metadata(self, doc_id: int) -> Optional[dict]:
        return self._doc_metadata.get(doc_id)

    def get_doc_frequency(self, term: str) -> int:
        """Returns the number of documents containing the term (n(q))."""
        postings = self._index.get(term)
        return len(postings) if postings else 0

    def get_postings(self, term: str) -> List[Posting]:
        """Returns the posting list for a term."""
        return self._index.get(term, [])

    def add_document(self, doc_id: int, text: str, metadata: Optional[dict] = None) -> None:
        """
        Tokenizes and indexes a document. If doc_id already exists, it is overwritten.
        """
        if doc_id in self._doc_lengths:
            self.remove_document(doc_id)

        tokens = self.tokenizer.tokenize(text)
        doc_length = len(tokens)
        self._doc_lengths[doc_id] = doc_length
        self._doc_metadata[doc_id] = metadata or {}
        self._total_terms += doc_length

        # Aggregate positions and frequencies by stemmed term
        term_map: Dict[str, List[int]] = {}
        for token in tokens:
            term = token.stemmed
            if term not in term_map:
                term_map[term] = []
            term_map[term].append(token.position)

        for term, positions in term_map.items():
            if term not in self._index:
                self._index[term] = []
            self._index[term].append(
                Posting(doc_id=doc_id, term_freq=len(positions), positions=positions)
            )

    def remove_document(self, doc_id: int) -> bool:
        """Removes a document from the inverted index."""
        if doc_id not in self._doc_lengths:
            return False

        old_length = self._doc_lengths.pop(doc_id)
        self._doc_metadata.pop(doc_id, None)
        self._total_terms -= old_length

        # Remove postings for this doc_id
        for term, postings in list(self._index.items()):
            self._index[term] = [p for p in postings if p.doc_id != doc_id]
            if not self._index[term]:
                del self._index[term]

        return True

    def get_all_doc_ids(self) -> Set[int]:
        return set(self._doc_lengths.keys())

    def clear(self) -> None:
        self._index.clear()
        self._doc_lengths.clear()
        self._doc_metadata.clear()
        self._total_terms = 0
