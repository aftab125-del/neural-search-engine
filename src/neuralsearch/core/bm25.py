"""
Okapi BM25 ranking algorithm with term frequency saturation,
document length normalization, and explainability breakdown.
"""

from __future__ import annotations
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from .inverted_index import InvertedIndex


@dataclass(slots=True)
class BM25Result:
    """Detailed result of a BM25 query evaluation for a single document."""
    doc_id: int
    score: float
    matched_terms: List[str]
    term_contributions: Dict[str, float] = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)


class BM25Ranker:
    """
    Okapi BM25 Ranking Engine.
    
    Default Hyperparameters:
    - k1 = 1.5: Term frequency saturation parameter.
    - b  = 0.75: Document length normalization weight.
    """

    def __init__(self, index: InvertedIndex, k1: float = 1.5, b: float = 0.75):
        self.index = index
        self.k1 = k1
        self.b = b

    def idf(self, term: str) -> float:
        """
        Calculates smoothed Inverse Document Frequency (IDF) with floor at 0.0:
        IDF(q) = ln(1 + (N - n(q) + 0.5) / (n(q) + 0.5))
        """
        n_q = self.index.get_doc_frequency(term)
        if n_q == 0:
            return 0.0

        n_docs = self.index.total_docs
        idf_val = math.log(1.0 + (n_docs - n_q + 0.5) / (n_q + 0.5))
        return max(0.0, idf_val)

    def search(self, query: str, top_k: int = 50) -> List[BM25Result]:
        """
        Executes Okapi BM25 search for the given query string.
        Returns top-K results sorted by score descending.
        """
        query_terms = self.index.tokenizer.tokenize_terms(query)
        if not query_terms or self.index.total_docs == 0:
            return []

        avg_dl = self.index.avg_doc_length
        if avg_dl <= 0:
            avg_dl = 1.0

        # Accumulators per document
        doc_scores: Dict[int, float] = {}
        doc_matched_terms: Dict[int, List[str]] = {}
        doc_term_contributions: Dict[int, Dict[str, float]] = {}

        # Evaluate each query term against the inverted index postings
        for term in set(query_terms):
            term_idf = self.idf(term)
            if term_idf <= 0.0:
                continue

            postings = self.index.get_postings(term)
            for p in postings:
                doc_len = self.index.get_doc_length(p.doc_id)
                
                # Length normalization term
                len_norm = 1.0 - self.b + self.b * (doc_len / avg_dl)
                # Term frequency saturation
                tf_score = (p.term_freq * (self.k1 + 1.0)) / (p.term_freq + self.k1 * len_norm)
                term_score = term_idf * tf_score

                # Accumulate score
                doc_scores[p.doc_id] = doc_scores.get(p.doc_id, 0.0) + term_score

                if p.doc_id not in doc_matched_terms:
                    doc_matched_terms[p.doc_id] = []
                    doc_term_contributions[p.doc_id] = {}

                doc_matched_terms[p.doc_id].append(term)
                doc_term_contributions[p.doc_id][term] = round(term_score, 4)

        if not doc_scores:
            return []

        # Sort candidate documents by total BM25 score descending
        sorted_docs = sorted(doc_scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

        results: List[BM25Result] = []
        for doc_id, score in sorted_docs:
            results.append(
                BM25Result(
                    doc_id=doc_id,
                    score=round(score, 4),
                    matched_terms=doc_matched_terms.get(doc_id, []),
                    term_contributions=doc_term_contributions.get(doc_id, {}),
                    metadata=self.index.get_doc_metadata(doc_id) or {},
                )
            )

        return results
