"""
Hybrid score and rank fusion algorithms: Reciprocal Rank Fusion (RRF)
and Relative Score Fusion (RSF).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from .bm25 import BM25Result
from .vector_index import VectorResult


@dataclass(slots=True)
class FusedCandidate:
    """A fused search candidate containing provenance scores from both retrieval branches."""
    doc_id: int
    score: float
    bm25_score: Optional[float] = None
    bm25_rank: Optional[int] = None
    dense_score: Optional[float] = None
    dense_rank: Optional[int] = None
    metadata: dict = field(default_factory=dict)


def reciprocal_rank_fusion(
    bm25_results: List[BM25Result],
    dense_results: List[VectorResult],
    k: int = 60,
    w_bm25: float = 0.5,
    w_dense: float = 0.5,
    top_k: int = 50,
) -> List[FusedCandidate]:
    """
    Combines lexical and dense candidate rankings using Reciprocal Rank Fusion (RRF).
    
    RRF_Score(d) = sum( w_m / (k + rank_m(d)) )
    """
    scores: Dict[int, float] = {}
    bm25_info: Dict[int, tuple[float, int, dict]] = {}
    dense_info: Dict[int, tuple[float, int]] = {}

    # Accumulate BM25 ranks
    for rank, res in enumerate(bm25_results, start=1):
        bm25_info[res.doc_id] = (res.score, rank, res.metadata)
        rrf_contrib = w_bm25 / (k + rank)
        scores[res.doc_id] = scores.get(res.doc_id, 0.0) + rrf_contrib

    # Accumulate Dense ranks
    for rank, res in enumerate(dense_results, start=1):
        dense_info[res.doc_id] = (res.score, rank)
        rrf_contrib = w_dense / (k + rank)
        scores[res.doc_id] = scores.get(res.doc_id, 0.0) + rrf_contrib

    if not scores:
        return []

    # Sort candidates by combined RRF score descending
    sorted_candidates = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

    fused: List[FusedCandidate] = []
    for doc_id, score in sorted_candidates:
        b_score, b_rank, b_meta = bm25_info.get(doc_id, (None, None, {}))
        d_score, d_rank = dense_info.get(doc_id, (None, None))
        fused.append(
            FusedCandidate(
                doc_id=doc_id,
                score=round(score, 5),
                bm25_score=b_score,
                bm25_rank=b_rank,
                dense_score=d_score,
                dense_rank=d_rank,
                metadata=b_meta,
            )
        )

    return fused


def relative_score_fusion(
    bm25_results: List[BM25Result],
    dense_results: List[VectorResult],
    alpha: float = 0.5,
    top_k: int = 50,
) -> List[FusedCandidate]:
    """
    Min-max normalizes lexical and dense scores into [0, 1] and computes a weighted sum:
    Score = alpha * BM25_norm + (1 - alpha) * Dense_norm
    """
    all_doc_ids = set([r.doc_id for r in bm25_results] + [r.doc_id for r in dense_results])
    if not all_doc_ids:
        return []

    # Min-max normalize BM25 scores
    b_scores = {r.doc_id: r.score for r in bm25_results}
    b_meta = {r.doc_id: r.metadata for r in bm25_results}
    b_min = min(b_scores.values()) if b_scores else 0.0
    b_max = max(b_scores.values()) if b_scores else 1.0
    b_range = b_max - b_min if b_max > b_min else 1.0

    # Dense scores (already cosine similarities in [-1, 1], shifted to [0, 1])
    d_scores = {r.doc_id: r.score for r in dense_results}
    d_min = min(d_scores.values()) if d_scores else 0.0
    d_max = max(d_scores.values()) if d_scores else 1.0
    d_range = d_max - d_min if d_max > d_min else 1.0

    scores: Dict[int, float] = {}
    for doc_id in all_doc_ids:
        b_norm = (b_scores[doc_id] - b_min) / b_range if doc_id in b_scores else 0.0
        d_norm = (d_scores[doc_id] - d_min) / d_range if doc_id in d_scores else 0.0
        scores[doc_id] = alpha * b_norm + (1.0 - alpha) * d_norm

    sorted_candidates = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

    fused: List[FusedCandidate] = []
    for doc_id, score in sorted_candidates:
        fused.append(
            FusedCandidate(
                doc_id=doc_id,
                score=round(score, 4),
                bm25_score=b_scores.get(doc_id),
                dense_score=d_scores.get(doc_id),
                metadata=b_meta.get(doc_id, {}),
            )
        )

    return fused
