"""
Quantitative Information Retrieval evaluation metrics: NDCG@K, MRR, Precision@K, Recall@K.
"""

from __future__ import annotations
import math
from typing import Dict, List, Set


def dcg_at_k(relevance_scores: List[float], k: int = 10) -> float:
    """Computes Discounted Cumulative Gain at rank K."""
    score = 0.0
    for i, rel in enumerate(relevance_scores[:k], start=1):
        if rel > 0:
            score += (2.0**rel - 1.0) / math.log2(i + 1)
    return score


def ndcg_at_k(actual_doc_ids: List[int], ground_truth_relevance: Dict[int, float], k: int = 10) -> float:
    """
    Computes Normalized Discounted Cumulative Gain at rank K.
    """
    actual_rels = [ground_truth_relevance.get(doc_id, 0.0) for doc_id in actual_doc_ids[:k]]
    actual_dcg = dcg_at_k(actual_rels, k=k)

    # Ideal ranking
    ideal_rels = sorted(ground_truth_relevance.values(), reverse=True)[:k]
    ideal_dcg = dcg_at_k(ideal_rels, k=k)

    if ideal_dcg == 0.0:
        return 0.0
    return round(actual_dcg / ideal_dcg, 4)


def reciprocal_rank(actual_doc_ids: List[int], relevant_doc_ids: Set[int]) -> float:
    """
    Computes Reciprocal Rank (1 / rank of first relevant document).
    """
    for rank, doc_id in enumerate(actual_doc_ids, start=1):
        if doc_id in relevant_doc_ids:
            return round(1.0 / rank, 4)
    return 0.0


def precision_at_k(actual_doc_ids: List[int], relevant_doc_ids: Set[int], k: int = 10) -> float:
    """Computes Precision at rank K."""
    if k <= 0:
        return 0.0
    top_k_docs = actual_doc_ids[:k]
    relevant_retrieved = sum(1 for d in top_k_docs if d in relevant_doc_ids)
    return round(relevant_retrieved / k, 4)


def recall_at_k(actual_doc_ids: List[int], relevant_doc_ids: Set[int], k: int = 10) -> float:
    """Computes Recall at rank K."""
    if not relevant_doc_ids:
        return 0.0
    top_k_docs = actual_doc_ids[:k]
    relevant_retrieved = sum(1 for d in top_k_docs if d in relevant_doc_ids)
    return round(relevant_retrieved / len(relevant_doc_ids), 4)
