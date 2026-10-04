"""
High-performance in-memory Vector Index using L2-normalized float32 embeddings
and SIMD-accelerated BLAS matrix-vector dot product for sub-millisecond retrieval.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np


@dataclass(slots=True)
class VectorResult:
    """A single vector similarity result."""
    doc_id: int
    score: float


class VectorIndex:
    """
    In-memory vector index for dense semantic embeddings.
    Embeddings are assumed to be L2-normalized so that dot product equals cosine similarity.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self._doc_ids: List[int] = []
        self._doc_id_to_idx: Dict[int, int] = {}
        # Pre-allocated dynamic matrix
        self._matrix: np.ndarray = np.empty((0, dimension), dtype=np.float32)

    @property
    def total_vectors(self) -> int:
        return len(self._doc_ids)

    def add_vector(self, doc_id: int, vector: np.ndarray) -> None:
        """
        Adds or updates a vector in the index.
        Vector must be 1D with length equal to self.dimension.
        """
        vec = np.asarray(vector, dtype=np.float32).flatten()
        if vec.shape[0] != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dimension}, got {vec.shape[0]}"
            )

        # Ensure L2 normalization
        norm = np.linalg.norm(vec)
        if norm > 1e-12:
            vec = vec / norm
        else:
            vec = np.zeros_like(vec)

        if doc_id in self._doc_id_to_idx:
            # Overwrite existing vector
            idx = self._doc_id_to_idx[doc_id]
            self._matrix[idx] = vec
        else:
            # Append new vector
            idx = len(self._doc_ids)
            self._doc_ids.append(doc_id)
            self._doc_id_to_idx[doc_id] = idx
            self._matrix = np.vstack([self._matrix, vec.reshape(1, self.dimension)])

    def remove_vector(self, doc_id: int) -> bool:
        """Removes a vector by doc_id."""
        if doc_id not in self._doc_id_to_idx:
            return False

        idx_to_remove = self._doc_id_to_idx[doc_id]
        last_idx = len(self._doc_ids) - 1

        if idx_to_remove != last_idx:
            # Swap with last element for O(1) removal
            last_doc_id = self._doc_ids[last_idx]
            self._matrix[idx_to_remove] = self._matrix[last_idx]
            self._doc_ids[idx_to_remove] = last_doc_id
            self._doc_id_to_idx[last_doc_id] = idx_to_remove

        # Pop the last element
        self._matrix = self._matrix[:-1]
        self._doc_ids.pop()
        del self._doc_id_to_idx[doc_id]

        return True

    def search(self, query_vector: np.ndarray, top_k: int = 50) -> List[VectorResult]:
        """
        Performs fast cosine similarity search using matrix-vector multiplication.
        Returns top-K results sorted by cosine similarity descending.
        """
        if self.total_vectors == 0:
            return []

        q_vec = np.asarray(query_vector, dtype=np.float32).flatten()
        if q_vec.shape[0] != self.dimension:
            raise ValueError(
                f"Query vector dimension mismatch: expected {self.dimension}, got {q_vec.shape[0]}"
            )

        # Normalize query vector
        norm = np.linalg.norm(q_vec)
        if norm > 1e-12:
            q_vec = q_vec / norm

        # Batch dot product: (N, D) @ (D,) -> (N,)
        scores = self._matrix.dot(q_vec)

        # Retrieve top-K
        k = min(top_k, len(scores))
        if k == len(scores):
            top_indices = np.argsort(-scores)
        else:
            # Use argpartition for O(N) selection followed by sorting top-K
            partitioned = np.argpartition(-scores, k)[:k]
            top_indices = partitioned[np.argsort(-scores[partitioned])]

        results: List[VectorResult] = []
        for idx in top_indices:
            results.append(
                VectorResult(
                    doc_id=self._doc_ids[idx],
                    score=round(float(scores[idx]), 4),
                )
            )

        return results

    def clear(self) -> None:
        self._doc_ids.clear()
        self._doc_id_to_idx.clear()
        self._matrix = np.empty((0, self.dimension), dtype=np.float32)
