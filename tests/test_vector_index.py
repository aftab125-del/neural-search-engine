import numpy as np
import pytest
from neuralsearch.core.vector_index import VectorIndex


def test_vector_index_add_and_search():
    index = VectorIndex(dimension=4)
    
    # 4 distinct orthogonal-ish vectors
    v1 = np.array([1.0, 0.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0, 0.0])
    v3 = np.array([0.9, 0.1, 0.0, 0.0])  # Very close to v1
    v4 = np.array([0.0, 0.0, 1.0, 0.0])

    index.add_vector(1, v1)
    index.add_vector(2, v2)
    index.add_vector(3, v3)
    index.add_vector(4, v4)

    assert index.total_vectors == 4

    # Search with query identical to v1
    results = index.search(v1, top_k=2)
    assert len(results) == 2
    assert results[0].doc_id == 1
    assert pytest.approx(results[0].score, 0.001) == 1.0
    assert results[1].doc_id == 3  # v3 is next closest


def test_vector_index_remove():
    index = VectorIndex(dimension=3)
    v1 = np.array([1.0, 0.0, 0.0])
    v2 = np.array([0.0, 1.0, 0.0])

    index.add_vector(10, v1)
    index.add_vector(20, v2)
    assert index.total_vectors == 2

    removed = index.remove_vector(10)
    assert removed is True
    assert index.total_vectors == 1

    results = index.search(v1, top_k=5)
    assert len(results) == 1
    assert results[0].doc_id == 20


def test_vector_dimension_mismatch():
    index = VectorIndex(dimension=3)
    with pytest.raises(ValueError):
        index.add_vector(1, np.array([1.0, 2.0]))
