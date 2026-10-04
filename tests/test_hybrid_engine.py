import pytest
from neuralsearch.core.engine import HybridEngine


@pytest.fixture(scope="module")
def hybrid_engine():
    engine = HybridEngine()
    engine.add_document(
        1,
        "Python is a versatile programming language widely used in machine learning, artificial intelligence, and data science.",
        {"title": "Introduction to Python"},
    )
    engine.add_document(
        2,
        "FastAPI is a modern, high-performance web framework for building APIs with Python based on standard Python type hints.",
        {"title": "FastAPI Web Framework"},
    )
    engine.add_document(
        3,
        "PostgreSQL is a powerful, open source object-relational database system with advanced indexing capabilities like B-Tree and GIN.",
        {"title": "PostgreSQL Database"},
    )
    engine.add_document(
        4,
        "Hierarchical Navigable Small World (HNSW) graphs allow logarithmic time approximate nearest neighbor search over dense vector embeddings.",
        {"title": "HNSW Vector Indexing"},
    )
    return engine


def test_hybrid_search_conceptual(hybrid_engine):
    # Conceptual query that matches HNSW vector concepts even without exact words
    response = hybrid_engine.search("how to quickly find nearest embeddings in high dimensional space", mode="hybrid")
    
    assert response.total_hits > 0
    # Top result should be doc 4 (HNSW Vector Indexing)
    assert response.hits[0].doc_id == 4
    assert response.telemetry.total_ms > 0
    assert response.hits[0].rank == 1


def test_hybrid_search_exact_keyword(hybrid_engine):
    # Query with exact keyword "FastAPI" and "type hints"
    response = hybrid_engine.search("FastAPI type hints", mode="hybrid")
    
    assert response.total_hits > 0
    assert response.hits[0].doc_id == 2


def test_modes_telemetry(hybrid_engine):
    res_bm25 = hybrid_engine.search("PostgreSQL", mode="bm25")
    assert res_bm25.telemetry.bm25_ms >= 0
    assert res_bm25.telemetry.vector_search_ms == 0.0

    res_dense = hybrid_engine.search("PostgreSQL", mode="dense")
    assert res_dense.telemetry.vector_search_ms >= 0
    assert res_dense.telemetry.bm25_ms == 0.0
