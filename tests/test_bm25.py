import pytest
from neuralsearch.core.inverted_index import InvertedIndex
from neuralsearch.core.bm25 import BM25Ranker


@pytest.fixture
def sample_index():
    index = InvertedIndex()
    index.add_document(1, "The quick brown fox jumps over the lazy dog", {"title": "Doc 1"})
    index.add_document(2, "Neural search engines use dense vector embeddings and BM25", {"title": "Doc 2"})
    index.add_document(3, "Vector databases like Pinecone, Milvus and Chroma use HNSW", {"title": "Doc 3"})
    index.add_document(4, "BM25 is a ranking function used by search engines to estimate relevance", {"title": "Doc 4"})
    return index


def test_bm25_relevance(sample_index):
    ranker = BM25Ranker(sample_index)
    results = ranker.search("neural search", top_k=5)
    
    assert len(results) > 0
    # Doc 2 contains both "neural" and "search", it should be top ranked
    assert results[0].doc_id == 2
    assert "search" in results[0].matched_terms
    assert "neural" in results[0].matched_terms
    assert results[0].score > 0.0


def test_bm25_term_frequency_monotonicity():
    index = InvertedIndex()
    # Doc 1 mentions "python" once
    index.add_document(1, "Learn python programming today.")
    # Doc 2 mentions "python" three times in a similar length doc
    index.add_document(2, "Python is great. Python is fast. Python rocks.")
    # Doc 3 is unrelated
    index.add_document(3, "Cooking recipes and baking bread.")

    ranker = BM25Ranker(index)
    results = ranker.search("python")
    
    assert len(results) == 2
    assert results[0].doc_id == 2
    assert results[1].doc_id == 1
    assert results[0].score > results[1].score


def test_bm25_document_length_penalty():
    index = InvertedIndex()
    # Short concise document with query term
    index.add_document(1, "HNSW vector indexing")
    # Long diluted document with query term once and lots of fluff
    index.add_document(2, "HNSW is an algorithm with many details and extra words padding out this very long sentence about other irrelevant concepts and topics in computing.")

    ranker = BM25Ranker(index)
    results = ranker.search("HNSW")
    
    assert len(results) == 2
    # Short document should be ranked higher due to density and length normalization
    assert results[0].doc_id == 1
    assert results[0].score > results[1].score


def test_bm25_remove_document(sample_index):
    ranker = BM25Ranker(sample_index)
    initial_results = ranker.search("fox")
    assert any(r.doc_id == 1 for r in initial_results)

    # Remove doc 1
    sample_index.remove_document(1)
    new_results = ranker.search("fox")
    assert len(new_results) == 0
