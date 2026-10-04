import tempfile
from pathlib import Path
import pytest
from neuralsearch.core.engine import HybridEngine
from neuralsearch.storage.chunker import MarkdownChunker
from neuralsearch.storage.sqlite_store import SQLiteStore
from neuralsearch.storage.ingestor import IngestionManager


def test_markdown_chunker_headings():
    chunker = MarkdownChunker(max_tokens=50, min_tokens=10)
    markdown_sample = """# System Overview
Here is a high level overview of the search engine.

## Retrieval Subsystem
The retrieval subsystem runs BM25 and vector search in parallel.

### BM25 Details
BM25 uses an inverted index with postings lists.
"""
    chunks = chunker.chunk_document(markdown_sample, document_title="Architecture")
    assert len(chunks) >= 2
    assert any("Retrieval Subsystem" in c.headings for c in chunks)
    assert any("BM25 Details" in c.headings for c in chunks)


def test_sqlite_store_and_ingestion():
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        db_file = tmp_path / "test.db"
        store = SQLiteStore(db_path=db_file)
        engine = HybridEngine(enable_reranker=False)
        ingestor = IngestionManager(engine=engine, store=store)

        # Create sample files
        doc1 = tmp_path / "doc1.md"
        doc1.write_text("# Deep Learning\nNeural networks learn representations from data.")

        doc2 = tmp_path / "doc2.txt"
        doc2.write_text("Linear algebra forms the mathematical basis of machine learning.")

        # Ingest directory
        res = ingestor.ingest_directory(tmp_path)
        assert res["indexed_files"] == 2
        assert res["total_chunks_created"] >= 2

        # Second ingestion should skip because hash is unchanged
        res2 = ingestor.ingest_directory(tmp_path)
        assert res2["indexed_files"] == 0
        assert res2["skipped_files"] == 2

        # Test query
        search_res = engine.search("neural networks", mode="hybrid")
        assert search_res.total_hits > 0
        assert "Deep Learning" in search_res.hits[0].content

        # Test rehydration into a fresh engine
        fresh_engine = HybridEngine(enable_reranker=False)
        fresh_ingestor = IngestionManager(engine=fresh_engine, store=store)
        restored = fresh_ingestor.rehydrate_engine()
        assert restored >= 2

        fresh_res = fresh_engine.search("neural networks", mode="hybrid")
        assert fresh_res.total_hits > 0
