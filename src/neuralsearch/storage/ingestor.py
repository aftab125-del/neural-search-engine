"""
Ingestion Manager handling directory walking, file hashing for incremental indexing,
chunking, embedding batching, and index rehydration.
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import List, Optional
from .chunker import MarkdownChunker
from .sqlite_store import SQLiteStore
from ..core.engine import HybridEngine

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {
    ".md", ".txt", ".markdown", ".py", ".json", ".rst", ".html",
    ".ts", ".tsx", ".js", ".jsx", ".css", ".sql", ".yaml", ".yml",
    ".java", ".cpp", ".c", ".h", ".rs", ".go"
}


IGNORE_DIRS = {
    ".venv", "venv", "env", ".git", ".cache", "__pycache__",
    "node_modules", "dist", "build", "data", ".pytest_cache", ".idea", ".vscode"
}


class IngestionManager:
    """
    Orchestrates file reading, incremental hashing, chunking, and dual-index updates.
    """

    def __init__(
        self,
        engine: HybridEngine,
        store: SQLiteStore,
        chunker: Optional[MarkdownChunker] = None,
    ):
        self.engine = engine
        self.store = store
        self.chunker = chunker or MarkdownChunker()

    def rehydrate_engine(self) -> int:
        """
        Loads all persistent chunks from SQLite directly into the in-memory engine.
        Returns the number of chunks restored.
        """
        records = self.store.load_all_chunks()
        for chunk_id, content, metadata, vec in records:
            # Add to lexical inverted index
            self.engine.inverted_index.add_document(chunk_id, content, metadata)
            # Add to dense vector index
            self.engine.vector_index.add_vector(chunk_id, vec)
        return len(records)

    def ingest_file(self, file_path: Path | str) -> dict:
        """
        Ingests an individual file incrementally.
        """
        path = Path(file_path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        try:
            content_bytes = path.read_bytes()
            content_text = content_bytes.decode("utf-8", errors="replace")
        except Exception as e:
            return {"status": "error", "file": str(path), "error": str(e)}

        file_hash = self.store.compute_file_hash(content_bytes)
        file_size = len(content_bytes)

        # Check existing hash
        existing = self.store.get_document_by_path(str(path))
        if existing and existing["file_hash"] == file_hash:
            return {"status": "skipped", "file": str(path), "reason": "content unchanged"}

        # 1. Chunk document
        chunks = self.chunker.chunk_document(content_text, document_title=path.name)
        if not chunks:
            return {"status": "skipped", "file": str(path), "reason": "empty content"}

        # 2. Batch embed chunks
        texts_to_embed = [c.content for c in chunks]
        embeddings = self.engine.dense_encoder.encode_batch(texts_to_embed)

        # 3. Save to SQLite
        chunk_ids = self.store.save_document(
            file_path=str(path),
            file_name=path.name,
            file_hash=file_hash,
            file_size=file_size,
            chunks=chunks,
            embeddings=embeddings,
        )

        # 4. Update in-memory search engine
        for i, chunk_id in enumerate(chunk_ids):
            chunk = chunks[i]
            meta = {
                "title": path.name,
                "file_path": str(path),
                "headings": " > ".join(chunk.headings),
                "start_line": chunk.start_line,
                "end_line": chunk.end_line,
                "content": chunk.content,
            }
            self.engine.inverted_index.add_document(chunk_id, chunk.content, meta)
            self.engine.vector_index.add_vector(chunk_id, embeddings[i])

        return {
            "status": "indexed",
            "file": str(path),
            "chunks_count": len(chunks),
            "chunk_ids": chunk_ids,
        }

    def ingest_directory(
        self,
        directory_path: Path | str,
        recursive: bool = True,
    ) -> dict:
        """
        Recursively scans and ingests all supported files in a directory, ignoring virtualenvs and caches.
        """
        root = Path(directory_path).resolve()
        if not root.is_dir():
            raise NotADirectoryError(f"Directory not found: {root}")

        pattern = "**/*" if recursive else "*"
        indexed_count = 0
        skipped_count = 0
        total_chunks = 0
        errors = []

        for p in root.glob(pattern):
            if not p.is_file():
                continue

            # Skip ignored directories and hidden system folders
            try:
                rel_parts = p.relative_to(root).parts
                if any(part in IGNORE_DIRS or (part.startswith(".") and part != ".") for part in rel_parts[:-1]):
                    continue
            except ValueError:
                pass

            if p.suffix.lower() in SUPPORTED_EXTENSIONS:
                res = self.ingest_file(p)
                if res["status"] == "indexed":
                    indexed_count += 1
                    total_chunks += res.get("chunks_count", 0)
                elif res["status"] == "skipped":
                    skipped_count += 1
                elif res["status"] == "error":
                    errors.append(res)

        return {
            "directory": str(root),
            "indexed_files": indexed_count,
            "skipped_files": skipped_count,
            "total_chunks_created": total_chunks,
            "errors": errors,
        }
