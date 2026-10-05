"""
Persistent SQLite Storage Layer with WAL mode and binary vector serialization.
Enables fast rehydration of the search engine on application startup.
"""

from __future__ import annotations
from contextlib import contextmanager
import hashlib
from pathlib import Path
import sqlite3
from typing import Generator, List, Optional, Tuple
import numpy as np
from .chunker import Chunk


import os

BASE_DIR = Path(__file__).resolve().parents[3]
DEFAULT_DB_PATH = (
    Path("/tmp/index.db")
    if bool(os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"))
    else BASE_DIR / "data" / "index.db"
)


class SQLiteStore:
    """
    Embedded relational and vector blob storage manager.
    """

    def __init__(self, db_path: Path | str | None = None):
        if db_path is not None:
            self.db_path = Path(db_path)
        else:
            self.db_path = DEFAULT_DB_PATH
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            self.db_path = Path("/tmp/index.db")
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path), timeout=30.0)
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        conn.execute("PRAGMA foreign_keys = ON;")
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    file_path TEXT UNIQUE NOT NULL,
                    file_name TEXT NOT NULL,
                    file_hash TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    indexed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    token_count INTEGER NOT NULL,
                    headings TEXT,
                    start_line INTEGER,
                    end_line INTEGER
                );

                CREATE TABLE IF NOT EXISTS chunk_vectors (
                    chunk_id INTEGER PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
                    vector_blob BLOB NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);
                """
            )

    @staticmethod
    def compute_file_hash(content_bytes: bytes) -> str:
        return hashlib.sha256(content_bytes).hexdigest()

    def get_document_by_path(self, file_path: str) -> Optional[dict]:
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE file_path = ?", (file_path,)
            ).fetchone()
            return dict(row) if row else None

    def delete_document(self, file_path: str) -> bool:
        with self._get_connection() as conn:
            cursor = conn.execute("DELETE FROM documents WHERE file_path = ?", (file_path,))
            return cursor.rowcount > 0

    def save_document(
        self,
        file_path: str,
        file_name: str,
        file_hash: str,
        file_size: int,
        chunks: List[Chunk],
        embeddings: np.ndarray,
    ) -> List[int]:
        """
        Saves document, its chunks, and corresponding vector blobs.
        Returns the list of generated chunk IDs.
        """
        with self._get_connection() as conn:
            # Remove old version if present
            conn.execute("DELETE FROM documents WHERE file_path = ?", (file_path,))

            # Insert document
            cursor = conn.execute(
                """
                INSERT INTO documents (file_path, file_name, file_hash, file_size)
                VALUES (?, ?, ?, ?)
                """,
                (file_path, file_name, file_hash, file_size),
            )
            doc_id = cursor.lastrowid

            chunk_ids: List[int] = []
            for i, chunk in enumerate(chunks):
                c_cursor = conn.execute(
                    """
                    INSERT INTO chunks (document_id, chunk_index, content, token_count, headings, start_line, end_line)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        doc_id,
                        chunk.chunk_index,
                        chunk.content,
                        chunk.token_count,
                        " > ".join(chunk.headings),
                        chunk.start_line,
                        chunk.end_line,
                    ),
                )
                chunk_id = c_cursor.lastrowid
                chunk_ids.append(chunk_id)

                # Store embedding as raw float32 bytes
                vec = embeddings[i].astype(np.float32).tobytes()
                conn.execute(
                    "INSERT INTO chunk_vectors (chunk_id, vector_blob) VALUES (?, ?)",
                    (chunk_id, vec),
                )

            return chunk_ids

    def load_all_chunks(self) -> List[Tuple[int, str, dict, np.ndarray]]:
        """
        Loads all chunks and embeddings from SQLite for fast engine hydration.
        Returns: list of (chunk_id, content, metadata, embedding_vector)
        """
        items = []
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT c.id, c.content, c.token_count, c.headings, c.start_line, c.end_line,
                       d.file_path, d.file_name, v.vector_blob
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                JOIN chunk_vectors v ON c.id = v.chunk_id
                ORDER BY c.id ASC
                """
            )
            for row in cursor:
                chunk_id = row["id"]
                content = row["content"]
                metadata = {
                    "title": row["file_name"],
                    "file_path": row["file_path"],
                    "headings": row["headings"],
                    "start_line": row["start_line"],
                    "end_line": row["end_line"],
                    "content": content,
                }
                vec = np.frombuffer(row["vector_blob"], dtype=np.float32)
                items.append((chunk_id, content, metadata, vec))
        return items

    def get_stats(self) -> dict:
        with self._get_connection() as conn:
            doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            size_bytes = self.db_path.stat().st_size if self.db_path.exists() else 0
            return {
                "total_documents": doc_count,
                "total_chunks": chunk_count,
                "db_size_mb": round(size_bytes / (1024 * 1024), 2),
            }
