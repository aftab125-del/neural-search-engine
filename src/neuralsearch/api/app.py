"""
FastAPI REST and UI server for NeuralSearch.
"""

from __future__ import annotations
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from ..core.engine import HybridEngine
from ..core.highlighter import SnippetHighlighter
from ..storage.ingestor import IngestionManager
from ..storage.sqlite_store import SQLiteStore

# Paths
BASE_DIR = Path("D:/neural-search-engine")
DB_PATH = BASE_DIR / "data" / "index.db"
UI_PATH = Path(__file__).parent.parent / "ui" / "index.html"

# Global state
store = SQLiteStore(db_path=DB_PATH)
engine = HybridEngine(enable_reranker=True)
ingestor = IngestionManager(engine=engine, store=store)
highlighter = SnippetHighlighter()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Rehydrate in-memory engine from SQLite
    restored = ingestor.rehydrate_engine()
    print(f"[NeuralSearch] Rehydrated {restored} chunks from persistent store.")

    # If the database is completely empty, automatically seed with project documentation!
    if restored == 0:
        seed_files = [
            BASE_DIR / "PRD.md",
            BASE_DIR / "ARCHITECTURE.md",
            BASE_DIR / "DESIGN.md",
            BASE_DIR / "PHASES.md",
        ]
        for f in seed_files:
            if f.exists():
                ingestor.ingest_file(f)
        print(f"[NeuralSearch] Seeded database with project documentation ({len(seed_files)} files).")

    yield
    print("[NeuralSearch] Shutting down cleanly.")


app = FastAPI(
    title="NeuralSearch API",
    description="Local-First Hybrid Neural Search Engine",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class IndexDirectoryRequest(BaseModel):
    directory_path: str
    recursive: bool = True


class IndexFileRequest(BaseModel):
    file_path: str


@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serves the light minimalist search web UI."""
    if not UI_PATH.exists():
        raise HTTPException(status_code=404, detail="UI index.html not found")
    return HTMLResponse(content=UI_PATH.read_text(encoding="utf-8"))


@app.get("/api/search")
async def search_endpoint(
    q: str = Query(..., description="Query search string"),
    mode: str = Query("web", description="Retrieval mode: web | local | bm25 | dense"),
    category: str = Query("all", description="Web search category: all | tech | news"),
    top_k: int = Query(15, ge=1, le=50, description="Max results to return"),
    rerank: bool = Query(True, description="Enable cross-encoder neural reranker"),
):
    """
    Executes live private web search or local hybrid search with privacy audit telemetry.
    """
    if not q.strip():
        return {
            "query": q,
            "mode": mode,
            "category": category,
            "total_hits": 0,
            "hits": [],
            "instant_answer": None,
            "ads_blocked_count": 0,
            "trackers_purged_count": 0,
            "telemetry": {
                "bm25_ms": 0.0,
                "embed_ms": 0.0,
                "vector_search_ms": 0.0,
                "fusion_ms": 0.0,
                "rerank_ms": 0.0,
                "total_ms": 0.0,
            },
        }

    # 1. LIVE INTERNET WEB SEARCH (Default)
    if mode == "web":
        web_resp = await engine.search_web(query=q, category=category, top_k=top_k, rerank=rerank)
        query_terms = engine.inverted_index.tokenizer.tokenize_terms(q)

        formatted_hits = []
        for hit in web_resp.hits:
            highlighted_snippet = highlighter.highlight(hit.snippet, query_terms)
            formatted_hits.append({
                "rank": hit.rank,
                "score": hit.score,
                "title": hit.title,
                "url": hit.url,
                "display_url": hit.display_url,
                "domain": hit.domain,
                "snippet": highlighted_snippet,
                "trackers_purged": hit.trackers_purged,
                "is_organic": hit.is_organic,
                "provenance": {
                    "rerank_score": hit.rerank_score,
                    "domain": hit.domain,
                    "clean_url": hit.url,
                },
            })

        instant_answer_dict = None
        if web_resp.instant_answer:
            instant_answer_dict = {
                "title": web_resp.instant_answer.title,
                "extract": web_resp.instant_answer.extract,
                "url": web_resp.instant_answer.url,
                "source": web_resp.instant_answer.source,
            }

        return {
            "query": web_resp.query,
            "mode": "web",
            "category": web_resp.category,
            "total_hits": len(formatted_hits),
            "hits": formatted_hits,
            "instant_answer": instant_answer_dict,
            "ads_blocked_count": web_resp.ads_blocked_count,
            "trackers_purged_count": web_resp.trackers_purged_count,
            "telemetry": {
                "bm25_ms": 0.0,
                "embed_ms": web_resp.telemetry.embed_ms,  # Web fetch time
                "vector_search_ms": 0.0,
                "fusion_ms": 0.0,
                "rerank_ms": web_resp.telemetry.rerank_ms,
                "total_ms": web_resp.telemetry.total_ms,
            },
        }

    # 2. LOCAL FILE SEARCH (Optional Fallback)
    response = engine.search(query=q, mode=mode if mode in ("hybrid", "bm25", "dense") else "hybrid", top_k=top_k, rerank=rerank)
    query_terms = engine.inverted_index.tokenizer.tokenize_terms(q)
    results = []
    for hit in response.hits:
        snippet = highlighter.extract_snippet(hit.content, query_terms)
        results.append({
            "doc_id": hit.doc_id,
            "rank": hit.rank,
            "score": hit.score,
            "title": hit.title,
            "url": "#",
            "display_url": hit.title,
            "domain": "local",
            "snippet": snippet,
            "content": hit.content,
            "metadata": hit.metadata,
            "provenance": hit.provenance,
        })

    return {
        "query": response.query,
        "mode": response.mode,
        "category": "local",
        "total_hits": len(results),
        "hits": results,
        "instant_answer": None,
        "ads_blocked_count": 0,
        "trackers_purged_count": 0,
        "telemetry": {
            "bm25_ms": response.telemetry.bm25_ms,
            "embed_ms": response.telemetry.embed_ms,
            "vector_search_ms": response.telemetry.vector_search_ms,
            "fusion_ms": response.telemetry.fusion_ms,
            "rerank_ms": response.telemetry.rerank_ms,
            "total_ms": response.telemetry.total_ms,
        },
    }


@app.post("/api/index/directory")
async def index_directory(req: IndexDirectoryRequest):
    """Recursively scans and indexes a folder."""
    target_path = Path(req.directory_path)
    if not target_path.exists():
        raise HTTPException(status_code=400, detail=f"Directory does not exist: {req.directory_path}")
    
    result = ingestor.ingest_directory(target_path, recursive=req.recursive)
    return result


@app.post("/api/index/file")
async def index_file(req: IndexFileRequest):
    """Indexes a single file."""
    target_file = Path(req.file_path)
    if not target_file.exists():
        raise HTTPException(status_code=400, detail=f"File does not exist: {req.file_path}")

    result = ingestor.ingest_file(target_file)
    return result


@app.get("/api/stats")
async def get_stats():
    """Returns database and memory index telemetry."""
    db_stats = store.get_stats()
    return {
        **db_stats,
        "vocabulary_size": engine.inverted_index.vocabulary_size,
        "total_in_memory_vectors": engine.vector_index.total_vectors,
        "avg_doc_length": round(engine.inverted_index.avg_doc_length, 1),
    }


@app.get("/api/health")
async def health():
    return {
        "status": "healthy",
        "models": {
            "bi_encoder": "all-MiniLM-L6-v2 (quantized ONNX)",
            "reranker": "ms-marco-MiniLM-L-6-v2 (quantized ONNX)",
        },
    }
