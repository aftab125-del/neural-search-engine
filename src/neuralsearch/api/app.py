"""
FastAPI REST and UI server for NeuralSearch.
Multi-modal search engine supporting All Web, Images, Videos, News, Shopping, Photo Search, and Autocomplete.
"""

from __future__ import annotations
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
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

    # Seed documentation if empty
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
    description="Privacy-Preserving Multi-Modal Search Engine with On-Device Neural Reranking",
    version="2.0.0",
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
    category: str = Query("all", description="Search category: all | images | videos | news | shopping | tech | local"),
    mode: str = Query("web", description="Retrieval mode: web | local | hybrid"),
    top_k: int = Query(15, ge=1, le=60, description="Max results to return"),
    rerank: bool = Query(True, description="Enable cross-encoder neural reranker"),
):
    """
    Unified multi-modal search endpoint supporting All Web, Images, Videos, News, Shopping, and Local Docs.
    """
    start_time = time.perf_counter()
    clean_q = q.strip()

    if not clean_q:
        return {
            "query": q,
            "category": category,
            "mode": mode,
            "total_hits": 0,
            "hits": [],
            "instant_answer": None,
            "ads_blocked_count": 0,
            "trackers_purged_count": 0,
            "telemetry": {"total_ms": 0.0},
        }

    # 1. IMAGES SEARCH
    if category == "images":
        img_results = await engine.web_fetcher.fetch_image_results(query=clean_q, top_k=top_k * 2)
        elapsed = round((time.perf_counter() - start_time) * 1000.0, 2)
        hits = [
            {
                "type": "image",
                "title": img.title,
                "image_url": img.image_url,
                "thumb_url": img.thumb_url,
                "source_url": img.source_url,
                "domain": img.domain,
                "width": img.width,
                "height": img.height,
            }
            for img in img_results
        ]
        return {
            "query": clean_q,
            "category": "images",
            "total_hits": len(hits),
            "hits": hits,
            "instant_answer": None,
            "ads_blocked_count": 0,
            "trackers_purged_count": len(hits),
            "telemetry": {"fetch_ms": elapsed, "total_ms": elapsed},
        }

    # 2. VIDEOS SEARCH
    if category == "videos":
        vid_results = await engine.web_fetcher.fetch_video_results(query=clean_q, top_k=top_k)
        elapsed = round((time.perf_counter() - start_time) * 1000.0, 2)
        hits = [
            {
                "type": "video",
                "title": v.title,
                "url": v.url,
                "thumb_url": v.thumb_url,
                "platform": v.platform,
                "duration": v.duration,
                "channel": v.channel,
            }
            for v in vid_results
        ]
        return {
            "query": clean_q,
            "category": "videos",
            "total_hits": len(hits),
            "hits": hits,
            "instant_answer": None,
            "ads_blocked_count": 0,
            "trackers_purged_count": len(hits),
            "telemetry": {"fetch_ms": elapsed, "total_ms": elapsed},
        }

    # 3. NEWS SEARCH
    if category == "news":
        news_results = await engine.web_fetcher.fetch_news_results(query=clean_q, top_k=top_k)
        elapsed = round((time.perf_counter() - start_time) * 1000.0, 2)
        hits = [
            {
                "type": "news",
                "title": n.title,
                "url": n.url,
                "source": n.source,
                "source_url": n.source_url,
                "pub_date": n.pub_date,
                "snippet": n.snippet,
            }
            for n in news_results
        ]
        return {
            "query": clean_q,
            "category": "news",
            "total_hits": len(hits),
            "hits": hits,
            "instant_answer": None,
            "ads_blocked_count": 0,
            "trackers_purged_count": len(hits),
            "telemetry": {"fetch_ms": elapsed, "total_ms": elapsed},
        }

    # 4. SHOPPING SEARCH
    if category == "shopping":
        shop_results = await engine.web_fetcher.fetch_shopping_results(query=clean_q, top_k=top_k)
        elapsed = round((time.perf_counter() - start_time) * 1000.0, 2)
        hits = [
            {
                "type": "shopping",
                "title": p.title,
                "url": p.url,
                "store": p.store,
                "price": p.price,
                "snippet": p.snippet,
                "domain": p.domain,
                "thumb_url": p.thumb_url,
                "rating": p.rating,
            }
            for p in shop_results
        ]
        return {
            "query": clean_q,
            "category": "shopping",
            "total_hits": len(hits),
            "hits": hits,
            "instant_answer": None,
            "ads_blocked_count": 0,
            "trackers_purged_count": len(hits),
            "telemetry": {"fetch_ms": elapsed, "total_ms": elapsed},
        }

    # 5. LOCAL FILE SEARCH
    if category == "local" or mode == "local":
        response = engine.search(query=clean_q, mode="hybrid", top_k=top_k, rerank=rerank)
        query_terms = engine.inverted_index.tokenizer.tokenize_terms(clean_q)
        results = []
        for hit in response.hits:
            snippet = highlighter.extract_snippet(hit.content, query_terms)
            results.append({
                "type": "local",
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
            "category": "local",
            "mode": "hybrid",
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

    # 6. ALL WEB SEARCH (Default with Neural Cross-Encoder Reranker)
    web_resp = await engine.search_web(query=clean_q, category=category, top_k=top_k, rerank=rerank)
    query_terms = engine.inverted_index.tokenizer.tokenize_terms(clean_q)

    formatted_hits = []
    for hit in web_resp.hits:
        highlighted_snippet = highlighter.highlight(hit.snippet, query_terms)
        formatted_hits.append({
            "type": "web",
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
        "category": "all",
        "mode": "web",
        "total_hits": len(formatted_hits),
        "hits": formatted_hits,
        "instant_answer": instant_answer_dict,
        "ads_blocked_count": web_resp.ads_blocked_count,
        "trackers_purged_count": web_resp.trackers_purged_count,
        "telemetry": {
            "bm25_ms": 0.0,
            "embed_ms": web_resp.telemetry.embed_ms,
            "vector_search_ms": 0.0,
            "fusion_ms": 0.0,
            "rerank_ms": web_resp.telemetry.rerank_ms,
            "total_ms": web_resp.telemetry.total_ms,
        },
    }


@app.get("/api/autocomplete")
async def autocomplete_endpoint(q: str = Query("", description="Query prefix")):
    """Returns fast search suggestions as the user types."""
    suggestions = await engine.web_fetcher.fetch_autocomplete(q)
    return {"query": q, "suggestions": suggestions}


@app.post("/api/search/photo")
async def photo_search_endpoint(
    image_url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    """
    Visual Photo Search endpoint.
    Accepts an uploaded image file or public image URL.
    Returns visual matches, computer vision metrics, and safe reverse image search links.
    """
    img_bytes = None
    filename = "photo.jpg"
    if file:
        img_bytes = await file.read()
        filename = file.filename or "photo.jpg"

    if not image_url and not img_bytes:
        raise HTTPException(status_code=400, detail="Must provide either an image file upload or an image_url")

    result = await engine.web_fetcher.fetch_visual_search(
        image_url=image_url,
        image_bytes=img_bytes,
        filename=filename,
    )

    return {
        "source_image": result.source_image,
        "detected_info": result.detected_info,
        "visual_matches": result.visual_matches,
        "reverse_search_links": result.reverse_search_links,
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
        "features": ["all", "images", "videos", "news", "shopping", "photo_search", "autocomplete"],
    }
