"""
Command Line Interface (CLI) for NeuralSearch.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
import uvicorn
from .eval.benchmark import run_benchmark, format_markdown_table
from .storage.ingestor import IngestionManager
from .storage.sqlite_store import SQLiteStore
from .core.engine import HybridEngine

BASE_DIR = Path(__file__).resolve().parents[2]
DB_PATH = BASE_DIR / "data" / "index.db"


def main():
    parser = argparse.ArgumentParser(
        prog="neuralsearch",
        description="Local-First Hybrid Neural Search Engine CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Command: serve
    serve_parser = subparsers.add_parser("serve", help="Start the FastAPI web server & UI")
    serve_parser.add_argument("--host", default="127.0.0.1", help="Host address")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port number")
    serve_parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    # Command: index
    index_parser = subparsers.add_parser("index", help="Index a file or directory")
    index_parser.add_argument("path", help="Path to file or folder")
    index_parser.add_argument("--no-recursive", action="store_true", help="Do not index recursively")

    # Command: search
    search_parser = subparsers.add_parser("search", help="Execute search from terminal")
    search_parser.add_argument("query", help="Query string")
    search_parser.add_argument("--mode", choices=["hybrid", "bm25", "dense"], default="hybrid")
    search_parser.add_argument("--top-k", type=int, default=5)

    # Command: evaluate
    subparsers.add_parser("evaluate", help="Run quantitative IR benchmark (NDCG@5, MRR)")

    # Command: stats
    subparsers.add_parser("stats", help="Display index and database telemetry")

    args = parser.parse_args()

    if args.command == "serve":
        print(f"\n[NeuralSearch] Launching server on http://{args.host}:{args.port}")
        uvicorn.run("neuralsearch.api.app:app", host=args.host, port=args.port, reload=args.reload)

    elif args.command == "index":
        target = Path(args.path).resolve()
        if not target.exists():
            print(f"Error: Target path does not exist: {target}")
            sys.exit(1)

        store = SQLiteStore(db_path=DB_PATH)
        engine = HybridEngine(enable_reranker=False)
        ingestor = IngestionManager(engine=engine, store=store)

        if target.is_file():
            print(f"Indexing file: {target}")
            res = ingestor.ingest_file(target)
            print(f"Result: {res}")
        else:
            print(f"Scanning directory: {target} (recursive={not args.no_recursive})")
            res = ingestor.ingest_directory(target, recursive=not args.no_recursive)
            print(f"Indexed files: {res['indexed_files']}")
            print(f"Skipped files: {res['skipped_files']}")
            print(f"Total chunks created: {res['total_chunks_created']}")

    elif args.command == "search":
        store = SQLiteStore(db_path=DB_PATH)
        engine = HybridEngine(enable_reranker=True)
        ingestor = IngestionManager(engine=engine, store=store)
        restored = ingestor.rehydrate_engine()

        if restored == 0:
            print("Warning: Index is empty! Use `neuralsearch index <dir>` first.")
            sys.exit(0)

        res = engine.search(query=args.query, mode=args.mode, top_k=args.top_k)
        print(f"\nQuery: '{args.query}' (Mode: {args.mode})")
        print(f"Latency: {res.telemetry.total_ms}ms (BM25: {res.telemetry.bm25_ms}ms, Embed: {res.telemetry.embed_ms}ms, Rerank: {res.telemetry.rerank_ms}ms)")
        print("-" * 60)
        for hit in res.hits:
            print(f"Rank #{hit.rank} [Score: {hit.score}] {hit.title}")
            print(f"  Excerpt: {hit.content[:140]}...")
            print()

    elif args.command == "evaluate":
        print("Running quantitative Information Retrieval benchmark...")
        res = run_benchmark()
        print("\n" + format_markdown_table(res))

    elif args.command == "stats":
        store = SQLiteStore(db_path=DB_PATH)
        stats = store.get_stats()
        print("\n--- NeuralSearch Database Statistics ---")
        for k, v in stats.items():
            print(f"{k}: {v}")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
