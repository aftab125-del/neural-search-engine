"""
Benchmark evaluation harness running systematic IR quality and latency experiments.
"""

from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Dict, List, Set
from ..core.engine import HybridEngine
from .metrics import ndcg_at_k, reciprocal_rank, precision_at_k


@dataclass
class BenchmarkCase:
    query: str
    relevant_doc_ids: Dict[int, float]  # doc_id -> relevance grade (e.g. 1.0, 2.0, 3.0)


# Curated evaluation collection
BENCHMARK_DOCUMENTS = [
    (1, "B-Tree and LSM-Tree storage engines: B-Trees optimize for fast point lookups and in-place updates via balanced tree nodes, while Log-Structured Merge (LSM) trees optimize for high-throughput sequential writes using append-only commit logs and background compaction.", {"title": "Storage Engines: B-Trees vs LSM"}),
    (2, "Hierarchical Navigable Small World (HNSW) graphs: HNSW provides state-of-the-art approximate nearest neighbor search over high-dimensional vector embeddings with logarithmic time complexity using multi-layer graph skip-lists.", {"title": "HNSW Vector Indexing"}),
    (3, "Backpropagation and Automatic Differentiation: Reverse-mode automatic differentiation calculates gradients of the loss function with respect to weights using the chain rule, enabling efficient gradient descent in deep neural networks.", {"title": "Deep Learning Gradients & Autograd"}),
    (4, "FastAPI and ASGI Web Frameworks: FastAPI utilizes Starlette and Pydantic for asynchronous request handling, schema validation, and OpenAPI documentation with high concurrency on Uvicorn.", {"title": "Asynchronous Web APIs with FastAPI"}),
    (5, "Raft Distributed Consensus: Raft achieves fault-tolerant distributed state machine replication through leader election, log replication, and randomized election timeouts to prevent split-brain scenarios.", {"title": "Raft Distributed Consensus"}),
    (6, "Transformer Multi-Head Self-Attention: Scaled dot-product attention computes interactions between all pairs of input tokens with O(N^2) complexity, enabling bidirectional contextual representations in BERT and GPT architectures.", {"title": "Transformer Attention Mechanics"}),
]

BENCHMARK_QUERIES = [
    # 1. Conceptual queries (Dense & Hybrid advantage)
    BenchmarkCase(
        query="how to quickly route and search nearest embeddings in high dimensional space",
        relevant_doc_ids={2: 3.0},
    ),
    BenchmarkCase(
        query="preventing split-brain and achieving agreement in distributed systems clusters",
        relevant_doc_ids={5: 3.0},
    ),
    BenchmarkCase(
        query="calculating loss derivatives and weights optimization in neural networks",
        relevant_doc_ids={3: 3.0},
    ),
    # 2. Exact keyword and technical queries (BM25 & Hybrid advantage)
    BenchmarkCase(
        query="FastAPI Starlette Pydantic ASGI",
        relevant_doc_ids={4: 3.0},
    ),
    BenchmarkCase(
        query="Log-Structured Merge LSM-Tree append-only",
        relevant_doc_ids={1: 3.0},
    ),
    BenchmarkCase(
        query="Scaled dot-product attention O(N^2) complexity tokens",
        relevant_doc_ids={6: 3.0},
    ),
]


def run_benchmark(engine: HybridEngine | None = None) -> dict:
    """
    Executes benchmark comparison across BM25, Dense, Hybrid, and Hybrid + Reranker.
    """
    if engine is None:
        engine = HybridEngine(enable_reranker=True)

    # Index benchmark documents
    for doc_id, text, meta in BENCHMARK_DOCUMENTS:
        engine.add_document(doc_id, text, meta)

    configurations = [
        ("BM25 Lexical Only", {"mode": "bm25", "rerank": False}),
        ("Dense Semantic Only", {"mode": "dense", "rerank": False}),
        ("Hybrid (BM25 + Dense RRF)", {"mode": "hybrid", "rerank": False}),
        ("Hybrid + Cross-Encoder Reranker", {"mode": "hybrid", "rerank": True}),
    ]

    results = []

    for name, params in configurations:
        ndcg_scores = []
        mrr_scores = []
        p_scores = []
        latencies = []

        for case in BENCHMARK_QUERIES:
            t0 = time.perf_counter()
            resp = engine.search(
                query=case.query,
                mode=params["mode"],
                top_k=5,
                rerank=params["rerank"],
            )
            latencies.append((time.perf_counter() - t0) * 1000.0)

            retrieved_ids = [hit.doc_id for hit in resp.hits]
            relevant_set = set(case.relevant_doc_ids.keys())

            ndcg = ndcg_at_k(retrieved_ids, case.relevant_doc_ids, k=5)
            mrr = reciprocal_rank(retrieved_ids, relevant_set)
            prec = precision_at_k(retrieved_ids, relevant_set, k=1)

            ndcg_scores.append(ndcg)
            mrr_scores.append(mrr)
            p_scores.append(prec)

        avg_ndcg = round(sum(ndcg_scores) / len(ndcg_scores), 4)
        avg_mrr = round(sum(mrr_scores) / len(mrr_scores), 4)
        avg_p1 = round(sum(p_scores) / len(p_scores), 4)
        avg_lat = round(sum(latencies) / len(latencies), 2)
        p95_lat = round(sorted(latencies)[int(len(latencies) * 0.95)], 2)

        results.append({
            "strategy": name,
            "ndcg_at_5": avg_ndcg,
            "mrr": avg_mrr,
            "precision_at_1": avg_p1,
            "mean_latency_ms": avg_lat,
            "p95_latency_ms": p95_lat,
        })

    return {"results": results}


def format_markdown_table(benchmark_output: dict) -> str:
    """Formats benchmark results as a clean Markdown table for README and reports."""
    lines = [
        "| Retrieval Strategy | NDCG@5 | MRR | Precision@1 | Mean Latency | P95 Latency |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for row in benchmark_output["results"]:
        lines.append(
            f"| **{row['strategy']}** | `{row['ndcg_at_5']:.4f}` | `{row['mrr']:.4f}` | `{row['precision_at_1']:.4f}` | `{row['mean_latency_ms']}ms` | `{row['p95_latency_ms']}ms` |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    print("Running NeuralSearch IR Benchmark...")
    res = run_benchmark()
    print("\n" + format_markdown_table(res))
