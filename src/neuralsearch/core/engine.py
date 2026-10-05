"""
Unified Hybrid Search Engine coordinating Lexical BM25, Dense Vector Retrieval,
Reciprocal Rank Fusion, and Cross-Encoder Reranking with microsecond telemetry.
"""

from __future__ import annotations
import asyncio
import time
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional
from .bm25 import BM25Ranker, BM25Result
from .dense_encoder import DenseEncoder
from .fusion import FusedCandidate, reciprocal_rank_fusion
from .inverted_index import InvertedIndex
from .reranker import CrossEncoderReranker
from .vector_index import VectorIndex
from ..web.fetcher import WebSearchFetcher, WebSearchResult, InstantAnswer


@dataclass(slots=True)
class WebSearchResponse:
    """Live web search response with organic results, instant answers, and privacy metrics."""
    query: str
    category: str
    total_results: int
    hits: List[WebSearchResult]
    instant_answer: Optional[InstantAnswer]
    ads_blocked_count: int
    trackers_purged_count: int
    telemetry: SearchTelemetry


@dataclass(slots=True)
class SearchHit:
    """Final unified search result item."""
    doc_id: int
    score: float
    rank: int
    title: str
    content: str
    metadata: dict = field(default_factory=dict)
    provenance: dict = field(default_factory=dict)  # Details on BM25, Vector, and Rerank stages


@dataclass(slots=True)
class SearchTelemetry:
    """Latency breakdown across all retrieval stages in milliseconds."""
    bm25_ms: float = 0.0
    embed_ms: float = 0.0
    vector_search_ms: float = 0.0
    fusion_ms: float = 0.0
    rerank_ms: float = 0.0
    total_ms: float = 0.0


@dataclass(slots=True)
class SearchResponse:
    """Full search response containing hits, latency breakdown, and query metadata."""
    query: str
    mode: str
    total_hits: int
    hits: List[SearchHit]
    telemetry: SearchTelemetry


class HybridEngine:
    """
    Production-grade local hybrid neural search engine.
    """

    def __init__(
        self,
        dense_encoder: Optional[DenseEncoder] = None,
        reranker: Optional[CrossEncoderReranker] = None,
        enable_reranker: bool = True,
    ):
        self.inverted_index = InvertedIndex()
        self.bm25_ranker = BM25Ranker(self.inverted_index)
        self.dense_encoder = dense_encoder or DenseEncoder()
        self.vector_index = VectorIndex(dimension=384)
        self.reranker = reranker or (CrossEncoderReranker() if enable_reranker else None)
        self.web_fetcher = WebSearchFetcher()

    def add_document(self, doc_id: int, text: str, metadata: Optional[dict] = None) -> None:
        """Indexes a document in both lexical and dense vector indices."""
        meta = metadata or {}
        if "content" not in meta:
            meta["content"] = text

        # 1. Lexical index
        self.inverted_index.add_document(doc_id, text, meta)

        # 2. Dense vector index
        vec = self.dense_encoder.encode(text)
        self.vector_index.add_vector(doc_id, vec)

    def remove_document(self, doc_id: int) -> bool:
        """Removes a document from both indices."""
        lex_removed = self.inverted_index.remove_document(doc_id)
        vec_removed = self.vector_index.remove_vector(doc_id)
        return lex_removed or vec_removed

    def search(
        self,
        query: str,
        mode: Literal["hybrid", "bm25", "dense"] = "hybrid",
        top_k: int = 10,
        candidate_pool_size: int = 30,
        rerank: bool = True,
    ) -> SearchResponse:
        """
        Executes search with comprehensive stage profiling.
        """
        start_total = time.perf_counter()
        telemetry = SearchTelemetry()

        bm25_results: List[BM25Result] = []
        dense_results = []

        # 1. Lexical Retrieval
        if mode in ("hybrid", "bm25"):
            t0 = time.perf_counter()
            bm25_results = self.bm25_ranker.search(query, top_k=candidate_pool_size)
            telemetry.bm25_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        # 2. Dense Vector Retrieval
        if mode in ("hybrid", "dense"):
            # Embed query
            t0 = time.perf_counter()
            q_vec = self.dense_encoder.encode(query)
            telemetry.embed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

            # Vector search
            t1 = time.perf_counter()
            dense_results = self.vector_index.search(q_vec, top_k=candidate_pool_size)
            telemetry.vector_search_ms = round((time.perf_counter() - t1) * 1000.0, 2)

        # 3. Stage Branching
        hits: List[SearchHit] = []

        if mode == "bm25":
            for rank, r in enumerate(bm25_results[:top_k], start=1):
                meta = r.metadata
                hits.append(
                    SearchHit(
                        doc_id=r.doc_id,
                        score=r.score,
                        rank=rank,
                        title=meta.get("title", f"Doc {r.doc_id}"),
                        content=meta.get("content", ""),
                        metadata=meta,
                        provenance={
                            "bm25_score": r.score,
                            "matched_terms": r.matched_terms,
                            "term_contributions": r.term_contributions,
                        },
                    )
                )

        elif mode == "dense":
            for rank, r in enumerate(dense_results[:top_k], start=1):
                meta = self.inverted_index.get_doc_metadata(r.doc_id) or {}
                hits.append(
                    SearchHit(
                        doc_id=r.doc_id,
                        score=r.score,
                        rank=rank,
                        title=meta.get("title", f"Doc {r.doc_id}"),
                        content=meta.get("content", ""),
                        metadata=meta,
                        provenance={"cosine_similarity": r.score},
                    )
                )

        else:  # Hybrid Mode
            # Fusion
            t0 = time.perf_counter()
            fused_candidates = reciprocal_rank_fusion(
                bm25_results,
                dense_results,
                k=60,
                top_k=candidate_pool_size,
            )
            telemetry.fusion_ms = round((time.perf_counter() - t0) * 1000.0, 2)

            # Stage 3: Neural Cross-Encoder Reranker
            if rerank and self.reranker and fused_candidates:
                t0 = time.perf_counter()
                rerank_candidates = fused_candidates[:candidate_pool_size]
                pairs = []
                for c in rerank_candidates:
                    meta = self.inverted_index.get_doc_metadata(c.doc_id) or {}
                    content_text = meta.get("content", "")
                    pairs.append((query, content_text))

                rerank_scores = self.reranker.predict(pairs)
                telemetry.rerank_ms = round((time.perf_counter() - t0) * 1000.0, 2)

                # Pair candidates with rerank scores and sort
                scored_candidates = []
                for cand, r_score in zip(rerank_candidates, rerank_scores):
                    scored_candidates.append((cand, r_score))

                scored_candidates.sort(key=lambda x: x[1], reverse=True)

                for rank, (c, r_score) in enumerate(scored_candidates[:top_k], start=1):
                    meta = self.inverted_index.get_doc_metadata(c.doc_id) or {}
                    hits.append(
                        SearchHit(
                            doc_id=c.doc_id,
                            score=r_score,
                            rank=rank,
                            title=meta.get("title", f"Doc {c.doc_id}"),
                            content=meta.get("content", ""),
                            metadata=meta,
                            provenance={
                                "rerank_score": r_score,
                                "rrf_score": c.score,
                                "bm25_score": c.bm25_score,
                                "bm25_rank": c.bm25_rank,
                                "dense_score": c.dense_score,
                                "dense_rank": c.dense_rank,
                            },
                        )
                    )
            else:
                for rank, c in enumerate(fused_candidates[:top_k], start=1):
                    meta = self.inverted_index.get_doc_metadata(c.doc_id) or {}
                    hits.append(
                        SearchHit(
                            doc_id=c.doc_id,
                            score=c.score,
                            rank=rank,
                            title=meta.get("title", f"Doc {c.doc_id}"),
                            content=meta.get("content", ""),
                            metadata=meta,
                            provenance={
                                "rrf_score": c.score,
                                "bm25_score": c.bm25_score,
                                "bm25_rank": c.bm25_rank,
                                "dense_score": c.dense_score,
                                "dense_rank": c.dense_rank,
                            },
                        )
                    )

        telemetry.total_ms = round((time.perf_counter() - start_total) * 1000.0, 2)
        return SearchResponse(
            query=query,
            mode=mode,
            total_hits=len(hits),
            hits=hits,
            telemetry=telemetry,
        )

    async def search_web(
        self,
        query: str,
        category: str = "all",
        top_k: int = 15,
        rerank: bool = True,
    ) -> WebSearchResponse:
        """
        Executes live internet search with tracker purge, ad filtering,
        instant Wikipedia knowledge card, and on-device neural reranking.
        """
        start_total = time.perf_counter()
        telemetry = SearchTelemetry()

        # 1. Fetch live web results and instant answers concurrently
        t0 = time.perf_counter()
        web_task = self.web_fetcher.fetch_web_results(query, category=category)
        answer_task = self.web_fetcher.fetch_instant_answer(query)

        (raw_results, ads_blocked, trackers_purged), instant_answer = await asyncio.gather(
            web_task, answer_task
        )
        telemetry.embed_ms = round((time.perf_counter() - t0) * 1000.0, 2)  # Network fetch latency

        # 2. Local Neural Cross-Encoder Reranking
        if rerank and self.reranker and raw_results:
            t0 = time.perf_counter()
            pairs = [(query, r.snippet) for r in raw_results]
            scores = self.reranker.predict(pairs)
            telemetry.rerank_ms = round((time.perf_counter() - t0) * 1000.0, 2)

            for i, r in enumerate(raw_results):
                r.rerank_score = scores[i]
                r.score = scores[i]

            # Re-sort organic results by semantic cross-encoder score
            raw_results.sort(key=lambda item: item.score, reverse=True)
            for idx, r in enumerate(raw_results, start=1):
                r.rank = idx

        telemetry.total_ms = round((time.perf_counter() - start_total) * 1000.0, 2)

        return WebSearchResponse(
            query=query,
            category=category,
            total_results=len(raw_results[:top_k]),
            hits=raw_results[:top_k],
            instant_answer=instant_answer,
            ads_blocked_count=ads_blocked,
            trackers_purged_count=trackers_purged,
            telemetry=telemetry,
        )
