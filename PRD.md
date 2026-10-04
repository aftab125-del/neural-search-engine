# Product Requirements Document (PRD): Private Hybrid Neural Search Engine

**Project Name**: NeuralSearch (Local-First Hybrid Neural Information Retrieval Engine)  
**Target Root**: `D:\neural-search-engine`  
**Author**: CSE (AI & Data Science) Engineering  
**Version**: 1.0.0  
**Status**: Approved for Scaffolding  

---

## 1. Executive Summary & Vision

### 1.1 The Problem
Standard search in local desktop environments and personal knowledge bases suffers from two distinct failure modes:
1. **Keyword-Only Search (Grep / Classical Inverted Index)**: Breaks down when users search for concepts rather than exact vocabulary (e.g., searching *"how to speed up database queries"* fails to retrieve documents discussing *"B+Tree index tuning"* or *"query execution planning"*).
2. **Pure Dense Vector Search (Naive RAG / Vector DBs)**: Catastrophically fails on exact technical symbols, specific error codes (e.g., `TypeError: 'NoneType' object`), function names (`get_user_by_id`), model numbers, or acronyms.
3. **Cloud Privacy Risk**: Uploading proprietary company code, personal notes, or internal documentation to third-party cloud APIs (OpenAI, Cohere) violates data sovereignty and introduces recurring latency and costs.

### 1.2 The Solution
**NeuralSearch** is a local-first, privacy-preserving hybrid neural search engine that marries **Lexical Search (BM25 Inverted Index)** with **Dense Semantic Retrieval (Quantized ONNX Bi-Encoders)** and a **Neural Cross-Encoder Reranker**. It delivers sub-50ms search over local documentation, codebases, and markdown knowledge bases without a single byte leaving the user's machine.

---

## 2. User Personas & Use Cases

| Persona | Primary Goal | Critical Need |
| :--- | :--- | :--- |
| **Software Engineers & AI Researchers** | Search internal codebases, technical specs, arXiv PDFs, and documentation | Instant retrieval of exact symbols + high-recall conceptual search |
| **Knowledge Workers & Students** | Query thousands of personal notes, markdown files, and research summaries | Zero-latency search-as-you-type with snippet highlighting |
| **Technical Interviewers & Hiring Managers** | Evaluate candidates on systems, AI/ML theory, and production software standards | Mathematical rigor (NDCG/MRR evaluation), clean architecture, and latency instrumentation |

---

## 3. Core Feature Requirements

### Feature 1: Multi-Stage Hybrid Retrieval Pipeline
- **Stage 1A (Lexical Retrieval - BM25)**:
  - Custom tokenization, lowercasing, stopword removal, and Porter stemming.
  - Inverted index storing term frequencies, document lengths, and posting lists.
  - Okapi BM25 scoring with configurable $k_1$ (1.2–2.0) and $b$ (0.75).
- **Stage 1B (Dense Semantic Retrieval - Bi-Encoder)**:
  - Local transformer embedding model (`all-MiniLM-L6-v2` or `bge-small-en-v1.5`) running via **quantized ONNX Runtime** on CPU.
  - Vector similarity search (Cosine Similarity / Dot Product) over dense 384-dimensional embeddings.
- **Stage 2 (Hybrid Rank Fusion)**:
  - **Reciprocal Rank Fusion (RRF)** merging top-$N$ lexical and semantic candidates:
    $$\text{RRF\_Score}(d) = \sum_{m \in \{\text{BM25}, \text{Dense}\}} \frac{1}{k + \text{rank}_m(d)}$$
  - Configurable weight sliders ($\alpha \cdot \text{BM25} + (1 - \alpha) \cdot \text{Dense}$).
- **Stage 3 (Cross-Encoder Neural Reranking)**:
  - Joint query-document attention scoring using `cross-encoder/ms-marco-MiniLM-L-6-v2` on the top 20 fused candidates to determine the final top 10 results.

### Feature 2: Context-Aware Document Ingestion & Chunking
- Support for Markdown (`.md`), Plain Text (`.txt`), Code files (`.py`, `.ts`, `.json`), and HTML.
- **Header-Aware Hierarchical Chunking**: Chunks documents along semantic sections (H1, H2, code blocks) with sliding window token overlap (256–512 tokens with 64 overlap).
- Document metadata extraction: title, file path, line ranges, heading hierarchy, document hash for incremental re-indexing.

### Feature 3: Light Minimalist User Interface
- Aesthetic based on the **Terracotta & Alabaster Palette**:
  - Primary Canvas: `#FCF2E5` (Warm Alabaster Cream)
  - Text & Headers: `#524646` & `#3A3131` (Deep Espresso Charcoal)
  - Subtle Borders & Muted Elements: `#E5DBCA` & `#A8A492`
  - Accent / Focus / Badges: `#EC5B38` (Burnt Terracotta Orange)
- Search-as-you-type with 150ms debouncing.
- Dynamic keyword and semantic match highlighting in text snippets.
- Keyboard navigation (`/` to focus, `ArrowUp`/`ArrowDown` or `j`/`k` to select, `Enter` to preview).
- **"Under-the-Hood" Explainability Drawer**: Displays exact BM25 scores, vector cosine distances, and reranking deltas for any selected result.

### Feature 4: Live Latency Instrumentation & Health Metrics
- Timing waterfall returned in every API query response:
  - Lexical Search: $T_{\text{bm25}}$ (Target: $<5\text{ms}$)
  - Vector Embedding: $T_{\text{embed}}$ (Target: $<15\text{ms}$ on CPU)
  - Vector Similarity: $T_{\text{vec}}$ (Target: $<3\text{ms}$)
  - Neural Reranking: $T_{\text{rerank}}$ (Target: $<20\text{ms}$)
  - Total Round-Trip: $<50\text{ms}$
- Corpus statistics endpoint (`/api/stats`): Document count, chunk count, vocabulary size, index memory footprint.

### Feature 5: Rigorous IR Evaluation Harness (The Resume Power Feature)
- Quantitative evaluation script against standard Information Retrieval benchmarks or curated test suites.
- Computes:
  - **NDCG@10** (Normalized Discounted Cumulative Gain)
  - **MRR** (Mean Reciprocal Rank)
  - **Precision@K** and **Recall@K**
- Generates automated comparative tables showing: `BM25 Only` vs `Dense Only` vs `Hybrid RRF` vs `Hybrid + Reranker`.

---

## 4. Non-Functional Requirements

### 4.1 Performance & Latency
- End-to-end query latency must remain below **60ms** on modern x86/ARM multi-core CPUs without GPU acceleration.
- Incremental indexing rate of at least **100 chunks/second**.

### 4.2 Storage & Memory
- In-memory index footprint under **250MB** for up to 10,000 document chunks.
- Persistent index backed by SQLite (WAL mode) and local binary vector matrices.

### 4.3 Privacy & Security
- 100% offline operation: Zero outbound network calls during search or indexing.
- Local model weights stored in `.cache/models/` using ONNX quantization.

---

## 5. Success Metrics
- **System Quality**: NDCG@10 improvement of $\ge +15\%$ for hybrid search compared to pure BM25 or pure dense vector search.
- **Engineering Excellence**: 100% test coverage on scoring algorithms (BM25 math, RRF fusion, cosine similarity) and sub-50ms P95 latency.
- **Resume Impact**: Interactive portfolio demo demonstrating high-level AI/IR theory backed by low-level systems engineering.
