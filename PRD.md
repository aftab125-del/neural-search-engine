# Product Requirements Document (PRD): Private, Ad-Free Neural Web Search Engine

**Project Name**: NeuralSearch (Privacy-Preserving, Ad-Free Internet Search Engine)  
**Target Root**: `D:\neural-search-engine`  
**Author**: CSE (AI & Data Science) Engineering  
**Version**: 2.0.0  
**Status**: Approved for Web Search Re-Architecture  

---

## 1. Executive Summary & Vision

### 1.1 The Problem
Commercial search engines (Google, Bing) operate on an ad-surveillance business model:
1. **User Profiling & Privacy Invasion**: Every query is tied to personal identity, IP address, location, and device fingerprints to build behavioral tracking profiles.
2. **Sponsored Ad Dominance**: Top search result slots are auctioned off to the highest bidder rather than the most relevant organic answer, forcing users to scroll past sponsored links.
3. **SEO Spam & Affiliate Bloat**: Search algorithms are gamed by affiliate marketers, ad-heavy recipe blogs, and content farms.

### 1.2 The Solution
**NeuralSearch** is an independent, privacy-first **Web Search Engine** that queries the live internet on behalf of the user. It acts as an **anonymizing privacy shield** between the user and the web:
- **Zero Query Logging & Tracking**: No cookies, no query history retention, no IP forwarding.
- **100% Ad & Tracking Stripper**: Automatically strips sponsored search ads, tracking pixels, and affiliate redirects (`utm_*`, `fbclid`, `gclid`).
- **Neural Re-Ranking Engine**: Uses local AI (BM25 + ONNX Bi-Encoder + Cross-Encoder) on fetched web results to prioritize organic, high-signal information over sponsored fluff.
- **Light Minimalist Web UI**: A serene, clean search interface (Warm Alabaster & Terracotta) that lets you search the web without visual noise or ads.

---

## 2. Core Functional Requirements

### Feature 1: Privacy-Preserving Web Retrieval Layer
* **Live Internet Search**: Connects to the live web via privacy-friendly search backends (DuckDuckGo Lite/HTML, Wikipedia API, ArXiv, and direct web scrapers).
* **Anonymization Proxy**:
  - Drops all client headers (`Referer`, `Cookie`, client IP).
  - Rotates user-agents to prevent search-engine fingerprinting.
  - Emits zero external requests with personal user identifiers.

### Feature 2: Anti-Ad & Anti-Tracker Sanitization Engine
* **Ad Link Stripping**: Detects and discards sponsored result cards and ad auction blocks.
* **URL Sanitization**: Cleans target URLs by removing surveillance parameters:
  - `utm_source`, `utm_medium`, `utm_campaign`, `gclid`, `fbclid`, `msclkid`, `ref`.
* **Tracker Shield Badge**: Displays real-time metrics showing how many ads and tracking parameters were purged from the search result list.

### Feature 3: Local Neural Reranker (AI Secret Weapon)
* While commercial search engines rank by who paid for ads, NeuralSearch reranks the organic internet results locally on your CPU:
  - **BM25 Lexical Matching**: Ensures exact query terms and technical symbols are prioritized.
  - **Dense Semantic Embeddings**: Captures deeper meaning and intent using quantized `all-MiniLM-L6-v2`.
  - **Cross-Encoder Reranker**: Joint query-snippet attention (`ms-marco-MiniLM-L-6-v2`) to place the most genuinely informative web result at Rank #1.

### Feature 4: Google-Style Multi-Modal Search Verticals
* **All Web**: General live internet search with on-device Cross-Encoder neural reranker and Wikipedia Instant Answer cards.
* **Images**: High-resolution image search grid with dimensions badges, hover previews, and full-resolution lightbox viewer.
* **Videos**: Video search with video thumbnails, duration pills, YouTube / web platform badges, and direct playback links.
* **News**: Real-time breaking news via Google News RSS syndication with publication sources, timestamps ("X hours ago"), and article snippets.
* **Shopping**: Commercial product search displaying item pricing (`$XX.XX`), merchant store badges (Amazon, Best Buy, Walmart, eBay), and direct store links stripped of affiliate tracking parameters.
* **Local Docs**: Offline search across your local project code, notes, and PDF/Markdown files using BM25 and Dense vector embeddings.

### Feature 5: Photo Search (Visual Search / Google Lens Equivalent)
* Integrated camera button in the search bar allowing users to search by image.
* **Drag-and-Drop Image Upload** or **Image URL Paste**.
* **On-Device Computer Vision Analysis**: Extracts format, resolution, aspect ratio, and visual perceptual hash.
* **Visual Match Discovery**: Finds visually matching and similar photos across the web with clean 1-click external privacy reverse search links (Google Lens, Bing Visual, TinEye).

### Feature 6: Instant Query Autocomplete
* Real-time search suggestions dropdown as the user types with full keyboard navigation (Arrow Up, Arrow Down, Enter to select).

---

## 3. Non-Functional Requirements

### 3.1 Privacy Guarantee
* **100% No-Log Policy**: Search queries are processed in memory and never written to disk or third-party servers.
* **No Cookies**: Zero tracking cookies are set in the user's browser.

### 3.2 Performance
* End-to-end web search, sanitization, and local neural reranking in **<1.2 seconds**.
* Sub-20ms latency when deep reranking is disabled.

---

## 4. Success Metrics
* **Privacy**: 0 third-party tracking cookies or ad pixels delivered to the client.
* **Result Quality**: 100% of sponsored ads filtered out from the final results.
* **Resume Impact**: Complete showcase of full-stack web engineering, privacy proxies, network scraping, and neural information retrieval.
