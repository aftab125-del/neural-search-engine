"""
Privacy-Preserving Web Search Fetcher.
Queries live web endpoints anonymously, purges trackers and ads, and extracts organic results.
"""

from __future__ import annotations
import asyncio
import html
import random
import re
import urllib.parse
from dataclasses import dataclass
from typing import List, Optional, Tuple
import httpx

from .ad_blocker import AdBlocker
from .parser import DDGParser

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36 Edg/123.0.0.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
]


@dataclass(slots=True)
class WebSearchResult:
    """An organic web search hit."""
    title: str
    url: str
    display_url: str
    domain: str
    snippet: str
    score: float = 0.0
    rank: int = 1
    rerank_score: Optional[float] = None
    trackers_purged: int = 0
    is_organic: bool = True


@dataclass(slots=True)
class InstantAnswer:
    """Instant knowledge summary (e.g. from Wikipedia)."""
    title: str
    extract: str
    url: str
    source: str = "Wikipedia"


class WebSearchFetcher:
    """
    Asynchronous web search client with zero user profiling and ad stripping.
    """

    def __init__(self, timeout: float = 8.0):
        self.timeout = timeout
        self.ad_blocker = AdBlocker()

    def _get_headers(self) -> dict:
        """Emits anonymized, tracking-free headers."""
        return {
            "User-Agent": random.choice(USER_AGENTS),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
            "DNT": "1",  # Do Not Track
            "Sec-GPC": "1",  # Global Privacy Control
            "Upgrade-Insecure-Requests": "1",
        }

    async def fetch_web_results(
        self,
        query: str,
        category: str = "all",
    ) -> Tuple[List[WebSearchResult], int, int]:
        """
        Executes live internet search on behalf of the user.
        Returns: (results, ads_blocked_count, trackers_stripped_count)
        """
        refined_query = query.strip()
        if category == "tech":
            refined_query = f"{refined_query} (github OR stackoverflow OR docs OR programming)"
        elif category == "news":
            refined_query = f"{refined_query} news"

        headers = self._get_headers()
        results: List[WebSearchResult] = []
        ads_blocked = 0
        trackers_stripped = 0

        async with httpx.AsyncClient(timeout=self.timeout, follow_redirects=True) as client:
            # 1. Primary Retrieval: Clean Search Engine HTML
            try:
                resp = await client.get(
                    "https://www.bing.com/search",
                    params={"q": refined_query},
                    headers=headers,
                )
                if resp.status_code == 200:
                    algos = re.findall(r'<li class="b_algo"[^>]*>(.*?)</li>', resp.text, re.DOTALL)
                    for algo in algos:
                        # Check for sponsored ad indicator
                        if 'class="b_ad"' in algo or 'sponsored' in algo.lower():
                            ads_blocked += 1
                            continue

                        title_m = re.search(r'<h2[^>]*><a\s+[^>]*href="([^"]+)"[^>]*>(.*?)</h2>', algo, re.DOTALL)
                        if not title_m:
                            continue
                        raw_href = title_m.group(1)
                        raw_title = re.sub(r'<[^>]+>', '', title_m.group(2)).strip()
                        raw_title = html.unescape(raw_title)

                        snippet_m = re.search(r'<p[^>]*>(.*?)</p>', algo, re.DOTALL)
                        snippet = re.sub(r'<[^>]+>', '', snippet_m.group(1)).strip() if snippet_m else ""
                        snippet = html.unescape(snippet)

                        # Filter sponsored text
                        if self.ad_blocker.is_sponsored_text(raw_title, snippet):
                            ads_blocked += 1
                            continue

                        # Clean URL & purge trackers
                        clean_info = self.ad_blocker.clean_url(raw_href)
                        if clean_info.is_ad:
                            ads_blocked += 1
                            continue

                        trackers_stripped += clean_info.trackers_removed

                        try:
                            parsed_uri = urllib.parse.urlparse(clean_info.clean_url)
                            domain = parsed_uri.netloc or "web"
                            path = parsed_uri.path.strip("/")
                            display_url = f"{domain} > {path}" if path else domain
                        except Exception:
                            domain = "web"
                            display_url = clean_info.clean_url

                        results.append(
                            WebSearchResult(
                                title=raw_title,
                                url=clean_info.clean_url,
                                display_url=display_url,
                                domain=domain,
                                snippet=snippet,
                                trackers_purged=clean_info.trackers_removed,
                            )
                        )
            except Exception as e:
                # Bing error fallback
                pass

            # 2. Secondary Fallback: Wikipedia OpenSearch if results are few
            if len(results) < 3:
                try:
                    wiki_resp = await client.get(
                        "https://en.wikipedia.org/w/api.php",
                        params={
                            "action": "query",
                            "list": "search",
                            "srsearch": query,
                            "format": "json",
                            "utf8": 1,
                        },
                        headers={"User-Agent": "NeuralSearchEngine/1.0 (cse.student@project.local)"},
                    )
                    if wiki_resp.status_code == 200:
                        wiki_data = wiki_resp.json()
                        wiki_items = wiki_data.get("query", {}).get("search", [])
                        for item in wiki_items[:5]:
                            clean_snippet = re.sub(r'<[^>]+>', '', item.get("snippet", ""))
                            page_title = item.get("title", "")
                            page_url = f"https://en.wikipedia.org/wiki/{urllib.parse.quote(page_title.replace(' ', '_'))}"
                            results.append(
                                WebSearchResult(
                                    title=f"{page_title} - Wikipedia",
                                    url=page_url,
                                    display_url="en.wikipedia.org > wiki",
                                    domain="en.wikipedia.org",
                                    snippet=clean_snippet,
                                    trackers_purged=0,
                                )
                            )
                except Exception:
                    pass

        return results, ads_blocked, trackers_stripped

    async def fetch_instant_answer(self, query: str) -> Optional[InstantAnswer]:
        """
        Retrieves direct knowledge cards (e.g. Wikipedia summary) for informational queries.
        """
        clean_q = query.strip()
        # Clean query: take first 3-4 words for entity lookup if query is long
        words = clean_q.split()
        candidate = " ".join(words[:4]) if len(words) > 4 else clean_q

        headers = {
            "User-Agent": "NeuralSearchEngine/1.0 (cse.student@project.local)",
            "Accept": "application/json",
        }

        async with httpx.AsyncClient(timeout=4.0) as client:
            try:
                # Try direct REST summary
                title_slug = urllib.parse.quote(candidate.capitalize().replace(" ", "_"))
                url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{title_slug}"
                resp = await client.get(url, headers=headers)

                if resp.status_code == 200:
                    data = resp.json()
                    if data.get("type") in ("standard", "description") and data.get("extract"):
                        article_url = data.get("content_urls", {}).get("desktop", {}).get("page", f"https://en.wikipedia.org/wiki/{title_slug}")
                        return InstantAnswer(
                            title=data.get("title", candidate),
                            extract=data.get("extract", ""),
                            url=article_url,
                            source="Wikipedia Instant Answer",
                        )

                # Fallback: Query search API to find matching page title
                search_url = "https://en.wikipedia.org/w/api.php"
                s_resp = await client.get(
                    search_url,
                    params={
                        "action": "query",
                        "list": "search",
                        "srsearch": candidate,
                        "format": "json",
                        "srlimit": 1,
                    },
                    headers=headers,
                )
                if s_resp.status_code == 200:
                    s_data = s_resp.json()
                    search_hits = s_data.get("query", {}).get("search", [])
                    if search_hits:
                        top_title = search_hits[0]["title"]
                        top_slug = urllib.parse.quote(top_title.replace(" ", "_"))
                        top_resp = await client.get(f"https://en.wikipedia.org/api/rest_v1/page/summary/{top_slug}", headers=headers)
                        if top_resp.status_code == 200:
                            top_data = top_resp.json()
                            if top_data.get("extract"):
                                return InstantAnswer(
                                    title=top_data.get("title", top_title),
                                    extract=top_data.get("extract", ""),
                                    url=top_data.get("content_urls", {}).get("desktop", {}).get("page", f"https://en.wikipedia.org/wiki/{top_slug}"),
                                    source="Wikipedia Instant Answer",
                                )
            except Exception:
                pass

        return None
