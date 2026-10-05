import asyncio
import pytest
from neuralsearch.web.ad_blocker import AdBlocker
from neuralsearch.web.fetcher import WebSearchFetcher


def test_ad_blocker_strips_utm_and_click_ids():
    dirty_url = "https://example.com/article?utm_source=twitter&utm_medium=social&utm_campaign=spring2026&gclid=12345&fbclid=abcdef&article_id=999"
    sanitized = AdBlocker.clean_url(dirty_url)
    assert not sanitized.is_ad
    assert sanitized.trackers_removed == 5
    assert "utm_source" not in sanitized.clean_url
    assert "gclid" not in sanitized.clean_url
    assert "fbclid" not in sanitized.clean_url
    assert "article_id=999" in sanitized.clean_url


def test_ad_blocker_unmasks_bing_redirect():
    bing_redirect = "https://www.bing.com/ck/a?!&&p=123&u=a1aHR0cHM6Ly9mYXN0YXBpLnRpYW5nb2xvLmNvbS8&ntb=1"
    sanitized = AdBlocker.clean_url(bing_redirect)
    assert not sanitized.is_ad
    assert sanitized.clean_url == "https://fastapi.tiangolo.com/"
    assert sanitized.trackers_removed >= 1


def test_ad_blocker_unmasks_ddg_redirect():
    ddg_redirect = "https://duckduckgo.com/l/?uddg=https%3A%2F%2Fpython.org%2Fdownloads&rut=123"
    sanitized = AdBlocker.clean_url(ddg_redirect)
    assert not sanitized.is_ad
    assert sanitized.clean_url == "https://python.org/downloads"
    assert sanitized.trackers_removed >= 1


def test_ad_blocker_identifies_ad_domains():
    ad_url = "https://googleadservices.com/pagead/aclk?sa=L&ai=DChcSEw..."
    sanitized = AdBlocker.clean_url(ad_url)
    assert sanitized.is_ad


def test_ad_blocker_identifies_sponsored_text():
    assert AdBlocker.is_sponsored_text("Sponsored Product - Buy Cheap Cloud", "Get 50% off now")
    assert AdBlocker.is_sponsored_text("Cloud Computing Guide", "Advertisement: Top cloud providers")
    assert not AdBlocker.is_sponsored_text("FastAPI Documentation", "High performance web framework")


def test_instant_answer_wikipedia():
    async def _run():
        fetcher = WebSearchFetcher(timeout=5.0)
        return await fetcher.fetch_instant_answer("Python")

    ia = asyncio.run(_run())
    if ia:
        assert "Python" in ia.title
        assert len(ia.extract) > 20
        assert "wikipedia.org" in ia.url


def test_live_web_search():
    async def _run():
        fetcher = WebSearchFetcher(timeout=8.0)
        return await fetcher.fetch_web_results("fastapi framework")

    results, ads_blocked, trackers_purged = asyncio.run(_run())
    assert isinstance(results, list)
    assert len(results) > 0
    top = results[0]
    assert top.title
    assert top.url.startswith("http")
    assert top.domain


def test_image_search():
    async def _run():
        fetcher = WebSearchFetcher(timeout=8.0)
        return await fetcher.fetch_image_results("mountain landscape", top_k=5)

    images = asyncio.run(_run())
    assert isinstance(images, list)
    if images:
        top = images[0]
        assert top.image_url.startswith("http")
        assert top.title


def test_video_search():
    async def _run():
        fetcher = WebSearchFetcher(timeout=8.0)
        return await fetcher.fetch_video_results("python tutorial", top_k=5)

    videos = asyncio.run(_run())
    assert isinstance(videos, list)
    if videos:
        top = videos[0]
        assert top.title
        assert top.url.startswith("http")


def test_news_search():
    async def _run():
        fetcher = WebSearchFetcher(timeout=8.0)
        return await fetcher.fetch_news_results("technology", top_k=5)

    news = asyncio.run(_run())
    assert isinstance(news, list)
    if news:
        top = news[0]
        assert top.title
        assert top.url.startswith("http")
        assert top.source


def test_shopping_search():
    async def _run():
        fetcher = WebSearchFetcher(timeout=8.0)
        return await fetcher.fetch_shopping_results("laptop", top_k=5)

    products = asyncio.run(_run())
    assert isinstance(products, list)
    if products:
        top = products[0]
        assert top.title
        assert top.price


def test_autocomplete():
    async def _run():
        fetcher = WebSearchFetcher(timeout=4.0)
        return await fetcher.fetch_autocomplete("python")

    suggestions = asyncio.run(_run())
    assert isinstance(suggestions, list)
    assert len(suggestions) > 0
    assert any("python" in s.lower() for s in suggestions)


def test_visual_search_metadata():
    async def _run():
        fetcher = WebSearchFetcher(timeout=5.0)
        # 100 bytes of dummy image data
        dummy_png = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x01\x00\x00\x00\x01\x00\x08\x02\x00\x00\x00"
        return await fetcher.fetch_visual_search(image_bytes=dummy_png)

    res = asyncio.run(_run())
    assert res.detected_info["format"] == "PNG"
    assert "google_lens" in res.reverse_search_links
    assert "bing_visual" in res.reverse_search_links
