"""
Web retrieval, ad filtering, and privacy proxy subsystem.
"""

from .ad_blocker import AdBlocker
from .parser import DDGParser
from .fetcher import WebSearchFetcher, WebSearchResult, InstantAnswer

__all__ = ["AdBlocker", "DDGParser", "WebSearchFetcher", "WebSearchResult", "InstantAnswer"]
