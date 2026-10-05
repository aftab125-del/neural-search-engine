"""
Anti-Ad & Tracker Sanitizer for NeuralSearch.
Filters sponsored advertisements, purges tracking parameters, and de-references redirect hops.
"""

from __future__ import annotations
import base64
import html
import re
import urllib.parse
from dataclasses import dataclass
from typing import List, Tuple

# Tracking query parameters to strip from URLs
TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "gclid", "gclsrc", "dclid", "fbclid", "msclkid", "twclid", "yclid",
    "ref", "ref_src", "source", "affiliate", "affiliate_id", "partner",
    "fclid", "scid", "cvid", "form", "sp", "qs", "sk", "ghc"
}

# Domains typically associated with ad networks or sponsored link redirections
AD_DOMAINS = {
    "googleadservices.com", "doubleclick.net", "adservice.google.com",
    "adnxs.com", "advertising.com", "taboola.com", "outbrain.com",
    "criteo.com", "adroll.com", "bing.com/aclick", "duckduckgo.com/y.js"
}


@dataclass(slots=True)
class SanitizedURL:
    clean_url: str
    trackers_removed: int
    is_ad: bool


class AdBlocker:
    """
    Sanitizes web search results by stripping ad-networks and surveillance parameters.
    """

    @staticmethod
    def clean_url(raw_url: str) -> SanitizedURL:
        """
        De-references redirect hops (like Bing u=a1..., DDG uddg=, or Google url?q=)
        and removes tracking query parameters.
        """
        url = html.unescape(raw_url.strip())
        trackers_from_redirect = 0

        # 1. Check for Bing click redirect hop: bing.com/ck/a?!...&u=a1<base64>
        bing_m = re.search(r'[?&]u=a1([a-zA-Z0-9_-]+)', url)
        if bing_m:
            b64_part = bing_m.group(1)
            padding = len(b64_part) % 4
            if padding:
                b64_part += "=" * (4 - padding)
            try:
                decoded = base64.urlsafe_b64decode(b64_part).decode("utf-8", errors="ignore")
                if decoded.startswith("http"):
                    url = decoded
                    trackers_from_redirect += 1
            except Exception:
                pass

        # 2. Check for DuckDuckGo redirect hop: /l/?uddg=https%3A%2F%2F...
        if "duckduckgo.com/l/?" in url or "uddg=" in url:
            parsed = urllib.parse.urlparse(url)
            query_dict = urllib.parse.parse_qs(parsed.query)
            if "uddg" in query_dict:
                url = query_dict["uddg"][0]
                trackers_from_redirect += 1

        # 3. Check for Google redirect hop: /url?q=https://...
        if "/url?q=" in url:
            parsed = urllib.parse.urlparse(url)
            query_dict = urllib.parse.parse_qs(parsed.query)
            if "q" in query_dict:
                url = query_dict["q"][0]
                trackers_from_redirect += 1

        # 4. Check if target is an ad network
        for ad_domain in AD_DOMAINS:
            if ad_domain in url.lower():
                return SanitizedURL(clean_url=url, trackers_removed=trackers_from_redirect + 1, is_ad=True)

        # 5. Strip tracking parameters
        try:
            parsed = urllib.parse.urlparse(url)
            query_dict = urllib.parse.parse_qs(parsed.query, keep_blank_values=True)
            original_param_count = len(query_dict)

            clean_params = {k: v for k, v in query_dict.items() if k.lower() not in TRACKING_PARAMS}
            trackers_removed = (original_param_count - len(clean_params)) + trackers_from_redirect

            new_query = urllib.parse.urlencode(clean_params, doseq=True)
            clean_url = urllib.parse.urlunparse(
                (parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment)
            )
            return SanitizedURL(clean_url=clean_url, trackers_removed=trackers_removed, is_ad=False)
        except Exception:
            return SanitizedURL(clean_url=url, trackers_removed=trackers_from_redirect, is_ad=False)

    @staticmethod
    def is_sponsored_text(title: str, snippet: str) -> bool:
        """Detects whether a search card text indicates a sponsored ad."""
        combined = f"{title} {snippet}".lower()
        ad_indicators = [
            "sponsored", "advertisement", "ad •", "promoted",
            "partner content", "featured ad"
        ]
        return any(ind in combined for ind in ad_indicators)
