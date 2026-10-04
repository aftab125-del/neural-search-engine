"""
Snippet extractor and highlighter for search result presentation.
Generates contextual excerpts around matched terms with clean HTML highlight spans.
"""

from __future__ import annotations
import html
import re
from typing import List, Set


class SnippetHighlighter:
    """
    Finds the most relevant text window in a document for a query and highlights matching terms.
    """

    def __init__(self, snippet_window_words: int = 40):
        self.snippet_window_words = snippet_window_words

    def extract_snippet(self, content: str, query_terms: List[str]) -> str:
        """
        Extracts a concise snippet window centered around matching query terms.
        """
        if not content:
            return ""

        words = content.split()
        if len(words) <= self.snippet_window_words:
            return self.highlight(content, query_terms)

        normalized_query_terms = set(t.lower() for t in query_terms if len(t) > 1)
        if not normalized_query_terms:
            # Default to first window
            return self.highlight(" ".join(words[:self.snippet_window_words]) + "...", query_terms)

        # Score sliding windows of words
        best_start = 0
        best_score = -1

        for i in range(0, len(words) - self.snippet_window_words + 1, 5):
            window = words[i : i + self.snippet_window_words]
            score = sum(1 for w in window if re.sub(r"\W+", "", w).lower() in normalized_query_terms)
            if score > best_score:
                best_score = score
                best_start = i

        snippet_words = words[best_start : best_start + self.snippet_window_words]
        snippet_text = " ".join(snippet_words)

        prefix = "..." if best_start > 0 else ""
        suffix = "..." if (best_start + self.snippet_window_words) < len(words) else ""

        full_snippet = f"{prefix}{snippet_text}{suffix}"
        return self.highlight(full_snippet, query_terms)

    @staticmethod
    def highlight(text: str, query_terms: List[str]) -> str:
        """
        Safely escapes HTML and wraps matched query terms in highlight tags.
        """
        escaped_text = html.escape(text)
        clean_terms = [re.escape(t.strip()) for t in query_terms if len(t.strip()) > 1]
        if not clean_terms:
            return escaped_text

        pattern = re.compile(r"\b(" + "|".join(clean_terms) + r")\b", re.IGNORECASE)
        highlighted = pattern.sub(r'<mark class="highlight">\1</mark>', escaped_text)
        return highlighted
