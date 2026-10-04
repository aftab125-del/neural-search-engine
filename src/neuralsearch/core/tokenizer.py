"""
Robust, zero-dependency tokenizer and Porter Stemmer for lexical information retrieval.
Tracks token positions and character offsets for snippet highlighting.
"""

from __future__ import annotations
import re
from dataclasses import dataclass
from typing import List, Set


@dataclass(slots=True)
class Token:
    """Represents a token with position and character offsets."""
    raw: str
    normalized: str
    stemmed: str
    position: int
    start_char: int
    end_char: int


# Standard curated English stopwords
DEFAULT_STOPWORDS: Set[str] = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}


class PorterStemmer:
    """
    Standard Martin Porter 1980 stemming algorithm implemented in pure Python.
    Reduces words to their morphological roots (e.g., 'retrieval' -> 'retriev').
    """

    @staticmethod
    def _is_consonant(word: str, i: int) -> bool:
        letter = word[i]
        if letter in "aeiou":
            return False
        if letter == "y":
            return i == 0 or not PorterStemmer._is_consonant(word, i - 1)
        return True

    @staticmethod
    def _measure(word: str) -> int:
        """Measures the number of consonant sequences between vowels (m)."""
        m = 0
        i = 0
        n = len(word)
        while i < n:
            if not PorterStemmer._is_consonant(word, i):
                break
            i += 1
        while i < n:
            while i < n and not PorterStemmer._is_consonant(word, i):
                i += 1
            if i >= n:
                break
            m += 1
            while i < n and PorterStemmer._is_consonant(word, i):
                i += 1
        return m

    @staticmethod
    def _contains_vowel(stem: str) -> bool:
        for i in range(len(stem)):
            if not PorterStemmer._is_consonant(stem, i):
                return True
        return False

    @staticmethod
    def _ends_double_consonant(stem: str) -> bool:
        if len(stem) < 2:
            return False
        return stem[-1] == stem[-2] and PorterStemmer._is_consonant(stem, len(stem) - 1)

    @staticmethod
    def _cvc(stem: str) -> bool:
        if len(stem) < 3:
            return False
        return (
            PorterStemmer._is_consonant(stem, len(stem) - 1)
            and not PorterStemmer._is_consonant(stem, len(stem) - 2)
            and PorterStemmer._is_consonant(stem, len(stem) - 3)
            and stem[-1] not in "wxy"
        )

    def stem(self, word: str) -> str:
        word = word.lower()
        if len(word) <= 2:
            return word

        # Step 1a
        if word.endswith("sses"):
            word = word[:-2]
        elif word.endswith("ies"):
            word = word[:-2]
        elif not word.endswith("ss") and word.endswith("s"):
            word = word[:-1]

        # Step 1b
        extra_step = False
        if word.endswith("eed"):
            stem = word[:-3]
            if self._measure(stem) > 0:
                word = stem + "ee"
        elif word.endswith("ed"):
            stem = word[:-2]
            if self._contains_vowel(stem):
                word = stem
                extra_step = True
        elif word.endswith("ing"):
            stem = word[:-3]
            if self._contains_vowel(stem):
                word = stem
                extra_step = True

        if extra_step:
            if word.endswith("at") or word.endswith("bl") or word.endswith("iz"):
                word += "e"
            elif self._ends_double_consonant(word) and word[-1] not in "lsz":
                word = word[:-1]
            elif self._measure(word) == 1 and self._cvc(word):
                word += "e"

        # Step 1c
        if word.endswith("y"):
            stem = word[:-1]
            if self._contains_vowel(stem):
                word = stem + "i"

        # Step 2
        step2_suffixes = {
            "ational": "ate", "tional": "tion", "enci": "ence", "anci": "ance",
            "izer": "ize", "abli": "able", "alli": "al", "entli": "ent",
            "eli": "e", "ousli": "ous", "ization": "ize", "ation": "ate",
            "ator": "ate", "alism": "al", "iveness": "ive", "fulness": "ful",
            "ousness": "ous", "aliti": "al", "iviti": "ive", "biliti": "ble"
        }
        for suffix, repl in step2_suffixes.items():
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if self._measure(stem) > 0:
                    word = stem + repl
                break

        # Step 3
        step3_suffixes = {
            "icate": "ic", "ative": "", "alize": "al", "iciti": "ic",
            "ical": "ic", "ful": "", "ness": ""
        }
        for suffix, repl in step3_suffixes.items():
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if self._measure(stem) > 0:
                    word = stem + repl
                break

        # Step 4
        step4_suffixes = [
            "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement",
            "ment", "ent", "ou", "ism", "ate", "iti", "ous", "ive", "ize"
        ]
        for suffix in step4_suffixes:
            if word.endswith(suffix):
                stem = word[:-len(suffix)]
                if self._measure(stem) > 1:
                    word = stem
                break
        else:
            if word.endswith("ion"):
                stem = word[:-3]
                if self._measure(stem) > 1 and len(stem) > 0 and stem[-1] in "st":
                    word = stem

        # Step 5a
        if word.endswith("e"):
            stem = word[:-1]
            m = self._measure(stem)
            if m > 1 or (m == 1 and not self._cvc(stem)):
                word = stem

        # Step 5b
        if self._measure(word) > 1 and self._ends_double_consonant(word) and word.endswith("l"):
            word = word[:-1]

        return word


class Tokenizer:
    """
    High-performance regex tokenizer with stopword removal, stemming,
    and character span offset tracking.
    """

    # Matches alphanumeric sequences, optionally with internal hyphens or underscores
    TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9]+(?:[_\-][a-zA-Z0-9]+)*")

    def __init__(self, stopwords: Set[str] | None = None, enable_stemming: bool = True):
        self.stopwords = stopwords if stopwords is not None else DEFAULT_STOPWORDS
        self.enable_stemming = enable_stemming
        self.stemmer = PorterStemmer() if enable_stemming else None

    def tokenize(self, text: str) -> List[Token]:
        """
        Tokenizes input text into a list of Token objects with exact offsets.
        """
        tokens: List[Token] = []
        position = 0

        for match in self.TOKEN_PATTERN.finditer(text):
            raw = match.group(0)
            normalized = raw.lower()

            if normalized in self.stopwords:
                continue

            stemmed = self.stemmer.stem(normalized) if self.stemmer else normalized
            tokens.append(
                Token(
                    raw=raw,
                    normalized=normalized,
                    stemmed=stemmed,
                    position=position,
                    start_char=match.start(),
                    end_char=match.end(),
                )
            )
            position += 1

        return tokens

    def tokenize_terms(self, text: str) -> List[str]:
        """Convenience method returning just the list of stemmed query terms."""
        return [t.stemmed for t in self.tokenize(text)]
