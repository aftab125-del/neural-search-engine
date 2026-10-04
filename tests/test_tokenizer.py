import pytest
from neuralsearch.core.tokenizer import Tokenizer, PorterStemmer


def test_porter_stemmer():
    stemmer = PorterStemmer()
    assert stemmer.stem("retrieval") == "retriev"
    assert stemmer.stem("searching") == "search"
    assert stemmer.stem("searched") == "search"
    assert stemmer.stem("connects") == "connect"
    assert stemmer.stem("connecting") == "connect"
    assert stemmer.stem("connection") == "connect"
    assert stemmer.stem("vector") == "vector"


def test_tokenizer_basic():
    tokenizer = Tokenizer()
    text = "Dense vector embeddings for neural search!"
    tokens = tokenizer.tokenize(text)
    
    raw_words = [t.raw for t in tokens]
    stemmed_words = [t.stemmed for t in tokens]
    
    # "for" is a stopword and should be omitted
    assert "for" not in [t.normalized for t in tokens]
    assert "dense" in [t.normalized for t in tokens]
    assert "neural" in stemmed_words
    assert "search" in stemmed_words


def test_tokenizer_offsets():
    tokenizer = Tokenizer()
    text = "Fast HNSW index"
    tokens = tokenizer.tokenize(text)
    
    assert len(tokens) == 3
    # Check character slicing matches raw
    for token in tokens:
        assert text[token.start_char:token.end_char] == token.raw


def test_token_positions():
    tokenizer = Tokenizer()
    text = "first second third"
    tokens = tokenizer.tokenize(text)
    
    assert [t.position for t in tokens] == [0, 1, 2]
