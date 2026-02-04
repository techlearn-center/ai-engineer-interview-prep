import sys
import os
import math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from p1_text_preprocessing import (
    tokenize,
    compute_tf,
    compute_idf,
    compute_tfidf,
    clean_for_embedding,
    chunk_text,
)


class TestTokenize:
    def test_basic(self):
        result = tokenize("Hello, World! This is a test.")
        assert result == ["hello", "world", "this", "is", "a", "test"]

    def test_numbers(self):
        result = tokenize("I have 3 cats and 2 dogs")
        assert result == ["i", "have", "3", "cats", "and", "2", "dogs"]

    def test_empty(self):
        assert tokenize("") == []

    def test_special_chars(self):
        result = tokenize("hello---world***test")
        assert result == ["helloworldtest"] or result == ["hello", "world", "test"]


class TestComputeTF:
    def test_basic(self):
        result = compute_tf(["the", "cat", "sat", "on", "the", "mat"])
        assert abs(result["the"] - 2/6) < 1e-10
        assert abs(result["cat"] - 1/6) < 1e-10

    def test_single_word(self):
        result = compute_tf(["hello"])
        assert result == {"hello": 1.0}


class TestComputeIDF:
    def test_basic(self):
        docs = [["the", "cat"], ["the", "dog"], ["a", "bird"]]
        result = compute_idf(docs)
        assert abs(result["the"] - math.log(3/2)) < 1e-10
        assert abs(result["cat"] - math.log(3/1)) < 1e-10
        assert abs(result["a"] - math.log(3/1)) < 1e-10

    def test_word_in_all_docs(self):
        docs = [["hello"], ["hello"], ["hello"]]
        result = compute_idf(docs)
        assert abs(result["hello"] - math.log(3/3)) < 1e-10  # 0


class TestComputeTFIDF:
    def test_returns_list(self):
        docs = ["the cat sat", "the dog barked"]
        result = compute_tfidf(docs)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_values_are_floats(self):
        docs = ["hello world", "world peace"]
        result = compute_tfidf(docs)
        for doc_tfidf in result:
            for val in doc_tfidf.values():
                assert isinstance(val, float)


class TestCleanForEmbedding:
    def test_removes_urls(self):
        result = clean_for_embedding("Visit https://example.com today")
        assert "https://example.com" not in result
        assert "Visit" in result

    def test_removes_emails(self):
        result = clean_for_embedding("Contact info@test.com please")
        assert "info@test.com" not in result

    def test_collapses_whitespace(self):
        result = clean_for_embedding("hello    world\n\nfoo")
        assert result == "hello world foo"

    def test_strips(self):
        result = clean_for_embedding("  hello  ")
        assert result == "hello"


class TestChunkText:
    def test_basic(self):
        text = " ".join([f"word{i}" for i in range(10)])
        result = chunk_text(text, chunk_size=4, overlap=1)
        assert len(result) >= 3
        # First chunk should have 4 words
        assert len(result[0].split()) == 4

    def test_overlap(self):
        text = "word1 word2 word3 word4 word5"
        result = chunk_text(text, chunk_size=3, overlap=1)
        assert result[0] == "word1 word2 word3"
        assert result[1] == "word3 word4 word5"

    def test_short_text(self):
        result = chunk_text("hello world", chunk_size=10, overlap=2)
        assert len(result) == 1
        assert result[0] == "hello world"
