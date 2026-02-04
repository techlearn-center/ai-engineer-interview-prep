"""
Problem 1: Text Preprocessing for NLP/LLM
==========================================
Difficulty: Easy -> Medium

Text preprocessing is fundamental for any NLP/AI engineer role.
These patterns show up in RAG, fine-tuning data prep, and feature engineering.

Run tests:
    pytest 04_data_processing/tests/test_p1_text_preprocessing.py -v
"""
import re
from collections import Counter


def tokenize(text: str) -> list[str]:
    """
    Simple word tokenizer:
        1. Convert to lowercase
        2. Remove all non-alphanumeric characters (except spaces)
        3. Split on whitespace
        4. Remove empty strings

    Example:
        tokenize("Hello, World! This is a test.")
        -> ["hello", "world", "this", "is", "a", "test"]
    """
    # YOUR CODE HERE
    pass


def compute_tf(document: list[str]) -> dict[str, float]:
    """
    Compute Term Frequency for a tokenized document.
    TF(word) = count(word) / total_words

    Example:
        compute_tf(["the", "cat", "sat", "on", "the", "mat"])
        -> {"the": 0.333, "cat": 0.167, "sat": 0.167, "on": 0.167, "mat": 0.167}
    """
    # YOUR CODE HERE
    pass


def compute_idf(documents: list[list[str]]) -> dict[str, float]:
    """
    Compute Inverse Document Frequency.
    IDF(word) = log(total_documents / documents_containing_word)

    Use natural log (math.log or numpy.log).

    Example:
        docs = [["the", "cat"], ["the", "dog"], ["a", "bird"]]
        idf("the") = log(3/2) = 0.405
        idf("cat") = log(3/1) = 1.099
    """
    # YOUR CODE HERE
    pass


def compute_tfidf(documents: list[str]) -> list[dict[str, float]]:
    """
    Compute TF-IDF for a list of raw text documents.
    TF-IDF = TF * IDF

    Steps:
        1. Tokenize each document
        2. Compute TF for each document
        3. Compute IDF across all documents
        4. Multiply TF * IDF for each word in each document

    Return a list of dicts (one per document).
    """
    # YOUR CODE HERE
    pass


def clean_for_embedding(text: str) -> str:
    """
    Clean text for sending to an embedding model:
        1. Remove URLs
        2. Remove email addresses
        3. Remove extra whitespace (collapse multiple spaces/newlines to single space)
        4. Strip leading/trailing whitespace
        5. Remove any non-ASCII characters

    Example:
        clean_for_embedding("Visit https://example.com or email info@test.com\n\nHello   world!")
        -> "Visit or email Hello world!"
    """
    # YOUR CODE HERE
    pass


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """
    Split text into overlapping chunks for RAG systems.
    Split by words (not characters).

    chunk_size: max number of words per chunk
    overlap: number of overlapping words between consecutive chunks

    Example:
        text = "word1 word2 word3 word4 word5"
        chunk_text(text, chunk_size=3, overlap=1)
        -> ["word1 word2 word3", "word3 word4 word5"]
    """
    # YOUR CODE HERE
    pass
