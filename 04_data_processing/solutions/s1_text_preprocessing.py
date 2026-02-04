"""
SOLUTIONS - Text Preprocessing
================================
Try to solve the problems yourself first!
"""
import re
import math
from collections import Counter


def tokenize(text: str) -> list[str]:
    """
    Key insight: regex sub to remove non-alphanumeric, then split.
    """
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    tokens = text.split()
    return [t for t in tokens if t]  # remove empty strings


def compute_tf(document: list[str]) -> dict[str, float]:
    """
    Key insight: Counter gives word counts, divide by total.
    """
    counts = Counter(document)
    total = len(document)
    return {word: count / total for word, count in counts.items()}


def compute_idf(documents: list[list[str]]) -> dict[str, float]:
    """
    Key insight: For each word, count how many documents contain it.
    Use set() for each doc to avoid counting duplicates within a doc.
    """
    n_docs = len(documents)
    # Count document frequency for each word
    doc_freq = Counter()
    for doc in documents:
        unique_words = set(doc)
        for word in unique_words:
            doc_freq[word] += 1

    return {word: math.log(n_docs / df) for word, df in doc_freq.items()}


def compute_tfidf(documents: list[str]) -> list[dict[str, float]]:
    """
    Key insight: Combine tokenize -> TF -> IDF -> multiply.
    """
    # Tokenize all documents
    tokenized = [tokenize(doc) for doc in documents]

    # Compute IDF across all docs
    idf = compute_idf(tokenized)

    # Compute TF-IDF for each document
    result = []
    for doc_tokens in tokenized:
        tf = compute_tf(doc_tokens)
        doc_tfidf = {word: tf_val * idf.get(word, 0) for word, tf_val in tf.items()}
        result.append(doc_tfidf)

    return result


def clean_for_embedding(text: str) -> str:
    """
    Key insight: Use regex patterns for URLs and emails.
    Order matters: remove URLs/emails first, then clean whitespace.
    """
    # Remove URLs
    text = re.sub(r'https?://\S+', '', text)
    # Remove emails
    text = re.sub(r'\S+@\S+\.\S+', '', text)
    # Remove non-ASCII
    text = text.encode('ascii', errors='ignore').decode('ascii')
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text)
    # Strip
    text = text.strip()
    return text


def chunk_text(text: str, chunk_size: int = 200, overlap: int = 50) -> list[str]:
    """
    Key insight: Use a sliding window over the word list.
    Step size = chunk_size - overlap.
    """
    words = text.split()
    if len(words) <= chunk_size:
        return [text]

    chunks = []
    step = chunk_size - overlap
    for i in range(0, len(words), step):
        chunk_words = words[i:i + chunk_size]
        if chunk_words:  # don't add empty chunks
            chunks.append(" ".join(chunk_words))
        if i + chunk_size >= len(words):
            break

    return chunks
