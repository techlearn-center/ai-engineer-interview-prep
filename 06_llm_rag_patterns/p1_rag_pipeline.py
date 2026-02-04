"""
Problem 1: RAG (Retrieval-Augmented Generation) Pipeline
=========================================================
Difficulty: Medium -> Hard

This is THE hot topic for AI engineer interviews in 2024-2025.
You need to understand the full RAG pipeline.

Run tests:
    pytest 06_llm_rag_patterns/tests/test_p1_rag_pipeline.py -v
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class Document:
    """A document with content and metadata."""
    id: str
    content: str
    metadata: dict


class SimpleVectorStore:
    """
    A minimal vector store for RAG.
    In production you'd use Pinecone, Weaviate, ChromaDB, etc.
    This tests your understanding of the core concepts.
    """

    def __init__(self, embedding_dim: int = 4):
        self.embedding_dim = embedding_dim
        self.documents: list[Document] = []
        self.embeddings: list[np.ndarray] = []

    def _fake_embed(self, text: str) -> np.ndarray:
        """
        Fake embedding function for testing.
        In production, you'd call OpenAI/Cohere/etc.
        This creates a deterministic embedding from the text.
        """
        np.random.seed(hash(text) % 2**31)
        return np.random.randn(self.embedding_dim)

    def add_document(self, doc: Document):
        """
        Add a document to the store.
        Compute its embedding and store both.
        """
        # YOUR CODE HERE
        pass

    def similarity_search(self, query: str, top_k: int = 3) -> list[tuple[Document, float]]:
        """
        Find the top_k most similar documents to the query.

        Steps:
            1. Embed the query
            2. Compute cosine similarity between query and all stored embeddings
            3. Return top_k documents with their similarity scores
            4. Sort by similarity descending

        Return: list of (Document, similarity_score) tuples
        """
        # YOUR CODE HERE
        pass

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        # YOUR CODE HERE
        pass


class RAGPipeline:
    """
    A complete RAG pipeline that:
        1. Retrieves relevant documents
        2. Builds a prompt with context
        3. (In production) sends to LLM
    """

    def __init__(self, vector_store: SimpleVectorStore):
        self.vector_store = vector_store

    def build_prompt(self, query: str, top_k: int = 3) -> str:
        """
        Build a RAG prompt by:
            1. Retrieving top_k relevant documents
            2. Formatting them into a context string
            3. Combining with the query into a structured prompt

        Return format:
        '''
        Answer the question based on the following context.

        Context:
        [1] {doc1 content}
        [2] {doc2 content}
        ...

        Question: {query}
        Answer:'''
        """
        # YOUR CODE HERE
        pass

    def get_sources(self, query: str, top_k: int = 3) -> list[dict]:
        """
        Return source information for transparency/citations.
        Return list of dicts with: id, content_preview (first 100 chars), score, metadata
        """
        # YOUR CODE HERE
        pass


def build_chat_messages(system_prompt: str, user_message: str,
                        context: str, chat_history: list[dict] = None) -> list[dict]:
    """
    Build a properly formatted chat messages list for an LLM API call.
    This is the format used by OpenAI, Anthropic, etc.

    Format:
    [
        {"role": "system", "content": system_prompt + "\n\nContext:\n" + context},
        ...chat_history messages...,
        {"role": "user", "content": user_message}
    ]

    chat_history is a list of {"role": "user"/"assistant", "content": "..."} dicts.
    If chat_history is None, treat it as empty.
    """
    # YOUR CODE HERE
    pass


def simple_prompt_template(template: str, **kwargs) -> str:
    """
    A simple prompt template engine.
    Replace {variable_name} placeholders in the template with provided kwargs.
    Raise KeyError if a required variable is missing.

    Example:
        simple_prompt_template(
            "Summarize this {doc_type}: {content}",
            doc_type="article",
            content="Hello world"
        )
        -> "Summarize this article: Hello world"
    """
    # YOUR CODE HERE
    pass
