"""
SOLUTIONS - RAG Pipeline
==========================
Try to solve the problems yourself first!
"""
import numpy as np
from dataclasses import dataclass


@dataclass
class Document:
    id: str
    content: str
    metadata: dict


class SimpleVectorStore:
    """
    Key concepts to explain in interview:
    - Embeddings convert text to dense vectors that capture meaning
    - Similar texts have similar embeddings (high cosine similarity)
    - Vector stores enable fast similarity search
    - In production: Pinecone, ChromaDB, Weaviate, Qdrant, pgvector
    """

    def __init__(self, embedding_dim: int = 4):
        self.embedding_dim = embedding_dim
        self.documents: list[Document] = []
        self.embeddings: list[np.ndarray] = []

    def _fake_embed(self, text: str) -> np.ndarray:
        np.random.seed(hash(text) % 2**31)
        return np.random.randn(self.embedding_dim)

    def add_document(self, doc: Document):
        embedding = self._fake_embed(doc.content)
        self.documents.append(doc)
        self.embeddings.append(embedding)

    def similarity_search(self, query: str, top_k: int = 3) -> list[tuple[Document, float]]:
        """
        Key insight: This is the core retrieval step in RAG.
        Embed the query, then compare against all stored embeddings.
        """
        query_embedding = self._fake_embed(query)

        # Compute similarity to all documents
        scores = []
        for i, doc_embedding in enumerate(self.embeddings):
            sim = self.cosine_similarity(query_embedding, doc_embedding)
            scores.append((self.documents[i], sim))

        # Sort by similarity descending, take top_k
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        dot = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot / (norm_a * norm_b))


class RAGPipeline:
    """
    Key concepts to explain in interview:
    - RAG = Retrieval + Augmented Generation
    - Step 1: Retrieve relevant docs using vector similarity
    - Step 2: Stuff retrieved docs into the prompt as context
    - Step 3: Send augmented prompt to LLM
    - This grounds the LLM in factual data and reduces hallucination
    """

    def __init__(self, vector_store: SimpleVectorStore):
        self.vector_store = vector_store

    def build_prompt(self, query: str, top_k: int = 3) -> str:
        results = self.vector_store.similarity_search(query, top_k=top_k)

        # Build context from retrieved documents
        context_parts = []
        for i, (doc, score) in enumerate(results, 1):
            context_parts.append(f"[{i}] {doc.content}")

        context = "\n".join(context_parts)

        prompt = f"""Answer the question based on the following context.

Context:
{context}

Question: {query}
Answer:"""
        return prompt

    def get_sources(self, query: str, top_k: int = 3) -> list[dict]:
        results = self.vector_store.similarity_search(query, top_k=top_k)
        return [
            {
                "id": doc.id,
                "content_preview": doc.content[:100],
                "score": score,
                "metadata": doc.metadata,
            }
            for doc, score in results
        ]


def build_chat_messages(system_prompt: str, user_message: str,
                        context: str, chat_history: list[dict] = None) -> list[dict]:
    """
    Key insight: This is the exact format used by OpenAI, Anthropic, etc.
    System message sets behavior, then history, then current user message.
    Context goes in the system message so the LLM always sees it.
    """
    messages = [
        {"role": "system", "content": f"{system_prompt}\n\nContext:\n{context}"}
    ]

    if chat_history:
        messages.extend(chat_history)

    messages.append({"role": "user", "content": user_message})

    return messages


def simple_prompt_template(template: str, **kwargs) -> str:
    """
    Key insight: Python's str.format_map works, but we use format(**kwargs)
    which raises KeyError for missing variables automatically.
    """
    return template.format(**kwargs)
