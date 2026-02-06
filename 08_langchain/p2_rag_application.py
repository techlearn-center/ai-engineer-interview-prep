"""
Problem 2: RAG Application Components
======================================
Difficulty: Medium

Build the components for a RAG (Retrieval-Augmented Generation) system.
Uses MOCK components - no API keys or databases needed.

Run tests:
    pytest 08_langchain/tests/test_p2_rag_application.py -v
"""
import re
import math
from dataclasses import dataclass, field


# ============================================================
# MOCK COMPONENTS
# ============================================================

@dataclass
class Document:
    """Represents a document with content and metadata."""
    page_content: str
    metadata: dict = field(default_factory=dict)


class MockEmbeddings:
    """Simulates an embedding model. Creates fake but consistent vectors."""

    def __init__(self, dimension: int = 8):
        self.dimension = dimension

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query."""
        return self._fake_embed(text)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple documents."""
        return [self._fake_embed(text) for text in texts]

    def _fake_embed(self, text: str) -> list[float]:
        """Create a deterministic fake embedding from text."""
        import hashlib
        # Use hash to create consistent "embeddings"
        hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
        result = []
        for i in range(self.dimension):
            # Generate pseudo-random but deterministic values
            val = ((hash_val >> (i * 4)) & 0xF) / 15.0 - 0.5
            result.append(val)
        return result


# ============================================================
# YOUR TASKS
# ============================================================


class TextSplitter:
    """
    TASK 1: Implement a text splitter.

    Splits long text into smaller chunks with overlap.
    This is essential for RAG - documents are often too long
    for embedding models or LLM context windows.

    Example:
        splitter = TextSplitter(chunk_size=100, chunk_overlap=20)
        chunks = splitter.split_text("Your very long document...")
        # Returns list of strings, each ~100 chars, with 20 char overlap

    Also implement split_documents to work with Document objects:
        docs = [Document(page_content="..."), ...]
        split_docs = splitter.split_documents(docs)
        # Returns more Document objects with same metadata

    Implement:
        - __init__(self, chunk_size: int = 500, chunk_overlap: int = 50)
        - split_text(self, text: str) -> list[str]
        - split_documents(self, documents: list[Document]) -> list[Document]

    Splitting strategy:
        1. Split on sentence boundaries first (. ! ?)
        2. Accumulate sentences until chunk_size is reached
        3. Include overlap from previous chunk
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        # YOUR CODE HERE
        pass

    def split_text(self, text: str) -> list[str]:
        # YOUR CODE HERE
        pass

    def split_documents(self, documents: list[Document]) -> list[Document]:
        # YOUR CODE HERE
        pass


class VectorStore:
    """
    TASK 2: Implement a simple vector store.

    Stores documents with their embeddings and enables similarity search.

    Example:
        embeddings = MockEmbeddings()
        store = VectorStore(embeddings)

        # Add documents
        docs = [Document(page_content="Python is great"), ...]
        store.add_documents(docs)

        # Search
        results = store.similarity_search("What is Python?", k=3)
        # Returns top 3 most similar documents

    Implement:
        - __init__(self, embeddings: MockEmbeddings)
        - add_documents(self, documents: list[Document])
            Store each doc with its embedding
        - add_texts(self, texts: list[str], metadatas: list[dict] = None)
            Create Documents from texts and add them
        - similarity_search(self, query: str, k: int = 4) -> list[Document]
            Return top k most similar documents
        - similarity_search_with_score(self, query: str, k: int = 4) -> list[tuple[Document, float]]
            Return documents with their similarity scores

    Use cosine similarity for comparing vectors.
    """

    def __init__(self, embeddings: MockEmbeddings):
        # YOUR CODE HERE
        pass

    def add_documents(self, documents: list[Document]):
        # YOUR CODE HERE
        pass

    def add_texts(self, texts: list[str], metadatas: list[dict] = None):
        # YOUR CODE HERE
        pass

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        # YOUR CODE HERE
        pass

    def similarity_search_with_score(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        # YOUR CODE HERE
        pass

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        """Compute cosine similarity between two vectors."""
        # YOUR CODE HERE
        pass


class Retriever:
    """
    TASK 3: Implement a retriever.

    A retriever wraps a vector store and provides a simple interface
    for finding relevant documents.

    Example:
        retriever = Retriever(vector_store, search_kwargs={"k": 5})
        docs = retriever.invoke("What is machine learning?")
        # Returns list of relevant Document objects

    Implement:
        - __init__(self, vector_store: VectorStore, search_kwargs: dict = None)
        - invoke(self, query: str) -> list[Document]
        - get_relevant_documents(self, query: str) -> list[Document]
            (alias for invoke, for compatibility)
    """

    def __init__(self, vector_store: VectorStore, search_kwargs: dict = None):
        # YOUR CODE HERE
        pass

    def invoke(self, query: str) -> list[Document]:
        # YOUR CODE HERE
        pass

    def get_relevant_documents(self, query: str) -> list[Document]:
        # YOUR CODE HERE
        pass


class RAGChain:
    """
    TASK 4: Implement a complete RAG chain.

    Combines retrieval and generation:
    1. Retrieve relevant documents for the question
    2. Format them into a context string
    3. Build a prompt with context + question
    4. Get response from LLM
    5. Return the answer

    Example:
        from p1_langchain_basics import MockLLM

        llm = MockLLM()
        rag = RAGChain(retriever, llm)
        result = rag.invoke("What is the refund policy?")
        # {"answer": "...", "source_documents": [...]}

    Implement:
        - __init__(self, retriever: Retriever, llm, prompt_template: str = None)
        - format_docs(self, docs: list[Document]) -> str
            Join document contents with newlines
        - invoke(self, question: str) -> dict
            Return {"answer": str, "source_documents": list[Document]}

    Default prompt template:
        "Answer based on the context below.\n\nContext:\n{context}\n\nQuestion: {question}\n\nAnswer:"
    """

    DEFAULT_TEMPLATE = """Answer based on the context below.

Context:
{context}

Question: {question}

Answer:"""

    def __init__(self, retriever: Retriever, llm, prompt_template: str = None):
        # YOUR CODE HERE
        pass

    def format_docs(self, docs: list[Document]) -> str:
        # YOUR CODE HERE
        pass

    def invoke(self, question: str) -> dict:
        # YOUR CODE HERE
        pass


class DocumentLoader:
    """
    TASK 5: Implement document loaders for different file types.

    In real LangChain, there are loaders for PDF, CSV, HTML, etc.
    Here we'll implement text and simple CSV loading.

    Example:
        # Text loader
        loader = TextLoader("notes.txt")
        docs = loader.load()

        # CSV loader (each row becomes a document)
        loader = CSVLoader("data.csv")
        docs = loader.load()

    Implement the TextLoader and CSVLoader classes.
    For testing, we'll pass content directly instead of file paths.
    """
    pass


class TextLoader:
    """
    Load a plain text file as a single Document.

    Example:
        loader = TextLoader(content="Hello world", source="notes.txt")
        docs = loader.load()
        # [Document(page_content="Hello world", metadata={"source": "notes.txt"})]
    """

    def __init__(self, content: str, source: str = "unknown"):
        # YOUR CODE HERE
        pass

    def load(self) -> list[Document]:
        # YOUR CODE HERE
        pass


class CSVLoader:
    """
    Load a CSV where each row becomes a Document.

    Example:
        csv_content = '''name,description
        Python,A programming language
        JavaScript,Web scripting language'''

        loader = CSVLoader(content=csv_content, source="languages.csv")
        docs = loader.load()
        # [
        #     Document(page_content="name: Python\ndescription: A programming language", ...),
        #     Document(page_content="name: JavaScript\ndescription: Web scripting language", ...),
        # ]
    """

    def __init__(self, content: str, source: str = "unknown"):
        # YOUR CODE HERE
        pass

    def load(self) -> list[Document]:
        # YOUR CODE HERE
        pass
