import sys
import os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from p2_rag_application import (
    Document,
    MockEmbeddings,
    TextSplitter,
    VectorStore,
    Retriever,
    RAGChain,
    TextLoader,
    CSVLoader,
)
from p1_langchain_basics import MockLLM


class TestTextSplitter:
    def test_split_text_basic(self):
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
        text = "This is sentence one. This is sentence two. This is sentence three."
        chunks = splitter.split_text(text)
        assert len(chunks) >= 1
        assert all(isinstance(c, str) for c in chunks)

    def test_split_text_respects_size(self):
        splitter = TextSplitter(chunk_size=30, chunk_overlap=5)
        text = "Short. " * 20
        chunks = splitter.split_text(text)
        # Most chunks should be under chunk_size (allowing some flexibility)
        for chunk in chunks:
            assert len(chunk) < 100  # Reasonable upper bound

    def test_split_documents(self):
        splitter = TextSplitter(chunk_size=50, chunk_overlap=10)
        docs = [
            Document(page_content="First doc. Has content.", metadata={"source": "a.txt"}),
            Document(page_content="Second doc. More content.", metadata={"source": "b.txt"}),
        ]
        split_docs = splitter.split_documents(docs)
        assert len(split_docs) >= 2
        assert all(isinstance(d, Document) for d in split_docs)

    def test_metadata_preserved(self):
        splitter = TextSplitter(chunk_size=20, chunk_overlap=5)
        docs = [Document(page_content="Test content here.", metadata={"source": "test.txt"})]
        split_docs = splitter.split_documents(docs)
        for doc in split_docs:
            assert doc.metadata.get("source") == "test.txt"


class TestVectorStore:
    def test_add_documents(self):
        embeddings = MockEmbeddings()
        store = VectorStore(embeddings)
        docs = [
            Document(page_content="Python programming"),
            Document(page_content="JavaScript development"),
        ]
        store.add_documents(docs)
        # Should be able to search after adding
        results = store.similarity_search("Python", k=1)
        assert len(results) == 1

    def test_add_texts(self):
        embeddings = MockEmbeddings()
        store = VectorStore(embeddings)
        store.add_texts(
            ["Machine learning", "Deep learning"],
            [{"topic": "ml"}, {"topic": "dl"}]
        )
        results = store.similarity_search("learning", k=2)
        assert len(results) == 2

    def test_similarity_search(self):
        embeddings = MockEmbeddings()
        store = VectorStore(embeddings)
        store.add_texts([
            "Python is a programming language",
            "Cats are cute animals",
            "Python is great for data science",
        ])
        results = store.similarity_search("Python programming", k=2)
        assert len(results) == 2
        # First result should be more relevant to Python
        assert "Python" in results[0].page_content

    def test_similarity_search_with_score(self):
        embeddings = MockEmbeddings()
        store = VectorStore(embeddings)
        store.add_texts(["Test document"])
        results = store.similarity_search_with_score("Test", k=1)
        assert len(results) == 1
        doc, score = results[0]
        assert isinstance(doc, Document)
        assert isinstance(score, float)
        assert -1 <= score <= 1  # Cosine similarity range

    def test_cosine_similarity(self):
        # Identical vectors should have similarity 1
        a = [1.0, 0.0, 0.0]
        b = [1.0, 0.0, 0.0]
        assert abs(VectorStore.cosine_similarity(a, b) - 1.0) < 0.0001

        # Orthogonal vectors should have similarity 0
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert abs(VectorStore.cosine_similarity(a, b)) < 0.0001


class TestRetriever:
    def test_invoke(self):
        embeddings = MockEmbeddings()
        store = VectorStore(embeddings)
        store.add_texts(["Document one", "Document two", "Document three"])

        retriever = Retriever(store, search_kwargs={"k": 2})
        docs = retriever.invoke("Document")
        assert len(docs) == 2

    def test_get_relevant_documents(self):
        embeddings = MockEmbeddings()
        store = VectorStore(embeddings)
        store.add_texts(["Test content"])

        retriever = Retriever(store)
        docs = retriever.get_relevant_documents("Test")
        assert len(docs) > 0


class TestRAGChain:
    def test_format_docs(self):
        embeddings = MockEmbeddings()
        store = VectorStore(embeddings)
        store.add_texts(["First doc", "Second doc"])
        retriever = Retriever(store)
        llm = MockLLM()
        rag = RAGChain(retriever, llm)

        docs = [Document(page_content="A"), Document(page_content="B")]
        formatted = rag.format_docs(docs)
        assert "A" in formatted
        assert "B" in formatted

    def test_invoke(self):
        embeddings = MockEmbeddings()
        store = VectorStore(embeddings)
        store.add_texts([
            "The refund policy allows returns within 30 days.",
            "Contact support at help@example.com.",
        ])
        retriever = Retriever(store, search_kwargs={"k": 1})
        llm = MockLLM({"refund": "You can return items within 30 days."})

        rag = RAGChain(retriever, llm)
        result = rag.invoke("What is the refund policy?")

        assert "answer" in result
        assert "source_documents" in result
        assert isinstance(result["source_documents"], list)


class TestTextLoader:
    def test_load(self):
        loader = TextLoader(content="Hello world", source="test.txt")
        docs = loader.load()
        assert len(docs) == 1
        assert docs[0].page_content == "Hello world"
        assert docs[0].metadata["source"] == "test.txt"


class TestCSVLoader:
    def test_load(self):
        csv_content = """name,description
Python,A programming language
JavaScript,Web scripting"""

        loader = CSVLoader(content=csv_content, source="langs.csv")
        docs = loader.load()

        assert len(docs) == 2
        assert "Python" in docs[0].page_content
        assert "JavaScript" in docs[1].page_content

    def test_metadata(self):
        csv_content = """col1,col2
a,b"""
        loader = CSVLoader(content=csv_content, source="test.csv")
        docs = loader.load()
        assert docs[0].metadata["source"] == "test.csv"
        assert "row" in docs[0].metadata
