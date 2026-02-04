import sys
import os
import numpy as np
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from p1_rag_pipeline import (
    Document,
    SimpleVectorStore,
    RAGPipeline,
    build_chat_messages,
    simple_prompt_template,
)


class TestSimpleVectorStore:
    def setup_method(self):
        self.store = SimpleVectorStore(embedding_dim=4)
        self.docs = [
            Document("1", "Python is a programming language", {"topic": "programming"}),
            Document("2", "Machine learning uses algorithms", {"topic": "ml"}),
            Document("3", "Neural networks are deep learning models", {"topic": "dl"}),
        ]
        for doc in self.docs:
            self.store.add_document(doc)

    def test_add_documents(self):
        assert len(self.store.documents) == 3
        assert len(self.store.embeddings) == 3

    def test_embedding_shape(self):
        for emb in self.store.embeddings:
            assert emb.shape == (4,)

    def test_similarity_search_returns_results(self):
        results = self.store.similarity_search("python programming", top_k=2)
        assert len(results) == 2

    def test_similarity_search_returns_tuples(self):
        results = self.store.similarity_search("test query", top_k=1)
        assert len(results) == 1
        doc, score = results[0]
        assert isinstance(doc, Document)
        assert isinstance(score, float)

    def test_similarity_search_sorted_descending(self):
        results = self.store.similarity_search("test", top_k=3)
        scores = [score for _, score in results]
        assert scores == sorted(scores, reverse=True)

    def test_cosine_similarity_identical(self):
        a = np.array([1.0, 2.0, 3.0])
        sim = SimpleVectorStore.cosine_similarity(a, a)
        assert abs(sim - 1.0) < 1e-10

    def test_cosine_similarity_orthogonal(self):
        a = np.array([1.0, 0.0])
        b = np.array([0.0, 1.0])
        sim = SimpleVectorStore.cosine_similarity(a, b)
        assert abs(sim) < 1e-10


class TestRAGPipeline:
    def setup_method(self):
        store = SimpleVectorStore(embedding_dim=4)
        store.add_document(Document("1", "Python is great", {"source": "docs"}))
        store.add_document(Document("2", "ML is fun", {"source": "blog"}))
        self.pipeline = RAGPipeline(store)

    def test_build_prompt_format(self):
        prompt = self.pipeline.build_prompt("What is Python?", top_k=2)
        assert "Context:" in prompt
        assert "Question:" in prompt
        assert "Answer:" in prompt
        assert "[1]" in prompt

    def test_get_sources(self):
        sources = self.pipeline.get_sources("test", top_k=2)
        assert len(sources) == 2
        assert "id" in sources[0]
        assert "score" in sources[0]
        assert "content_preview" in sources[0]
        assert "metadata" in sources[0]


class TestBuildChatMessages:
    def test_basic(self):
        messages = build_chat_messages(
            system_prompt="You are helpful.",
            user_message="Hello",
            context="Some context",
        )
        assert messages[0]["role"] == "system"
        assert "You are helpful" in messages[0]["content"]
        assert "Some context" in messages[0]["content"]
        assert messages[-1]["role"] == "user"
        assert messages[-1]["content"] == "Hello"

    def test_with_history(self):
        history = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        messages = build_chat_messages(
            system_prompt="Be nice.",
            user_message="How are you?",
            context="ctx",
            chat_history=history,
        )
        assert len(messages) == 4  # system + 2 history + user
        assert messages[1]["role"] == "user"
        assert messages[2]["role"] == "assistant"

    def test_no_history(self):
        messages = build_chat_messages("sys", "hello", "ctx")
        assert len(messages) == 2


class TestPromptTemplate:
    def test_basic(self):
        result = simple_prompt_template(
            "Hello {name}, you are {age}",
            name="Alice",
            age="30",
        )
        assert result == "Hello Alice, you are 30"

    def test_missing_variable(self):
        with pytest.raises(KeyError):
            simple_prompt_template("Hello {name}", wrong_key="value")

    def test_no_variables(self):
        result = simple_prompt_template("Hello world")
        assert result == "Hello world"
