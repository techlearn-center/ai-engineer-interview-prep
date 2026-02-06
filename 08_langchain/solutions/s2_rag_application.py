"""
SOLUTIONS - RAG Application
=============================
Try to solve the problems yourself first!
"""
import re
import csv
import io
import math
from p2_rag_application import Document, MockEmbeddings


class TextSplitter:
    """Split text into chunks with overlap."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> list[str]:
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)

        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            if current_length + sentence_length > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(" ".join(current_chunk))

                # Start new chunk with overlap
                overlap_text = " ".join(current_chunk)
                if len(overlap_text) > self.chunk_overlap:
                    # Keep last part for overlap
                    overlap_start = len(overlap_text) - self.chunk_overlap
                    overlap = overlap_text[overlap_start:]
                    current_chunk = [overlap]
                    current_length = len(overlap)
                else:
                    current_chunk = []
                    current_length = 0

            current_chunk.append(sentence)
            current_length += sentence_length + 1  # +1 for space

        # Don't forget the last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks

    def split_documents(self, documents: list[Document]) -> list[Document]:
        result = []
        for doc in documents:
            chunks = self.split_text(doc.page_content)
            for i, chunk in enumerate(chunks):
                new_doc = Document(
                    page_content=chunk,
                    metadata={**doc.metadata, "chunk": i}
                )
                result.append(new_doc)
        return result


class VectorStore:
    """Store documents with embeddings for similarity search."""

    def __init__(self, embeddings: MockEmbeddings):
        self.embeddings = embeddings
        self.documents: list[Document] = []
        self.vectors: list[list[float]] = []

    def add_documents(self, documents: list[Document]):
        for doc in documents:
            vector = self.embeddings.embed_query(doc.page_content)
            self.documents.append(doc)
            self.vectors.append(vector)

    def add_texts(self, texts: list[str], metadatas: list[dict] = None):
        if metadatas is None:
            metadatas = [{} for _ in texts]

        docs = [
            Document(page_content=text, metadata=meta)
            for text, meta in zip(texts, metadatas)
        ]
        self.add_documents(docs)

    def similarity_search(self, query: str, k: int = 4) -> list[Document]:
        results = self.similarity_search_with_score(query, k)
        return [doc for doc, score in results]

    def similarity_search_with_score(self, query: str, k: int = 4) -> list[tuple[Document, float]]:
        query_vector = self.embeddings.embed_query(query)

        # Compute similarity with all documents
        scores = []
        for i, doc_vector in enumerate(self.vectors):
            similarity = self.cosine_similarity(query_vector, doc_vector)
            scores.append((self.documents[i], similarity))

        # Sort by similarity descending
        scores.sort(key=lambda x: x[1], reverse=True)

        return scores[:k]

    @staticmethod
    def cosine_similarity(a: list[float], b: list[float]) -> float:
        dot_product = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)


class Retriever:
    """Wrapper around vector store for document retrieval."""

    def __init__(self, vector_store: VectorStore, search_kwargs: dict = None):
        self.vector_store = vector_store
        self.search_kwargs = search_kwargs or {"k": 4}

    def invoke(self, query: str) -> list[Document]:
        k = self.search_kwargs.get("k", 4)
        return self.vector_store.similarity_search(query, k=k)

    def get_relevant_documents(self, query: str) -> list[Document]:
        return self.invoke(query)


class RAGChain:
    """Complete RAG chain: retrieve → format → prompt → generate."""

    DEFAULT_TEMPLATE = """Answer based on the context below.

Context:
{context}

Question: {question}

Answer:"""

    def __init__(self, retriever: Retriever, llm, prompt_template: str = None):
        self.retriever = retriever
        self.llm = llm
        self.prompt_template = prompt_template or self.DEFAULT_TEMPLATE

    def format_docs(self, docs: list[Document]) -> str:
        return "\n\n".join(doc.page_content for doc in docs)

    def invoke(self, question: str) -> dict:
        # 1. Retrieve relevant documents
        docs = self.retriever.invoke(question)

        # 2. Format context
        context = self.format_docs(docs)

        # 3. Build prompt
        prompt = self.prompt_template.format(context=context, question=question)

        # 4. Get response
        response = self.llm.invoke(prompt)

        return {
            "answer": response.content,
            "source_documents": docs,
        }


class TextLoader:
    """Load plain text as a Document."""

    def __init__(self, content: str, source: str = "unknown"):
        self.content = content
        self.source = source

    def load(self) -> list[Document]:
        return [Document(page_content=self.content, metadata={"source": self.source})]


class CSVLoader:
    """Load CSV where each row becomes a Document."""

    def __init__(self, content: str, source: str = "unknown"):
        self.content = content
        self.source = source

    def load(self) -> list[Document]:
        reader = csv.DictReader(io.StringIO(self.content))
        docs = []

        for i, row in enumerate(reader):
            # Format row as "key: value" pairs
            content = "\n".join(f"{k}: {v}" for k, v in row.items())
            doc = Document(
                page_content=content,
                metadata={"source": self.source, "row": i}
            )
            docs.append(doc)

        return docs
