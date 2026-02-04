# Learn: LLM & RAG Patterns for AI Engineers

Read this BEFORE attempting the problems.

---

## 1. What is RAG?

**RAG = Retrieval-Augmented Generation**

The problem: LLMs (ChatGPT, Claude, etc.) only know what they were trained on.
They can't answer questions about YOUR company's data, recent events, or private docs.

The solution: Before asking the LLM a question, FIND relevant documents first,
then paste them into the prompt as context.

```
User asks: "What is our refund policy?"

WITHOUT RAG:
    LLM says: "I don't know your specific refund policy..." (hallucination risk)

WITH RAG:
    Step 1: Search your docs -> Find "refund_policy.pdf"
    Step 2: Put the relevant text into the prompt
    Step 3: LLM says: "According to your policy, refunds are available within 30 days..."
```

---

## 2. How RAG Works (The Pipeline)

```
[User Question]
       |
       v
[1. EMBED the question]  -->  Convert text to a vector (list of numbers)
       |
       v
[2. SEARCH vector store] -->  Find documents with similar vectors
       |
       v
[3. BUILD prompt]        -->  Stuff retrieved docs into the prompt
       |
       v
[4. SEND to LLM]         -->  LLM answers using the context
       |
       v
[Answer + Sources]
```

---

## 3. What Are Embeddings?

An embedding converts text into a list of numbers (a vector) that captures its meaning.

```python
# Conceptually:
embed("king")   -> [0.9, 0.1, 0.8, ...]   # 1536 numbers
embed("queen")  -> [0.85, 0.15, 0.75, ...]  # similar to king!
embed("banana") -> [0.1, 0.9, 0.2, ...]     # very different
```

**Similar texts have similar embeddings.** This is the key insight.

In production you'd use:
- OpenAI: `text-embedding-3-small`
- Cohere: `embed-v3`
- Open source: `sentence-transformers`

For our practice, we use a fake embedding function (same concept, just random numbers).

---

## 4. Cosine Similarity

How do you measure if two embeddings are "similar"?

**Cosine similarity** measures the angle between two vectors:
- 1.0 = identical direction (very similar)
- 0.0 = perpendicular (unrelated)
- -1.0 = opposite (very different)

```python
import numpy as np

def cosine_similarity(a, b):
    dot_product = np.dot(a, b)          # How much they point same direction
    norm_a = np.linalg.norm(a)          # Length of vector a
    norm_b = np.linalg.norm(b)          # Length of vector b
    return dot_product / (norm_a * norm_b)

# Example:
a = np.array([1, 0])    # points right
b = np.array([0, 1])    # points up
cosine_similarity(a, b)  # 0.0 - perpendicular, unrelated

c = np.array([1, 0])
d = np.array([1, 0])
cosine_similarity(c, d)  # 1.0 - identical
```

---

## 5. Vector Store (Simple Version)

A vector store is just a database optimized for similarity search.

```python
class SimpleVectorStore:
    def __init__(self):
        self.documents = []    # The actual text
        self.embeddings = []   # The vector representations

    def add(self, text):
        embedding = embed(text)            # Convert text to vector
        self.documents.append(text)
        self.embeddings.append(embedding)

    def search(self, query, top_k=3):
        query_embedding = embed(query)

        # Compare query to ALL stored embeddings
        scores = []
        for i, doc_emb in enumerate(self.embeddings):
            sim = cosine_similarity(query_embedding, doc_emb)
            scores.append((self.documents[i], sim))

        # Return the top_k most similar
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]
```

In production: ChromaDB, Pinecone, Weaviate, Qdrant, pgvector

---

## 6. Building the RAG Prompt

After retrieving relevant docs, you stuff them into a prompt:

```python
def build_rag_prompt(query, retrieved_docs):
    # Format the context
    context = ""
    for i, doc in enumerate(retrieved_docs, 1):
        context += f"[{i}] {doc}\n"

    # Build the final prompt
    prompt = f"""Answer the question based on the following context.

Context:
{context}

Question: {query}
Answer:"""

    return prompt
```

Example output:
```
Answer the question based on the following context.

Context:
[1] Our refund policy allows returns within 30 days of purchase.
[2] Refunds are processed to the original payment method within 5 business days.

Question: What is the refund policy?
Answer:
```

---

## 7. Chat Message Format

When calling LLM APIs (OpenAI, Anthropic), you send messages in this format:

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant. Use the context provided."},
    {"role": "user", "content": "What is Python?"},
    {"role": "assistant", "content": "Python is a programming language."},
    {"role": "user", "content": "What is it used for?"},
]
```

Three roles:
- **system**: Sets the LLM's behavior (invisible to the user)
- **user**: The human's messages
- **assistant**: The LLM's previous responses

For RAG, you put the retrieved context in the system message:
```python
{"role": "system", "content": f"You are helpful.\n\nContext:\n{retrieved_docs}"}
```

---

## 8. Text Chunking (Why It Matters)

Documents are often too long to embed as one piece. So we split them into chunks:

```
Original: 5000-word document

Chunked (chunk_size=200, overlap=50):
  Chunk 1: words 1-200
  Chunk 2: words 150-350    (overlaps with chunk 1!)
  Chunk 3: words 300-500
  ...
```

**Why overlap?** If an important sentence falls right at the boundary between
two chunks, the overlap ensures it's fully captured in at least one chunk.

```python
def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []
    step = chunk_size - overlap

    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if chunk:
            chunks.append(chunk)
        if i + chunk_size >= len(words):
            break
    return chunks
```

---

## 9. Prompt Templates

Instead of hardcoding prompts, use templates:

```python
def prompt_template(template, **kwargs):
    return template.format(**kwargs)

# Usage:
template = "Summarize this {doc_type} in {language}: {content}"

prompt = prompt_template(
    template,
    doc_type="email",
    language="English",
    content="Dear team, we need to ship by Friday..."
)
# -> "Summarize this email in English: Dear team, we need to ship by Friday..."
```

This is what LangChain, LlamaIndex, etc. do under the hood.

---

## 10. Interview Vocabulary

| Term | Meaning |
|------|---------|
| **RAG** | Retrieval-Augmented Generation - search docs, then generate |
| **Embedding** | Vector representation of text that captures meaning |
| **Vector Store** | Database for storing and searching embeddings |
| **Cosine Similarity** | Measure of how similar two vectors are (-1 to 1) |
| **Chunking** | Splitting long docs into smaller pieces for embedding |
| **Prompt Engineering** | Designing prompts to get better LLM outputs |
| **Hallucination** | When an LLM makes up false information |
| **Grounding** | Providing factual context to reduce hallucination |
| **Top-k** | Retrieve the k most relevant documents |
| **Re-ranking** | A second pass to improve retrieval quality |
| **Hybrid Search** | Combining vector search with keyword search |
| **HyDE** | Hypothetical Document Embeddings - generate a fake answer, embed it, search with it |

---

## Now Try the Problems

Open `p1_rag_pipeline.py` and implement:
1. `SimpleVectorStore` - add documents, search by similarity
2. `RAGPipeline` - build prompts from retrieved context
3. `build_chat_messages` - format messages for LLM APIs
4. `simple_prompt_template` - template engine

```bash
pytest 06_llm_rag_patterns/tests/test_p1_rag_pipeline.py -v
```
