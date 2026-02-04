# Hints - LLM & RAG Patterns

## P1: RAG Pipeline

### SimpleVectorStore.add_document
- Embed the document content: `embedding = self._fake_embed(doc.content)`
- Append both doc and embedding to their respective lists

### SimpleVectorStore.similarity_search
1. Embed the query: `query_emb = self._fake_embed(query)`
2. Compute cosine similarity with every stored embedding
3. Create list of (document, score) tuples
4. Sort by score descending
5. Return first `top_k` items

### cosine_similarity
```python
dot = np.dot(a, b)
norm_a = np.linalg.norm(a)
norm_b = np.linalg.norm(b)
return dot / (norm_a * norm_b)
```
- Handle zero norms: if either norm is 0, return 0.0

### RAGPipeline.build_prompt
- Call `self.vector_store.similarity_search(query, top_k)`
- Format each doc as `[i] {content}`
- Combine into the template:
```
Answer the question based on the following context.

Context:
[1] doc1 content
[2] doc2 content

Question: {query}
Answer:
```

### build_chat_messages
```python
messages = [
    {"role": "system", "content": f"{system_prompt}\n\nContext:\n{context}"}
]
if chat_history:
    messages.extend(chat_history)
messages.append({"role": "user", "content": user_message})
```
- This is the EXACT format used by OpenAI and Anthropic APIs
- System message sets behavior + provides context
- History maintains conversation state
- User message is always last

### simple_prompt_template
- Python's `str.format(**kwargs)` does exactly this
- It raises KeyError automatically for missing variables
- One-liner: `return template.format(**kwargs)`

### Key interview talking points about RAG
1. **Why RAG?** - LLMs have knowledge cutoffs and can hallucinate. RAG grounds them in your data.
2. **Chunking matters** - Too small = no context, too large = diluted relevance
3. **Embedding models** - OpenAI text-embedding-3-small, Cohere embed-v3, open-source options
4. **Vector DBs** - Pinecone (managed), ChromaDB (local), Weaviate, Qdrant, pgvector
5. **Evaluation** - Retrieval metrics (recall@k, MRR) + generation quality (faithfulness, relevance)
6. **Advanced patterns** - Hybrid search (vector + keyword), re-ranking, query expansion, HyDE
