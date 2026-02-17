# AI Engineering Trade-offs & Decision Guide

> A practical reference for AI system design interviews. Every section contains comparison
> tables, decision criteria, cost analysis, and concrete recommendations with real numbers.

---

## 1. RAG vs Fine-tuning vs Prompt Engineering

### Comparison Table

| Dimension              | Prompt Engineering      | RAG                          | Fine-tuning                  |
|------------------------|-------------------------|------------------------------|------------------------------|
| **Upfront Cost**       | ~$0                     | $500-5,000 (infra + embedding) | $50-10,000+ (compute + data) |
| **Ongoing Cost**       | Token cost only         | Vector DB + retrieval + tokens | Inference cost (can be lower per-token) |
| **Time to Deploy**     | Hours                   | Days to weeks                | Weeks to months              |
| **Data Requirement**   | 0 examples (few-shot: 3-10) | 100+ documents            | 500-100,000+ labeled examples |
| **Accuracy (domain)**  | Low-Medium              | High                         | High-Very High               |
| **Latency**            | Lowest (~200ms)         | Medium (+100-500ms retrieval) | Lowest for self-hosted       |
| **Update Frequency**   | Instant                 | Minutes (re-index docs)      | Hours-days (re-train)        |
| **Hallucination Risk** | Highest                 | Low (grounded in sources)    | Medium (memorized patterns)  |
| **Context Window Use** | Consumes context        | Consumes context             | Frees up context             |
| **Expertise Needed**   | Low                     | Medium                       | High                         |

### Decision Flowchart

```
START: "I need the model to know domain-specific information"
  |
  +-- Is the knowledge already in the base model?
  |     YES --> Prompt Engineering (few-shot, system prompts)
  |     NO  --> Continue
  |
  +-- Does the knowledge change frequently (weekly or more)?
  |     YES --> RAG (documents can be re-indexed without retraining)
  |     NO  --> Continue
  |
  +-- Do you need the model to cite sources or show provenance?
  |     YES --> RAG
  |     NO  --> Continue
  |
  +-- Do you need to change the model's STYLE or BEHAVIOR (not just knowledge)?
  |     YES --> Fine-tuning
  |     NO  --> Continue
  |
  +-- Do you have 1,000+ high-quality labeled examples?
  |     YES --> Fine-tuning (likely better quality + lower per-token cost)
  |     NO  --> RAG
  |
  +-- Budget < $500/month?
        YES --> RAG with a managed vector DB (Pinecone free tier, ChromaDB)
        NO  --> Evaluate hybrid RAG + fine-tuning
```

### Hybrid Approaches

| Hybrid Pattern                  | When to Use                                                        | Example                                           |
|---------------------------------|--------------------------------------------------------------------|----------------------------------------------------|
| RAG + Prompt Engineering        | Most common starting point; ground responses in retrieved docs     | Customer support bot with knowledge base           |
| Fine-tuning + RAG               | Need domain style AND up-to-date knowledge                        | Legal assistant trained on legal writing + case law |
| Fine-tuning + Prompt Eng.       | Need consistent output format with domain expertise               | Medical coding system                              |
| All three                       | Enterprise systems with strict requirements                       | Financial compliance system                        |

### Cost Analysis Example (1M queries/month, avg 500 input + 200 output tokens)

| Approach              | Monthly Cost Estimate |
|-----------------------|----------------------|
| **Prompt Engineering** (GPT-4o-mini) | ~$150 (tokens only) |
| **RAG** (GPT-4o-mini + Pinecone Standard) | ~$250 (tokens + vector DB + embedding) |
| **Fine-tuning** (GPT-4o-mini fine-tuned) | ~$200 (training: ~$50 one-time, inference: ~$180) |
| **RAG + Fine-tuning** | ~$350 |

### When Each Approach Fails

- **Prompt Engineering fails** when: domain knowledge is not in the base model's training data, output format needs to be highly consistent, or context window is too small for all the instructions.
- **RAG fails** when: the answer requires synthesizing information across dozens of documents, questions are ambiguous and hard to embed, latency budget is very tight (<200ms), or documents are poorly structured.
- **Fine-tuning fails** when: the domain changes rapidly, you lack high-quality training data, you need source attribution, or the base model already handles the task well with prompting.

---

## 2. Vector Database Comparison

### Feature Matrix

| Feature               | Pinecone      | ChromaDB       | Weaviate       | Qdrant         | Milvus         | pgvector       | Elasticsearch  |
|-----------------------|---------------|----------------|----------------|----------------|----------------|----------------|----------------|
| **Hosting**           | Managed only  | Self-hosted/embedded | Both      | Both           | Both           | Self-hosted (with PG) | Both      |
| **Free Tier**         | Yes (100K vectors) | Unlimited (local) | Yes (sandbox) | Yes (1GB cloud) | Open source | Open source  | Open source    |
| **Max Vectors**       | Billions      | Millions       | Billions       | Billions       | Billions       | Millions       | Billions       |
| **Hybrid Search**     | Yes (sparse+dense) | No (dense only) | Yes (BM25+vector) | Yes (sparse+dense) | Yes       | Limited        | Yes (native BM25+vector) |
| **Metadata Filtering**| Yes           | Yes            | Yes            | Yes (advanced) | Yes            | SQL-based      | Yes (advanced) |
| **Multi-tenancy**     | Namespaces    | Collections    | Native         | Collections    | Partitions     | Schemas/tables | Indices        |
| **HNSW Support**      | Yes           | Yes            | Yes            | Yes            | Yes            | Yes (0.5.0+)   | Yes            |
| **Disk-based Index**  | No            | Yes            | No             | Yes (mmap)     | Yes            | Yes            | Yes            |
| **GPU Acceleration**  | N/A (managed) | No             | No             | No             | Yes            | No             | No             |
| **Quantization**      | N/A           | No             | PQ, BQ         | Scalar, PQ     | PQ, SQ         | No             | No             |
| **Managed Pricing**   | $70+/mo (Standard) | N/A       | ~$25+/mo       | $25+/mo        | $65+/mo (Zilliz) | N/A          | $95+/mo        |

### Approximate Performance Benchmarks (1M vectors, 768 dimensions, p99 latency)

| Database       | QPS (queries/sec) | p99 Latency | Recall@10 | Memory Usage  |
|----------------|-------------------|-------------|-----------|---------------|
| Pinecone       | ~1,000            | ~15ms       | 0.95+     | Managed       |
| Qdrant         | ~2,500            | ~8ms        | 0.97+     | ~4 GB         |
| Weaviate       | ~1,500            | ~12ms       | 0.95+     | ~6 GB         |
| Milvus         | ~3,000            | ~6ms        | 0.96+     | ~5 GB         |
| pgvector       | ~300              | ~50ms       | 0.92+     | PG-dependent  |
| ChromaDB       | ~500              | ~30ms       | 0.94+     | ~3 GB         |
| Elasticsearch  | ~800              | ~20ms       | 0.93+     | ~8 GB         |

*Note: benchmarks vary significantly based on hardware, configuration, and workload. These are approximate.*

### Decision Guide

```
START: "Which vector database should I use?"
  |
  +-- Is this a prototype / local dev?
  |     YES --> ChromaDB (embedded, zero config, pip install)
  |     NO  --> Continue
  |
  +-- Do you already run PostgreSQL in production?
  |     YES --> pgvector (add extension, no new infra)
  |     NO  --> Continue
  |
  +-- Do you need managed infrastructure (no DevOps team)?
  |     YES --> Pinecone (simplest managed) or Qdrant Cloud
  |     NO  --> Continue
  |
  +-- Do you need hybrid search (vector + keyword/BM25)?
  |     YES --> Weaviate, Elasticsearch, or Qdrant
  |     NO  --> Continue
  |
  +-- Scale: > 100M vectors?
  |     YES --> Milvus (designed for massive scale) or Pinecone
  |     NO  --> Continue
  |
  +-- Need maximum performance / lowest latency?
  |     YES --> Qdrant or Milvus
  |     NO  --> Pinecone (simplest) or Weaviate (good all-around)
```

### Production Readiness Assessment

| Database       | Maturity | Community  | Enterprise Support | Production Track Record |
|----------------|----------|------------|-------------------|-------------------------|
| Pinecone       | High     | Medium     | Yes               | Widely used in startups  |
| Weaviate       | High     | Large      | Yes               | Growing enterprise use   |
| Qdrant         | High     | Growing    | Yes               | Strong in EU market      |
| Milvus         | High     | Large      | Yes (Zilliz)      | Used at large scale      |
| pgvector       | Medium   | Large (PG) | Via PG providers  | Good for small-medium    |
| ChromaDB       | Medium   | Growing    | Limited           | Mostly prototypes        |
| Elasticsearch  | High     | Very Large | Yes (Elastic)     | Proven at massive scale  |

---

## 3. Embedding Model Selection

### Comparison Table

| Model                          | Dimensions | Cost/1M tokens | MTEB Avg Score | Max Tokens | Latency (avg) | Provider  |
|--------------------------------|------------|-----------------|----------------|------------|----------------|-----------|
| text-embedding-3-small         | 1,536      | $0.02           | 62.3           | 8,191      | ~30ms          | OpenAI    |
| text-embedding-3-large         | 3,072      | $0.13           | 64.6           | 8,191      | ~50ms          | OpenAI    |
| embed-v3 (English)             | 1,024      | $0.10           | 64.5           | 512        | ~40ms          | Cohere    |
| voyage-3                       | 1,024      | $0.06           | 67.1           | 32,000     | ~45ms          | Voyage AI |
| voyage-3-lite                  | 512        | $0.02           | 62.4           | 32,000     | ~25ms          | Voyage AI |
| BGE-large-en-v1.5              | 1,024      | Free (self-host)| 63.6           | 512        | ~15ms (local)  | BAAI      |
| E5-large-v2                    | 1,024      | Free (self-host)| 62.0           | 512        | ~15ms (local)  | Microsoft |
| all-MiniLM-L6-v2               | 384        | Free (self-host)| 56.3           | 256        | ~5ms (local)   | SBERT     |
| Gemini text-embedding-004      | 768        | $0.00625        | 66.0           | 2,048      | ~35ms          | Google    |

### When to Use Which

| Use Case                        | Recommended Model                  | Why                                         |
|---------------------------------|------------------------------------|---------------------------------------------|
| **Prototype / hackathon**       | all-MiniLM-L6-v2                   | Free, fast, good enough for demos           |
| **Production (cost-sensitive)** | text-embedding-3-small             | Cheapest API, decent quality                 |
| **Production (quality-first)**  | voyage-3 or text-embedding-3-large | Highest MTEB scores                         |
| **Code search**                 | voyage-code-3                      | Specifically trained for code                |
| **Long documents**              | voyage-3 (32K context)             | Handles full documents without chunking      |
| **On-premise / air-gapped**     | BGE-large-en-v1.5                  | Best open-source quality, self-hostable      |
| **Edge / mobile**               | all-MiniLM-L6-v2                   | Smallest model, runs on CPU                  |
| **Multi-lingual**               | embed-v3 (multilingual) or BGE-M3  | Trained on 100+ languages                   |

### Fine-tuning Embeddings vs Off-the-Shelf

| Criterion                 | Off-the-Shelf              | Fine-tuned                        |
|---------------------------|----------------------------|-----------------------------------|
| **Setup time**            | Minutes                    | Days to weeks                     |
| **Data requirement**      | None                       | 1,000+ query-document pairs       |
| **Quality improvement**   | Baseline                   | +5-15% on domain-specific tasks   |
| **Cost**                  | API pricing only           | Training compute + API pricing    |
| **Maintenance**           | None                       | Retrain when domain evolves       |
| **When to fine-tune**     | --                         | Specialized domain (legal, medical, code), low retrieval recall with off-the-shelf |

**Rule of thumb:** Start with off-the-shelf. Only fine-tune if retrieval recall is below 80% on your evaluation set after optimizing chunking and search strategies.

### Multi-lingual Considerations

- **Cohere embed-v3 multilingual**: 100+ languages, single model for all; best for mixed-language corpora.
- **BGE-M3**: Open-source multilingual; supports dense, sparse, and ColBERT retrieval in one model.
- **Language-specific models** (e.g., Japanese E5): Better for single-language use cases but require separate models per language.
- **Cross-lingual retrieval**: Query in English, retrieve in French -- Cohere and BGE-M3 handle this well.
- **Pitfall**: Token costs increase ~30-50% for non-Latin scripts (CJK, Arabic) due to tokenization inefficiency.

---

## 4. LLM Model Selection

### Comparison Table (as of early 2026)

| Model                | Input $/1M tokens | Output $/1M tokens | Context Window | Latency (TTFT) | Quality Tier |
|----------------------|--------------------|--------------------|----------------|-----------------|--------------|
| GPT-4o               | $2.50              | $10.00             | 128K           | ~300ms          | Tier 1       |
| GPT-4o-mini          | $0.15              | $0.60              | 128K           | ~150ms          | Tier 2       |
| Claude Sonnet 4.5    | $3.00              | $15.00             | 200K           | ~350ms          | Tier 1       |
| Claude Haiku 3.5     | $0.80              | $4.00              | 200K           | ~150ms          | Tier 2       |
| Gemini 2.0 Flash     | $0.10              | $0.40              | 1M             | ~200ms          | Tier 2       |
| Gemini 2.0 Pro       | $1.25              | $5.00              | 1M             | ~300ms          | Tier 1       |
| Llama 3.1 70B        | Self-host or $0.50-0.90 | $0.50-0.90    | 128K           | ~400ms          | Tier 2       |
| Llama 3.1 405B       | Self-host or $2.00-3.00 | $2.00-3.00    | 128K           | ~800ms          | Tier 1       |
| Mistral Large        | $2.00              | $6.00              | 128K           | ~300ms          | Tier 1-2     |
| Mixtral 8x22B        | Self-host or $0.60 | $0.60              | 64K            | ~350ms          | Tier 2       |

*Prices are approximate and change frequently. Always check provider pricing pages.*

### Model Routing Strategies

Route queries to different models based on complexity to optimize cost:

```
User Query --> Complexity Classifier (lightweight model or heuristic)
  |
  +-- Simple (greeting, FAQ, lookup)      --> GPT-4o-mini / Haiku ($0.15-0.80/1M)
  +-- Medium (summarization, Q&A)         --> GPT-4o-mini / Gemini Flash ($0.15-0.40/1M)
  +-- Complex (reasoning, code gen, math) --> GPT-4o / Claude Sonnet ($2.50-3.00/1M)
  +-- Critical (legal, medical, finance)  --> GPT-4o / Claude Sonnet + verification step
```

**Complexity classification approaches:**
1. **Keyword heuristic**: Short queries (<10 tokens) likely simple; queries with "compare", "analyze", "explain why" likely complex.
2. **Lightweight classifier**: Fine-tune a small model on query-complexity labels.
3. **Token budget**: Estimate output length needed; route long-output tasks to cheaper models.

**Cost savings example:** A 70/20/10 split (simple/medium/complex) on 1M queries/month:
- Without routing (all GPT-4o): ~$2,500/mo
- With routing: ~$700/mo (72% savings)

### Open-Source vs Proprietary Decision Framework

| Factor                    | Choose Open-Source (Llama, Mistral) | Choose Proprietary (GPT-4o, Claude) |
|---------------------------|-------------------------------------|--------------------------------------|
| **Data privacy**          | Sensitive data, regulatory needs    | Non-sensitive, cloud-acceptable      |
| **Cost at scale**         | >10M tokens/day (amortize GPU)     | <10M tokens/day                      |
| **Customization**         | Need to fine-tune or modify         | Off-the-shelf is sufficient          |
| **Latency control**       | Need guaranteed SLAs                | Acceptable variability               |
| **Quality requirement**   | Tier 2 acceptable                   | Need Tier 1 quality                  |
| **Team expertise**        | Have ML/infra engineers             | Small team, no ML infra expertise    |
| **Vendor lock-in concern**| High concern                        | Acceptable risk                      |

### On-Premise vs API Trade-offs

| Dimension          | On-Premise / Self-Hosted       | API (OpenAI, Anthropic, etc.)  |
|--------------------|--------------------------------|--------------------------------|
| **Capital cost**   | $20K-200K+ (GPUs)             | $0 upfront                     |
| **Operational cost**| $2K-10K/mo (power, cooling, staff) | Pay-per-token              |
| **Break-even**     | ~6-12 months at high volume    | Cheaper below break-even       |
| **Scalability**    | Limited by hardware            | Near-infinite                  |
| **Data residency** | Full control                   | Data leaves your network       |
| **Model updates**  | Manual                         | Automatic                      |
| **Uptime SLA**     | Your responsibility            | 99.9%+ from provider           |

---

## 5. Chunking Strategies

### Comparison Table

| Strategy                    | How It Works                                           | Best For                        | Chunk Size     |
|-----------------------------|--------------------------------------------------------|---------------------------------|----------------|
| **Fixed-size**              | Split every N characters/tokens                        | Simple, uniform documents       | 256-1024 tokens |
| **Recursive character**     | Split by hierarchy: `\n\n` then `\n` then `. ` then ` ` | General-purpose text          | 256-1024 tokens |
| **Sentence-based**          | Split on sentence boundaries (NLP sentence tokenizer)  | Conversational content          | 3-10 sentences |
| **Semantic**                | Group sentences by embedding similarity                 | Documents with topic shifts     | Variable       |
| **Document-structure-aware**| Split by markdown headers, HTML tags, or PDF sections   | Structured documents            | Variable       |
| **Code-aware**              | Split by functions, classes, or code blocks             | Source code repositories        | Per function   |

### Implementation Examples

**Fixed-size chunking:**
```python
def fixed_size_chunk(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks
```

**Recursive character splitting (LangChain-style):**
```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["\n\n", "\n", ". ", " ", ""]
)
chunks = splitter.split_text(document)
```

**Semantic chunking:**
```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_chunk(sentences: list[str], threshold: float = 0.75) -> list[str]:
    embeddings = model.encode(sentences)
    chunks, current_chunk = [], [sentences[0]]
    for i in range(1, len(sentences)):
        sim = np.dot(embeddings[i], embeddings[i-1]) / (
            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[i-1])
        )
        if sim < threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i]]
        else:
            current_chunk.append(sentences[i])
    chunks.append(" ".join(current_chunk))
    return chunks
```

### Pros and Cons

| Strategy              | Pros                                          | Cons                                           |
|-----------------------|-----------------------------------------------|------------------------------------------------|
| Fixed-size            | Simple, predictable size, fast                 | Breaks mid-sentence, loses context             |
| Recursive character   | Respects natural boundaries, good default      | May still split related content                |
| Sentence-based        | Preserves sentence integrity                   | Uneven chunk sizes, needs NLP library          |
| Semantic              | Groups related content                         | Expensive (requires embedding each sentence), slower |
| Document-structure    | Preserves document hierarchy                   | Requires structured input, variable chunk sizes |

### Optimal Chunk Sizes by Use Case

| Use Case                      | Recommended Chunk Size | Overlap | Rationale                                    |
|-------------------------------|------------------------|---------|----------------------------------------------|
| Q&A over documentation        | 256-512 tokens         | 20%     | Small chunks for precise retrieval           |
| Summarization                 | 1,024-2,048 tokens     | 10%     | Larger chunks preserve context               |
| Code search                   | Per function/class     | 0%      | Natural boundaries, complete units           |
| Legal document analysis       | 512-1,024 tokens       | 20%     | Balance precision with clause context        |
| Chat / conversational         | 128-256 tokens         | 30%     | Short, focused responses                     |

### Impact on Retrieval Quality

- **Too small** (< 100 tokens): Loses context, retrieves fragments that are hard to use.
- **Too large** (> 2,000 tokens): Dilutes relevance signal, wastes context window, retrieves noise with signal.
- **Sweet spot**: 256-512 tokens for most Q&A applications.
- **Overlap** (10-20%): Prevents losing information at chunk boundaries. Higher overlap increases storage but improves recall.

---

## 6. Retrieval Strategies

### Strategy Comparison

| Strategy                  | Quality | Latency  | Cost     | Complexity | When to Use                          |
|---------------------------|---------|----------|----------|------------|--------------------------------------|
| **Naive vector search**   | Medium  | ~10ms    | Low      | Low        | MVP, simple Q&A                      |
| **Hybrid (vector + BM25)**| High    | ~20ms    | Low-Med  | Medium     | Keyword-sensitive queries            |
| **Re-ranking**            | Very High| ~100-200ms| Medium  | Medium     | When precision matters more than speed|
| **HyDE**                  | High    | ~300ms+  | High     | Medium     | Poorly-phrased queries               |
| **Multi-query**           | High    | ~200ms+  | High     | Medium     | Ambiguous queries                    |
| **Contextual compression**| High    | ~300ms+  | High     | High       | Long documents, limited context      |
| **ColBERT / late interaction** | Very High | ~15ms | Medium | High    | When you need both speed and quality |

### Architecture Patterns

**Naive Vector Search:**
```
Query --> Embed --> Vector DB (top-k) --> LLM --> Response
```

**Hybrid Search (Vector + BM25):**
```
Query --+--> Embed --> Vector DB (top-k dense results)
        |                                              +--> Reciprocal Rank Fusion --> LLM --> Response
        +--> BM25 Index (top-k sparse results) -------+
```

**Re-ranking Pipeline:**
```
Query --> Embed --> Vector DB (top-50) --> Cross-Encoder Re-ranker (top-5) --> LLM --> Response
```

**HyDE (Hypothetical Document Embeddings):**
```
Query --> LLM (generate hypothetical answer) --> Embed hypothetical --> Vector DB (top-k) --> LLM --> Response
```

**Multi-Query Retrieval:**
```
Query --> LLM (generate 3-5 query variants) --> Embed each --> Vector DB --> Deduplicate + Merge --> LLM --> Response
```

### Quality vs Latency vs Cost Trade-offs

| Pipeline                         | Total Latency | Monthly Cost (1M queries) | Recall@5 Improvement |
|----------------------------------|---------------|---------------------------|----------------------|
| Naive vector                     | ~50ms         | ~$50                      | Baseline             |
| + BM25 hybrid                    | ~70ms         | ~$60                      | +5-10%               |
| + Re-ranking (Cohere)            | ~200ms        | ~$160                     | +10-20%              |
| + HyDE                           | ~500ms        | ~$300                     | +5-15%               |
| + Multi-query + Re-ranking       | ~600ms        | ~$500                     | +15-25%              |

### When to Add Re-ranking

Add re-ranking when:
- Retrieval recall is below 85% on your eval set.
- Users complain about irrelevant results.
- You have a latency budget of 200ms+ for retrieval.
- The cost of a bad answer exceeds the cost of re-ranking ($1/1,000 queries with Cohere).

**Re-ranking options:**
| Re-ranker              | Cost/1K queries | Latency | Quality |
|------------------------|-----------------|---------|---------|
| Cohere Rerank v3       | ~$1.00          | ~100ms  | High    |
| Cross-encoder (self-hosted) | GPU cost   | ~50ms   | High    |
| Jina Reranker v2       | ~$0.50          | ~80ms   | Medium-High |
| Flashrank (local)      | Free            | ~30ms   | Medium  |

### Evaluation Metrics

| Metric        | What It Measures                                    | Good Score | When to Use                |
|---------------|-----------------------------------------------------|------------|----------------------------|
| **MRR**       | Rank of first relevant result (1/rank, averaged)    | > 0.7      | Single-answer retrieval    |
| **NDCG@K**    | Quality of ranking considering position and graded relevance | > 0.6 | Ranked results             |
| **Recall@K**  | Fraction of relevant docs found in top-K            | > 0.85     | Most common RAG metric     |
| **MAP**       | Mean of precision at each relevant result position  | > 0.6      | Multiple relevant docs     |
| **Hit Rate**  | Whether any relevant doc appears in top-K           | > 0.90     | Simple pass/fail metric    |

---

## 7. Agent Architectures

### Pattern Comparison

| Pattern              | Complexity | Reliability | Cost     | Best For                             |
|----------------------|------------|-------------|----------|--------------------------------------|
| **ReAct**            | Low        | Medium      | Medium   | Simple tool-use tasks                |
| **Function Calling** | Low        | High        | Low-Med  | Structured API interactions          |
| **Plan-and-Execute** | Medium     | High        | High     | Multi-step tasks with dependencies   |
| **Multi-Agent**      | High       | Medium      | High     | Complex workflows with specialization|
| **LangGraph / State Machine** | High | Very High | Medium  | Production agents with control flow  |

### Architecture Descriptions

**ReAct (Reason + Act):**
```
Loop:
  1. Thought: "I need to look up the weather in NYC"
  2. Action: call_weather_api(location="NYC")
  3. Observation: "72F, sunny"
  4. Thought: "I now have the answer"
  5. Final Answer: "It's 72F and sunny in NYC"
```
- Pros: Simple, interpretable reasoning trace.
- Cons: Can loop indefinitely, inconsistent tool selection.

**Function Calling (OpenAI / Anthropic native):**
```
System: You have these functions: [get_weather, search_docs, calculate]
User: "What's the weather in NYC?"
Assistant: function_call(get_weather, {location: "NYC"})
Tool Result: {temp: 72, condition: "sunny"}
Assistant: "It's 72F and sunny in NYC"
```
- Pros: Reliable structured output, provider-optimized, parallel tool calls.
- Cons: Vendor-specific, limited reasoning transparency.

**Plan-and-Execute:**
```
1. Planner LLM: Create step-by-step plan
   - Step 1: Search for competitor pricing
   - Step 2: Get our current pricing
   - Step 3: Create comparison table
   - Step 4: Generate recommendations
2. Executor: Execute each step, re-plan if needed
3. Verifier: Check if the plan is complete
```
- Pros: Better for complex tasks, can recover from failures.
- Cons: Higher latency and cost (multiple LLM calls for planning).

**Multi-Agent:**
```
Orchestrator Agent
  |
  +-- Research Agent (searches web, reads docs)
  +-- Analysis Agent (processes data, creates charts)
  +-- Writing Agent (drafts reports)
  +-- Review Agent (checks quality, fact-checks)
```
- Pros: Specialization, parallel execution.
- Cons: Complex coordination, higher cost, harder to debug.

### Error Handling and Retry Strategies

| Strategy                  | Implementation                                | When to Use                      |
|---------------------------|-----------------------------------------------|----------------------------------|
| **Simple retry**          | Retry same action up to 3 times               | Transient API failures           |
| **Retry with backoff**    | Exponential backoff (1s, 2s, 4s)              | Rate limiting                    |
| **Fallback tool**         | Try alternative tool if primary fails          | Tool-specific failures           |
| **Re-plan**               | Ask LLM to create new plan after failure       | Logical failures, wrong approach |
| **Human-in-the-loop**     | Escalate to human after N failures             | Critical tasks, low confidence   |
| **Circuit breaker**       | Stop calling a tool after N consecutive failures| Prevent cascading failures      |

### Cost Implications

| Pattern              | Avg LLM Calls/Task | Avg Tokens/Task | Cost per Task (GPT-4o) |
|----------------------|---------------------|-----------------|------------------------|
| Function Calling     | 1-2                 | 500-1,500       | $0.005-0.02            |
| ReAct (3 steps)      | 3-5                 | 2,000-5,000     | $0.02-0.06             |
| Plan-and-Execute     | 4-8                 | 3,000-8,000     | $0.03-0.10             |
| Multi-Agent (3 agents)| 6-15               | 5,000-15,000    | $0.06-0.20             |

---

## 8. Serving Infrastructure

### Feature Comparison

| Feature               | vLLM          | TGI (HuggingFace) | Triton        | SageMaker     | Vertex AI     | Together AI   | Replicate     |
|-----------------------|---------------|--------------------|---------------|---------------|---------------|---------------|---------------|
| **Type**              | Self-hosted   | Self-hosted        | Self-hosted   | Managed       | Managed       | Managed API   | Managed API   |
| **Continuous Batching**| Yes          | Yes                | Yes           | Yes           | Yes           | Yes           | Yes           |
| **PagedAttention**    | Yes           | Yes                | No            | Depends       | Depends       | Yes           | Yes           |
| **Quantization**      | AWQ, GPTQ, FP8| AWQ, GPTQ, BnB   | All           | All           | All           | Yes           | Yes           |
| **Streaming**         | Yes           | Yes                | Yes           | Yes           | Yes           | Yes           | Yes           |
| **Multi-GPU**         | Tensor parallel| Tensor parallel   | Model parallel| Yes           | Yes           | Yes           | Yes           |
| **LoRA Serving**      | Yes           | Yes                | No            | Yes           | No            | No            | No            |
| **OpenAI-compatible** | Yes           | Yes (Messages API) | No            | No            | No            | Yes           | No            |
| **Community**         | Very Active   | Active             | NVIDIA-backed | AWS support   | Google support| Growing       | Growing       |

### Cost Comparison: Serving Llama 3.1 70B (estimated monthly, moderate traffic)

| Option                           | Hardware                 | Monthly Cost   | Throughput        |
|----------------------------------|--------------------------|----------------|-------------------|
| **vLLM on AWS** (self-managed)   | 1x A100 80GB or 2x A10G | $2,500-4,000   | ~50 req/s         |
| **TGI on AWS**                   | 1x A100 80GB             | $2,500-3,500   | ~40 req/s         |
| **SageMaker Endpoint**           | ml.g5.12xlarge           | $4,000-6,000   | ~30 req/s         |
| **Together AI API**              | Pay-per-token            | $500-2,000*    | Unlimited         |
| **Replicate**                    | Pay-per-second           | $1,000-3,000*  | Unlimited         |
| **Vertex AI (Llama on Model Garden)** | Pay-per-token       | $1,000-3,000*  | Unlimited         |

*API costs depend heavily on traffic volume. Lower volume favors API; higher volume favors self-hosted.*

### Self-Hosted vs Managed Decision Guide

```
START: "How should I serve my LLM?"
  |
  +-- Using a proprietary model (GPT-4o, Claude)?
  |     YES --> Use provider API (no self-hosting option)
  |     NO  --> Continue (open-source model)
  |
  +-- Traffic < 10K requests/day?
  |     YES --> Managed API (Together AI, Replicate) -- cheaper, no DevOps
  |     NO  --> Continue
  |
  +-- Do you have GPU infrastructure or ML engineers?
  |     YES --> Self-host with vLLM (best performance) or TGI
  |     NO  --> Managed service (SageMaker, Vertex AI)
  |
  +-- Need sub-100ms TTFT?
  |     YES --> Self-host with vLLM + quantization + dedicated GPUs
  |     NO  --> Any option works
  |
  +-- Need to serve multiple LoRA adapters?
        YES --> vLLM (native LoRA support, hot-swap adapters)
        NO  --> Any option works
```

---

## 9. Evaluation Frameworks

### Framework Comparison

| Framework        | Type              | Key Metrics                              | LLM-as-Judge | Custom Metrics | Tracing | Pricing         |
|------------------|-------------------|------------------------------------------|--------------|----------------|---------|-----------------|
| **RAGAS**        | RAG-specific      | Faithfulness, answer relevance, context precision/recall | Yes   | Yes            | No      | Open source     |
| **DeepEval**     | General LLM eval  | 14+ metrics incl. hallucination, bias, toxicity | Yes        | Yes            | No      | Open source + cloud |
| **LangSmith**    | Tracing + eval    | Custom, comparative evals, regression testing | Yes        | Yes            | Yes     | Free tier + paid |
| **W&B Weave**    | Experiment tracking + eval | Custom metrics, model comparison   | Yes          | Yes            | Yes     | Free tier + paid |
| **Arize Phoenix**| Observability + eval | Retrieval metrics, drift detection     | Yes          | Yes            | Yes     | Open source     |
| **Custom**       | Anything          | Whatever you define                       | Optional     | Yes            | Custom  | Engineering time |

### Key Metrics Explained

| Metric                  | What It Measures                                        | Range  | Target  |
|-------------------------|---------------------------------------------------------|--------|---------|
| **Faithfulness**        | Is the answer supported by the retrieved context?       | 0-1    | > 0.85  |
| **Answer Relevance**    | Does the answer address the question?                   | 0-1    | > 0.80  |
| **Context Precision**   | Are the retrieved chunks relevant to the question?      | 0-1    | > 0.75  |
| **Context Recall**      | Does the retrieved context cover the ground truth?      | 0-1    | > 0.80  |
| **Hallucination Rate**  | Fraction of claims not supported by context             | 0-1    | < 0.10  |
| **Latency (P95)**       | 95th percentile end-to-end response time                | ms     | < 2000  |
| **Token Efficiency**    | Useful output tokens / total tokens consumed            | 0-1    | > 0.50  |

### Setting Up an Automated Eval Pipeline

```
                    +------------------+
                    |  Test Dataset    |
                    |  (Q, A, Context) |
                    +--------+---------+
                             |
                    +--------v---------+
                    |  RAG Pipeline     |
                    |  (your system)    |
                    +--------+---------+
                             |
              +--------------+--------------+
              |              |              |
     +--------v---+  +-------v----+  +------v------+
     | Retrieval   |  | Generation |  | End-to-End  |
     | Metrics     |  | Metrics    |  | Metrics     |
     | (NDCG, MRR) |  | (RAGAS)    |  | (latency,   |
     |             |  |            |  |  cost)       |
     +--------+----+  +------+-----+  +------+------+
              |              |               |
              +--------------+---------------+
                             |
                    +--------v---------+
                    |  Dashboard /      |
                    |  CI/CD Gate       |
                    +------------------+
```

**Minimum viable eval pipeline:**
1. Create a golden test set of 50-100 question-answer-context triples.
2. Run your RAG pipeline on all questions.
3. Compute faithfulness and answer relevance using RAGAS.
4. Compute retrieval recall@5 by checking if ground-truth context appears in retrieved chunks.
5. Set pass/fail thresholds (e.g., faithfulness > 0.85, recall@5 > 0.80).
6. Run on every PR or weekly as regression test.

### Human Eval vs Automated Eval

| Dimension        | Human Evaluation                    | Automated (LLM-as-Judge)            |
|------------------|-------------------------------------|--------------------------------------|
| **Cost**         | $0.50-5.00 per evaluation           | $0.001-0.01 per evaluation           |
| **Speed**        | Minutes to hours                    | Seconds                              |
| **Consistency**  | Variable (inter-rater agreement)    | High (deterministic with temp=0)     |
| **Nuance**       | Captures subtle quality issues      | Misses nuance, especially humor/tone |
| **Scalability**  | 100s of evaluations                 | 100,000s of evaluations              |
| **Best for**     | Final quality gates, edge cases     | Regression testing, CI/CD            |

**Recommendation:** Use automated eval (RAGAS + LLM-as-judge) for continuous regression testing. Use human eval for quarterly quality audits and for validating that automated metrics correlate with real quality.

---

## 10. Cost Optimization Strategies

### 1. Semantic Caching (Save 30-60% on LLM Costs)

Cache semantically similar queries to avoid redundant LLM calls.

```
User Query --> Embed Query --> Search Cache (cosine sim > 0.95?)
  |                                |
  YES: Return cached response      NO: Call LLM, cache response
```

**Implementation:**
```python
import hashlib
from redis import Redis

SIMILARITY_THRESHOLD = 0.95

def get_or_generate(query: str, embedding_model, llm, cache: Redis):
    query_embedding = embedding_model.encode(query)
    # Search cache for similar query
    cached = search_cache(query_embedding, threshold=SIMILARITY_THRESHOLD)
    if cached:
        return cached["response"]  # Cache hit -- free!
    # Cache miss -- call LLM
    response = llm.generate(query)
    store_in_cache(query, query_embedding, response)
    return response
```

**Cost savings example (1M queries/month, 40% cache hit rate):**
- Without caching: 1M x $0.005 = $5,000/mo
- With caching: 600K x $0.005 + cache infra $50 = $3,050/mo (39% savings)

### 2. Model Routing by Query Complexity

(See Section 4 for details.)

| Traffic Split            | Cost Without Routing | Cost With Routing | Savings |
|--------------------------|----------------------|-------------------|---------|
| 70% simple, 20% med, 10% complex | $2,500         | $700              | 72%     |
| 50% simple, 30% med, 20% complex | $2,500         | $1,100            | 56%     |

### 3. Token Budgeting and Prompt Optimization

| Technique                         | Token Reduction | Effort |
|-----------------------------------|-----------------|--------|
| Remove verbose system prompts     | 20-40%          | Low    |
| Use structured output (JSON)      | 10-20%          | Low    |
| Compress retrieved context        | 30-50%          | Medium |
| Use abbreviations in system prompt| 5-10%           | Low    |
| Dynamic few-shot (fewer examples) | 20-40%          | Medium |
| Prompt caching (Anthropic)        | 90% on cached portion | Low |

**Prompt caching (Anthropic):** Cache the system prompt and static instructions. Pay full price once, then 90% discount on subsequent uses of the cached prefix. Extremely effective for high-volume applications with consistent system prompts.

### 4. Batch Processing vs Real-Time

| Dimension       | Real-Time              | Batch Processing         |
|-----------------|------------------------|--------------------------|
| **Latency**     | 200ms-2s               | Minutes to hours         |
| **Cost**        | Full per-token pricing  | 50% discount (OpenAI Batch API) |
| **Use cases**   | Chatbots, search       | Document processing, analysis |
| **Throughput**  | Rate-limited           | Higher limits            |

**OpenAI Batch API:** 50% cost reduction for requests that can tolerate 24-hour turnaround. Ideal for nightly document processing, bulk embeddings, or report generation.

### 5. Embedding Caching

Cache embeddings for frequently queried or static content:
- Store document embeddings at index time (always do this).
- Cache query embeddings for repeated queries (save re-embedding cost).
- Invalidate only when source documents change.

**Savings:** Embedding costs are typically 5-10% of total LLM costs, but caching prevents re-computation during re-indexing and development.

### 6. Quantized Models

| Quantization | Model Size Reduction | Quality Loss | Speedup |
|-------------|----------------------|--------------|---------|
| FP16        | 2x vs FP32           | Negligible   | 1.5-2x  |
| INT8 (W8A8) | 4x vs FP32           | <1%          | 2-3x    |
| INT4 (W4A16)| 8x vs FP32           | 1-3%         | 2-4x    |
| GPTQ/AWQ    | 4-8x vs FP32         | 1-2%         | 2-4x    |

**Example:** Llama 3.1 70B at FP16 requires 2x A100 80GB (~$5,000/mo). At INT4 (AWQ), it fits on 1x A100 80GB (~$2,500/mo), with ~1-2% quality loss.

### Comprehensive Cost Calculation Example

**Scenario:** RAG-based customer support bot, 500K queries/month.

| Component               | Calculation                                        | Monthly Cost |
|--------------------------|----------------------------------------------------|--------------|
| **Embedding queries**    | 500K x 100 tokens x $0.02/1M tokens               | $1.00        |
| **Embedding documents**  | 100K docs x 500 tokens x $0.02/1M (one-time/month)| $1.00        |
| **Vector DB (Pinecone)** | 100K vectors, Standard tier                        | $70.00       |
| **Re-ranker (Cohere)**   | 500K queries x $1/1K                               | $500.00      |
| **LLM (GPT-4o-mini)**   | 500K x 800 avg tokens x $0.375/1M blended          | $150.00      |
| **Semantic cache (Redis)**| Managed Redis, small instance                     | $30.00       |
| **Total before optimization** |                                               | **$752.00**  |
| **With 40% cache hit rate** | Remove 40% of re-ranker + LLM costs             | **$492.00**  |
| **With model routing (70/30)**| Route 70% to mini, 30% to 4o                  | **$430.00**  |

---

## 11. The AI Engineering Decision Matrix

A single reference for "Given requirement X, choose Y."

### Latency Requirements

| Requirement               | Recommendation                                                      |
|---------------------------|---------------------------------------------------------------------|
| < 50ms total              | Pre-computed results, no LLM in the hot path, use cached embeddings |
| < 200ms total             | GPT-4o-mini or Haiku, naive vector search, no re-ranking            |
| < 500ms total             | Any model, simple retrieval, optional re-ranking                    |
| < 2s total                | Any model, full RAG pipeline with re-ranking                        |
| > 2s acceptable           | Complex agent pipelines, plan-and-execute, multi-agent              |

### Data Privacy Requirements

| Requirement                    | Recommendation                                                   |
|--------------------------------|------------------------------------------------------------------|
| PII in queries                 | On-premise model (Llama, Mistral) or provider with DPA (Azure OpenAI) |
| HIPAA compliance               | Azure OpenAI with BAA, or self-hosted with proper infra          |
| SOC 2 required                 | Azure OpenAI, AWS Bedrock, or Google Vertex AI                   |
| Air-gapped environment         | Self-hosted open-source (Llama 3 + vLLM)                        |
| EU data residency (GDPR)       | EU-region deployment, Mistral (French company), or self-hosted   |

### Budget Constraints

| Budget            | Recommendation                                                        |
|-------------------|-----------------------------------------------------------------------|
| **< $50/month**   | GPT-4o-mini or Gemini Flash, ChromaDB (local), no re-ranking         |
| **< $100/month**  | GPT-4o-mini, Pinecone free tier or pgvector, basic RAG               |
| **< $500/month**  | Model routing (mini + 4o), managed vector DB, semantic caching       |
| **< $2,000/month**| Full RAG pipeline with re-ranking, evaluation, monitoring            |
| **< $10,000/month**| Self-hosted open-source models, dedicated GPUs, full observability  |
| **> $10,000/month**| Multi-model ensemble, fine-tuned models, dedicated infrastructure   |

### Accuracy Requirements

| Requirement                  | Recommendation                                                     |
|------------------------------|--------------------------------------------------------------------|
| Best-effort (chatbot, search)| Standard RAG with GPT-4o-mini, good chunking                      |
| High accuracy (enterprise Q&A)| RAG + re-ranking + GPT-4o/Claude Sonnet, eval pipeline           |
| Very high accuracy (medical, legal)| RAG + re-ranking + GPT-4o + human-in-the-loop verification  |
| Near-perfect (safety-critical)| Multiple model consensus + human review + audit trail             |

### Scale Requirements

| Scale                   | Recommendation                                                       |
|-------------------------|----------------------------------------------------------------------|
| < 1K queries/day        | Single API, ChromaDB/pgvector, minimal infra                        |
| 1K-100K queries/day     | Managed vector DB, semantic caching, model routing                   |
| 100K-1M queries/day     | Self-hosted models, Milvus/Pinecone, batch processing where possible|
| > 1M queries/day        | Multi-region deployment, custom serving infra, aggressive caching    |

### Team Size and Expertise

| Team Profile                    | Recommendation                                                   |
|---------------------------------|------------------------------------------------------------------|
| Solo developer                  | API-based (OpenAI/Anthropic), Pinecone, LangChain, minimal infra|
| Small team (2-5), no ML eng.   | Managed services, API models, LangSmith for eval                 |
| Small team with ML engineer     | Open-source models, vLLM, custom eval pipeline                  |
| Large team (10+)                | Multi-model strategy, custom infra, fine-tuning, A/B testing    |

### Multi-lingual Requirements

| Requirement                   | Recommendation                                                    |
|-------------------------------|------------------------------------------------------------------|
| English only                  | Any model/embedding works                                        |
| 2-5 European languages        | Cohere multilingual embed, GPT-4o/Claude (strong multilingual)  |
| CJK languages                 | BGE-M3 embeddings, GPT-4o (strong CJK), budget +30-50% for tokens|
| 20+ languages                 | Cohere multilingual, Gemini (broadest language support)          |
| Low-resource languages        | GPT-4o or Gemini (largest training data), expect quality drop    |

### Deployment Environment

| Requirement                  | Recommendation                                                     |
|------------------------------|--------------------------------------------------------------------|
| Cloud (AWS)                  | Bedrock (managed), SageMaker (custom), or API providers            |
| Cloud (GCP)                  | Vertex AI (managed), GKE + vLLM (custom)                          |
| Cloud (Azure)                | Azure OpenAI (managed, best compliance), AKS + vLLM (custom)      |
| On-premise                   | vLLM + Llama 3 on NVIDIA GPUs, Qdrant/Milvus for vectors          |
| Edge / mobile                | Small models (Phi-3, Gemma 2B), ONNX runtime, all-MiniLM embeddings|
| Hybrid (cloud + on-prem)     | On-prem for sensitive data, cloud API for general queries          |

### Quick Decision Cheat Sheet

```
Need fast prototype?          --> OpenAI API + ChromaDB + LangChain
Need production RAG?          --> GPT-4o-mini + Pinecone + Cohere Rerank + RAGAS eval
Need lowest cost?             --> Gemini Flash + pgvector + semantic cache
Need highest quality?         --> GPT-4o/Claude Sonnet + hybrid search + re-rank + eval pipeline
Need data privacy?            --> Llama 3 + vLLM + Qdrant (all self-hosted)
Need to handle 1M+ docs?     --> Milvus + semantic chunking + model routing
Need real-time agents?        --> Function calling (OpenAI/Anthropic) + LangGraph
Need multi-modal?             --> GPT-4o or Gemini 2.0 (vision + audio + text)
Need to minimize hallucination? --> RAG + re-ranking + faithfulness eval + citations
Need regulatory compliance?  --> Azure OpenAI (HIPAA/SOC2) or self-hosted
```

---

## Appendix: Interview Tips for Trade-off Questions

When asked a system design trade-off question in an interview:

1. **Never give a single answer.** Always present at least two options with trade-offs.
2. **Ask clarifying questions first:** What is the latency budget? What is the data sensitivity? What is the team size? What is the scale?
3. **Use this framework:**
   - "It depends on [key factor]. If [condition A], I would choose [option A] because [reason]. If [condition B], I would choose [option B] because [reason]."
4. **Quantify when possible:** "Re-ranking adds ~100ms latency but improves recall by 10-20%, which is worth it if our latency budget is 500ms+."
5. **Mention what you would measure:** "I would set up an A/B test measuring [metric] to validate this decision."
6. **Acknowledge unknowns:** "In practice, I would benchmark both options on our specific data before committing."
7. **Show evolution:** "I would start with [simple approach] and migrate to [complex approach] when [trigger condition]."
