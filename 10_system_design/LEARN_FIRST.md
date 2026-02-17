# System Design for AI Engineers — Complete Guide

> **Read this first.** This is your definitive reference for designing production AI systems in interviews and on the job. Every architecture includes diagrams, technology choices, trade-offs, and real numbers.

---

## Table of Contents

1. [Why System Design Matters for AI Engineers](#1-why-system-design-matters)
2. [The AI System Design Framework](#2-the-ai-system-design-framework)
3. [Designing a Production RAG System](#3-designing-a-production-rag-system)
4. [Designing an LLM Application Platform](#4-designing-an-llm-application-platform)
5. [Designing ML Model Serving Infrastructure](#5-designing-ml-model-serving-infrastructure)
6. [Designing an MLOps Pipeline](#6-designing-an-mlops-pipeline)
7. [Production Considerations for AI Systems](#7-production-considerations)
8. [Common AI System Design Trade-offs](#8-common-trade-offs)
9. [AI System Design Interview Questions with Answers](#9-interview-questions-with-answers)

---

## 1. Why System Design Matters

AI engineering is no longer just about training models. In production:

- **80% of the work** is infrastructure, data pipelines, serving, and monitoring
- **Models are the easy part** — the hard part is reliability, cost, latency, and scale
- **Interviewers test** whether you can build systems that work at scale, not just notebooks

### What Interviewers Evaluate

| Dimension | What They Look For |
|-----------|-------------------|
| **Structured Thinking** | Do you clarify requirements before designing? |
| **Architecture** | Can you decompose a system into well-defined components? |
| **Trade-offs** | Do you explain WHY you chose X over Y? |
| **AI-Specific Knowledge** | Embeddings, retrieval, model serving, evaluation |
| **Production Readiness** | Monitoring, cost, failure handling, scaling |
| **Communication** | Can you explain complex systems clearly? |

---

## 2. The AI System Design Framework

Use this 5-phase framework for any 45-60 minute system design interview:

```
Phase 1: Clarify (5 min)     → Ask questions, define scope
Phase 2: High-Level (10 min) → Draw the architecture, name components
Phase 3: Deep Dive (20 min)  → Detail 2-3 key components
Phase 4: Operations (10 min) → Monitoring, cost, scaling, failure
Phase 5: Trade-offs (5 min)  → What would you change? Risks?
```

### Phase 1: Clarifying Questions to Always Ask

**For RAG Systems:**
- How many documents? What formats? How often updated?
- Who are the users? How many concurrent users?
- Accuracy vs latency — which matters more?
- Do we need source citations? Multi-turn conversations?
- Data sensitivity — PII, compliance, access controls?

**For LLM Applications:**
- What models are we using? Budget for API costs?
- Real-time vs batch? Latency requirements?
- Do we need to support multiple models? Open-source?
- What are the safety/guardrail requirements?
- How do we measure success? What metrics?

**For ML Serving:**
- QPS (queries per second)? P99 latency target?
- Model size? GPU requirements?
- How often do models update?
- A/B testing requirements?
- Batch vs real-time or both?

### Phase 2: High-Level Architecture Template

Always start with this skeleton and adapt:

```
┌──────────┐    ┌──────────┐    ┌──────────────┐    ┌──────────┐
│  Client  │───▶│ API GW / │───▶│  Application │───▶│  Data    │
│  (Web/   │    │ Load     │    │  Layer       │    │  Layer   │
│  Mobile) │    │ Balancer │    │              │    │          │
└──────────┘    └──────────┘    └──────────────┘    └──────────┘
                                       │
                                       ▼
                                ┌──────────────┐
                                │  AI/ML       │
                                │  Services    │
                                └──────────────┘
                                       │
                                       ▼
                                ┌──────────────┐
                                │  Monitoring  │
                                │  & Eval      │
                                └──────────────┘
```

---

## 3. Designing a Production RAG System

### Full Architecture

```
                        ┌─────────────────────────────────────────┐
                        │           INGESTION PIPELINE            │
                        │                                         │
  ┌──────────┐         │  ┌─────────┐  ┌──────────┐  ┌────────┐ │
  │Documents │─────────▶│  │ Parser  │─▶│ Chunker  │─▶│Embedder│ │
  │(PDF,HTML,│         │  │(Unstr.) │  │          │  │        │ │
  │ Slack,..)│         │  └─────────┘  └──────────┘  └────┬───┘ │
  └──────────┘         │                                   │     │
                        │                                   ▼     │
                        │                            ┌──────────┐ │
                        │                            │ Vector   │ │
                        │                            │ DB       │ │
                        │                            └──────────┘ │
                        └─────────────────────────────────────────┘

                        ┌─────────────────────────────────────────┐
                        │            QUERY PIPELINE               │
                        │                                         │
  ┌──────────┐         │  ┌─────────┐  ┌──────────┐  ┌────────┐ │
  │  User    │─────────▶│  │ Query   │─▶│Retriever │─▶│Reranker│ │
  │  Query   │         │  │Processor│  │          │  │        │ │
  └──────────┘         │  └─────────┘  └──────────┘  └────┬───┘ │
                        │                                   │     │
                        │       ┌──────────┐                │     │
                        │       │ Semantic  │◀───────────────┘     │
  ┌──────────┐         │       │ Cache     │                      │
  │ Response │◀────────│       └─────┬─────┘                      │
  │ + Sources│         │             ▼                             │
  └──────────┘         │       ┌──────────┐                       │
                        │       │   LLM    │                       │
                        │       │Generator │                       │
                        │       └──────────┘                       │
                        └─────────────────────────────────────────┘

                        ┌─────────────────────────────────────────┐
                        │         EVALUATION PIPELINE             │
                        │                                         │
                        │  ┌─────────┐  ┌──────────┐  ┌────────┐ │
                        │  │Retrieval│  │Generation│  │  Eval  │ │
                        │  │ Metrics │  │ Metrics  │  │Dashboard│ │
                        │  └─────────┘  └──────────┘  └────────┘ │
                        └─────────────────────────────────────────┘
```

### 3.1 Document Processing Pipeline

**Step 1: Document Parsing**

| Format | Tool | Notes |
|--------|------|-------|
| PDF | `unstructured`, `PyMuPDF` | Handle scanned PDFs with OCR |
| HTML | `BeautifulSoup`, `trafilatura` | Strip boilerplate, keep structure |
| Markdown | `markdownify` | Preserve headers for metadata |
| DOCX | `python-docx`, `unstructured` | Extract tables separately |
| Slack/API | Custom connectors | Incremental sync with webhooks |

**Technology choice:** `unstructured.io` — handles 25+ file types, extracts tables, maintains document structure. Use their API ($0.01/page) or self-host.

**Step 2: Chunking**

```python
# Recursive Character Splitting (most common, good default)
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=512,        # tokens, not characters
    chunk_overlap=50,      # 10% overlap prevents losing context at boundaries
    separators=["\n\n", "\n", ". ", " "]  # split on paragraph > line > sentence
)
```

| Strategy | Chunk Size | Best For | Weakness |
|----------|-----------|----------|----------|
| Fixed-size (512 tokens) | 512 | General purpose | Splits mid-sentence |
| Recursive character | 256-1024 | Structured text | Requires tuning |
| Semantic (embedding-based) | Variable | Research docs | 3-5x slower, higher cost |
| Sentence-based | 3-5 sentences | Q&A systems | Too small for complex topics |
| Document-structure | Varies | Technical docs with headers | Complex implementation |

**Recommended defaults:**
- **General RAG**: Recursive character, 512 tokens, 50 token overlap
- **Legal/Medical**: Semantic chunking, respect section boundaries
- **Code**: AST-based chunking (split on functions/classes)
- **Q&A**: Sentence-based, 3-5 sentences per chunk

**Step 3: Metadata Extraction**

Always attach metadata to chunks — it dramatically improves retrieval:

```python
chunk_metadata = {
    "source": "confluence/engineering/deploy-guide",
    "title": "Deployment Guide v2.3",
    "section": "Rolling Updates",
    "author": "jane@company.com",
    "last_updated": "2026-01-15",
    "doc_type": "technical_guide",
    "access_level": "engineering",    # For ACL-aware RAG
    "chunk_index": 3,                 # Position in original document
    "total_chunks": 12
}
```

### 3.2 Embedding Pipeline

| Model | Dimensions | Cost/1M tokens | MTEB Score | Latency |
|-------|-----------|----------------|------------|---------|
| OpenAI text-embedding-3-small | 1536 | $0.02 | 62.3 | ~50ms |
| OpenAI text-embedding-3-large | 3072 | $0.13 | 64.6 | ~80ms |
| Cohere embed-v3 | 1024 | $0.10 | 64.5 | ~60ms |
| Voyage AI voyage-3 | 1024 | $0.06 | 67.1 | ~55ms |
| BGE-large-en-v1.5 (open) | 1024 | Self-host | 63.6 | ~30ms* |
| all-MiniLM-L6-v2 (open) | 384 | Self-host | 56.3 | ~10ms* |

*Self-hosted latency on GPU.

**Decision guide:**
- **Budget-sensitive**: `text-embedding-3-small` — best price/performance
- **Quality-first**: `voyage-3` or `text-embedding-3-large`
- **On-premise/privacy**: `BGE-large-en-v1.5` — best open-source
- **Low-latency**: `all-MiniLM-L6-v2` — fastest, good enough for many use cases

**Batch vs Real-time:**
- **Ingestion**: Always batch. Process 100-1000 chunks per API call
- **Queries**: Real-time. Single embedding per query (~50ms)
- **Tip**: Use async batching for ingestion to maximize throughput

### 3.3 Vector Database Selection

| Feature | Pinecone | Weaviate | Qdrant | Milvus | pgvector | ChromaDB |
|---------|----------|----------|--------|--------|----------|----------|
| **Hosting** | Managed only | Both | Both | Both | Self-host* | Self-host |
| **Max Vectors** | Billions | Billions | Billions | Billions | Millions | Millions |
| **Hybrid Search** | Yes | Yes | Yes | Yes | Limited | No |
| **Filtering** | Excellent | Excellent | Excellent | Good | SQL-native | Basic |
| **Latency (P99)** | <50ms | <100ms | <50ms | <100ms | <200ms | <100ms |
| **Cost (1M vectors)** | ~$70/mo | ~$25/mo | ~$25/mo | Free* | Free* | Free |
| **Production Ready** | Yes | Yes | Yes | Yes | Yes | No** |

*Self-hosted costs depend on infrastructure. **ChromaDB is for prototyping, not production.

**Decision guide:**
- **Startup/MVP**: ChromaDB (prototype) → Pinecone or Qdrant (production)
- **Enterprise (managed)**: Pinecone — zero ops, excellent filtering
- **Enterprise (self-hosted)**: Qdrant or Weaviate — best self-hosted options
- **Already using Postgres**: pgvector — avoid new infrastructure for <1M vectors
- **Need hybrid search**: Weaviate or Pinecone — native BM25 + vector

### 3.4 Retrieval Strategies

**Level 1: Naive Vector Search (Baseline)**
```
Query → Embed → Top-K nearest neighbors → LLM
```
- Latency: ~100ms | Quality: Baseline | Cost: Lowest
- When to use: Simple use cases, internal tools, prototyping

**Level 2: Hybrid Search (Vector + BM25)**
```
Query → [Vector Search] + [BM25 Keyword Search] → Reciprocal Rank Fusion → LLM
```
- Latency: ~150ms | Quality: 15-25% better than naive | Cost: Low
- When to use: When exact keyword matches matter (product names, codes, IDs)

**Level 3: Hybrid + Re-ranking**
```
Query → [Hybrid Search (top-50)] → Re-ranker (top-5) → LLM
```
- Latency: ~300ms | Quality: 20-40% better than naive | Cost: Medium
- When to use: When accuracy matters more than latency
- Re-ranker options: Cohere Rerank ($1/1K queries), cross-encoder (self-hosted)

**Level 4: Advanced (Query Expansion + Re-ranking)**
```
Query → Query Decomposition → [Multiple Searches] → Merge → Re-rank → LLM
```
- Latency: ~500ms-1s | Quality: Best | Cost: Highest
- When to use: Complex questions, research use cases

**Level 5: Agentic RAG**
```
Query → Agent decides strategy → [Search / SQL / API / Calculator] → Synthesize → LLM
```
- Latency: 2-10s | Quality: Best for complex queries | Cost: Highest
- When to use: Multi-hop questions, queries needing multiple data sources

### 3.5 Generation Layer

**Prompt Template (Production RAG):**

```
You are a helpful assistant that answers questions based on the provided context.

RULES:
1. Only use information from the provided context
2. If the context doesn't contain the answer, say "I don't have enough information"
3. Always cite your sources using [Source: document_name]
4. Be concise and direct

CONTEXT:
{retrieved_chunks_with_metadata}

CONVERSATION HISTORY:
{last_3_turns}

USER QUESTION: {query}

ANSWER:
```

**Context Window Management:**
- Reserve 30% for system prompt + instructions
- Reserve 20% for conversation history
- Use remaining 50% for retrieved context
- Example: GPT-4o (128K) → ~64K tokens for context → ~40-60 chunks of 512 tokens

**Streaming:**
- Always stream responses for UX (first token in <500ms)
- Use Server-Sent Events (SSE) for web clients
- Buffer citations until full response is generated

### 3.6 Evaluation Pipeline

| Metric | What It Measures | Target | Tool |
|--------|-----------------|--------|------|
| **Context Precision** | Are retrieved docs relevant? | >0.8 | RAGAS |
| **Context Recall** | Did we find all relevant docs? | >0.7 | RAGAS |
| **Faithfulness** | Is the answer grounded in context? | >0.9 | RAGAS, DeepEval |
| **Answer Relevance** | Does the answer address the question? | >0.8 | RAGAS |
| **MRR (Mean Reciprocal Rank)** | Is the best doc ranked first? | >0.7 | Custom |
| **NDCG@5** | Overall ranking quality | >0.6 | Custom |
| **Hallucination Rate** | % of unsupported claims | <5% | DeepEval |
| **Latency (P99)** | End-to-end response time | <3s | Custom |

**Automated Eval Pipeline:**
```
┌──────────┐    ┌──────────────┐    ┌──────────┐    ┌──────────┐
│ Test     │───▶│ RAG Pipeline │───▶│ Eval     │───▶│Dashboard │
│ Dataset  │    │ (Query +     │    │ Metrics  │    │(Grafana/ │
│ (Q&A     │    │  Retrieve +  │    │(RAGAS/   │    │ W&B)     │
│  pairs)  │    │  Generate)   │    │ DeepEval)│    │          │
└──────────┘    └──────────────┘    └──────────┘    └──────────┘
```

### 3.7 Caching Layer

**Exact Cache:** Hash the query → cache the response. Hit rate: 10-20%.

**Semantic Cache:** Embed the query → find similar cached queries. Hit rate: 30-60%.

```
Query → Embed → Search Cache (cosine sim > 0.95) → Hit? Return cached response
                                                    → Miss? Run full pipeline, cache result
```

- **GPTCache**: Open-source semantic caching library
- **Savings**: At 40% hit rate with GPT-4o, saves ~$400/month per 100K queries

### 3.8 Complete RAG System — Technology Stack

```
Ingestion:    Unstructured.io → LangChain Splitter → OpenAI Embeddings → Qdrant
Query:        FastAPI → Hybrid Search (Qdrant) → Cohere Rerank → GPT-4o (streaming)
Evaluation:   RAGAS + DeepEval → Weights & Biases
Cache:        Redis (exact) + GPTCache (semantic)
Monitoring:   LangSmith (LLM traces) + Prometheus + Grafana
Infra:        Kubernetes + Docker + Terraform
CI/CD:        GitHub Actions → eval gate → deploy
```

---

## 4. Designing an LLM Application Platform

### Full Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                        API GATEWAY                                 │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  │
│  │ Rate       │  │ Auth       │  │ Input      │  │ Request    │  │
│  │ Limiter    │  │ (JWT/API   │  │ Validation │  │ Router     │  │
│  │            │  │  Key)      │  │            │  │            │  │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘  │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│                     APPLICATION LAYER                              │
│                                                                    │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  │
│  │ Prompt     │  │ Memory     │  │ Agent      │  │ Tool       │  │
│  │ Manager    │  │ Manager    │  │ Orchestrator│  │ Registry   │  │
│  │ (versions, │  │ (history,  │  │ (ReAct,    │  │ (search,   │  │
│  │  templates)│  │  summary)  │  │  plan+exec)│  │  code, DB) │  │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘  │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│                     MODEL ROUTER                                   │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Complexity Classifier → Route to appropriate model          │  │
│  │                                                              │  │
│  │  Simple (FAQ, greetings)  → GPT-4o-mini / Haiku ($0.25/1M)  │  │
│  │  Medium (analysis, Q&A)   → GPT-4o / Sonnet ($3/1M)         │  │
│  │  Complex (reasoning)      → Claude Opus / o1 ($15/1M)       │  │
│  │  Private/On-prem           → Llama 3.1 70B (self-hosted)     │  │
│  └──────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│                     GUARDRAILS LAYER                               │
│                                                                    │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  │
│  │ PII        │  │ Prompt     │  │ Output     │  │ Content    │  │
│  │ Detection  │  │ Injection  │  │ Filtering  │  │ Policy     │  │
│  │ (Presidio) │  │ Defense    │  │ (toxicity) │  │ Check      │  │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘  │
└────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────┐
│                   OBSERVABILITY                                    │
│                                                                    │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐  │
│  │ LLM        │  │ Cost       │  │ Quality    │  │ Usage      │  │
│  │ Traces     │  │ Tracking   │  │ Metrics    │  │ Analytics  │  │
│  │(LangSmith) │  │ (per-call) │  │(eval scores│  │(per tenant)│  │
│  └────────────┘  └────────────┘  └────────────┘  └────────────┘  │
└────────────────────────────────────────────────────────────────────┘
```

### 4.1 Multi-Model Routing

**Why route?** GPT-4o costs 60x more than GPT-4o-mini. Most queries don't need the expensive model.

**Routing strategies:**

1. **Keyword-based** (simplest): If query contains "summarize" → cheap model
2. **Classifier-based**: Train a small classifier on query complexity → route
3. **Cascade**: Try cheap model first → if confidence low, escalate to expensive model
4. **Semantic**: Embed query → cluster → route by cluster

**Cost savings example:**
- 100K queries/day
- Without routing: 100K × GPT-4o ($3/1M input) = ~$300/day
- With routing (70% simple, 25% medium, 5% complex):
  - 70K × mini ($0.15/1M) = $10.50
  - 25K × GPT-4o ($3/1M) = $75
  - 5K × Opus ($15/1M) = $75
  - **Total: $160.50/day (46% savings)**

### 4.2 Agent Architectures

**ReAct (Reasoning + Acting):**
```
Think → Act → Observe → Think → Act → Observe → ... → Answer
```
- Best for: Single-step tool use, Q&A with search
- Tools: LangChain ReAct agent, Claude tool use

**Plan-and-Execute:**
```
Plan (break into steps) → Execute Step 1 → Execute Step 2 → ... → Synthesize
```
- Best for: Multi-step tasks, research, analysis
- Tools: LangGraph, CrewAI

**Multi-Agent:**
```
Orchestrator → [Researcher Agent] + [Analyst Agent] + [Writer Agent] → Combine
```
- Best for: Complex workflows, different expertise needed
- Tools: AutoGen, CrewAI, LangGraph

### 4.3 Memory Management

| Strategy | Max History | Latency | Cost | Best For |
|----------|------------|---------|------|----------|
| Full history | All turns | High | High | Short conversations |
| Sliding window (last N) | Last 10 | Low | Low | Chat interfaces |
| Summarization | Unlimited* | Medium | Medium | Long conversations |
| Vector memory | Unlimited* | Medium | Medium | Knowledge-heavy chats |

**Production approach**: Sliding window (last 5 turns) + summarized older history + vector memory for key facts.

### 4.4 Guardrails and Safety

```
INPUT GUARDRAILS:
  User Input → PII Detection (Presidio) → Prompt Injection Check → Topic Filter
                    ↓                           ↓                       ↓
               Mask/Reject              Reject/Rephrase          Block/Redirect

OUTPUT GUARDRAILS:
  LLM Output → Hallucination Check → Toxicity Filter → PII Scan → Response
                     ↓                      ↓               ↓
                Flag/Retry              Filter/Edit     Mask/Remove
```

**Key tools:**
- **PII Detection**: Microsoft Presidio (open-source), AWS Comprehend
- **Prompt Injection**: Rebuff, custom classifiers, input/output sandwich
- **Toxicity**: Perspective API (Google), custom classifiers
- **Guardrails framework**: Guardrails AI, NeMo Guardrails (NVIDIA)

### 4.5 Cost Optimization

| Strategy | Savings | Implementation Effort |
|----------|---------|----------------------|
| Semantic caching | 30-60% | Medium |
| Model routing | 40-60% | Medium |
| Prompt optimization (fewer tokens) | 10-30% | Low |
| Batch processing (non-real-time) | 20-40% | Low |
| Open-source for simple tasks | 50-80% | High |
| Token budgeting (max_tokens) | 5-15% | Low |

### 4.6 Streaming and Real-Time UX

```
Client (SSE) ← API Gateway ← LLM Provider (streaming)

Timeline:
  0ms      → Request sent
  200ms    → First token received (TTFT - Time to First Token)
  200-3000ms → Tokens stream in (~50 tokens/sec for GPT-4o)
  3000ms   → Response complete
  3100ms   → Citations/sources appended
```

**Implementation:**
```python
# FastAPI streaming endpoint
from fastapi.responses import StreamingResponse

@app.post("/chat")
async def chat(request: ChatRequest):
    async def generate():
        async for chunk in llm.astream(request.messages):
            yield f"data: {json.dumps({'content': chunk})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
```

---

## 5. Designing ML Model Serving Infrastructure

### Full Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL SERVING PLATFORM                        │
│                                                                  │
│  ┌──────────┐    ┌──────────────┐    ┌────────────────────────┐ │
│  │ Request  │    │ Model        │    │ Inference Servers       │ │
│  │ Queue    │───▶│ Router       │───▶│                        │ │
│  │ (SQS/   │    │ (A/B test,   │    │ ┌────────┐ ┌────────┐ │ │
│  │  Redis)  │    │  canary,     │    │ │Model A │ │Model B │ │ │
│  └──────────┘    │  shadow)     │    │ │(prod)  │ │(canary)│ │ │
│                  └──────────────┘    │ │ 90%    │ │ 10%   │ │ │
│                                      │ └────────┘ └────────┘ │ │
│                                      └────────────────────────┘ │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    SUPPORTING SERVICES                    │   │
│  │                                                           │   │
│  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────────────┐ │   │
│  │  │Feature │  │Model   │  │Auto    │  │ Monitoring     │ │   │
│  │  │Store   │  │Registry│  │Scaler  │  │ (latency, GPU, │ │   │
│  │  │(Feast) │  │(MLflow)│  │(KEDA)  │  │  drift, errors)│ │   │
│  │  └────────┘  └────────┘  └────────┘  └────────────────┘ │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 5.1 Batch vs Real-Time Inference

| Aspect | Real-Time | Batch |
|--------|-----------|-------|
| Latency | <100ms-2s | Minutes to hours |
| Throughput | 100s-1000s QPS | Millions per run |
| Cost | Higher (always-on) | Lower (spot instances) |
| Use Cases | Chatbots, search, recommendations | Reports, email campaigns, scoring |
| Infrastructure | K8s + GPU, autoscaling | Spark, Ray, batch jobs |
| Scaling | HPA on request queue depth | More workers |

**Hybrid approach**: Pre-compute batch predictions, serve from cache, fall back to real-time for cache misses.

### 5.2 Model Registry and Versioning

```
┌────────────────────────────────────────────────────────────┐
│                    MODEL REGISTRY (MLflow)                   │
│                                                              │
│  Model: document_classifier                                  │
│  ├── v1.0 (Production)  - accuracy: 0.94, latency: 45ms    │
│  ├── v1.1 (Staging)     - accuracy: 0.96, latency: 42ms    │
│  └── v1.2 (Development) - accuracy: 0.95, latency: 38ms    │
│                                                              │
│  Each version stores:                                        │
│  - Model artifacts (weights, config)                         │
│  - Training metrics                                          │
│  - Training data version (DVC hash)                          │
│  - Environment (requirements.txt, Docker image)              │
│  - Evaluation results on test set                            │
└────────────────────────────────────────────────────────────┘
```

### 5.3 A/B Testing and Canary for Models

```
Traffic Split:
  ┌────────────────────────────────┐
  │ Request Router                  │
  │                                │
  │ Model v1.0 (Control)  ── 80%  │──▶ Log predictions + outcomes
  │ Model v1.1 (Treatment) ── 15% │──▶ Log predictions + outcomes
  │ Model v1.2 (Shadow)    ── 5%  │──▶ Log predictions only (no serve)
  └────────────────────────────────┘
                  │
                  ▼
  ┌────────────────────────────────┐
  │ Statistical Analysis            │
  │                                │
  │ - Conversion rate comparison   │
  │ - Latency comparison           │
  │ - Error rate comparison        │
  │ - Statistical significance     │
  │ - Auto-promote if p < 0.05    │
  └────────────────────────────────┘
```

### 5.4 Model Serving Frameworks

| Framework | GPU Support | Batching | Quantization | Streaming | Best For |
|-----------|-----------|----------|-------------|-----------|----------|
| vLLM | Excellent | PagedAttention | GPTQ, AWQ | Yes | LLM serving |
| TGI (HuggingFace) | Excellent | Continuous | GPTQ, bitsandbytes | Yes | LLM serving |
| Triton (NVIDIA) | Excellent | Dynamic | TensorRT | Yes | Multi-model |
| TorchServe | Good | Yes | Yes | No | PyTorch models |
| BentoML | Good | Adaptive | Yes | Yes | Easy deployment |
| Ray Serve | Good | Yes | Yes | Yes | Scaling |

**For LLM serving**: vLLM — 2-4x higher throughput than naive HuggingFace via PagedAttention.

**For traditional ML**: BentoML or Ray Serve — easy to deploy scikit-learn, XGBoost, PyTorch models.

### 5.5 Feature Stores

```
┌─────────────────────────────────────────────────────┐
│                   FEATURE STORE (Feast)               │
│                                                       │
│  ┌─────────────────┐    ┌─────────────────────────┐  │
│  │ OFFLINE STORE    │    │ ONLINE STORE             │  │
│  │ (S3/BigQuery)    │    │ (Redis/DynamoDB)         │  │
│  │                  │    │                          │  │
│  │ - Training data  │    │ - Serving features       │  │
│  │ - Batch features │───▶│ - Low-latency lookup     │  │
│  │ - Historical     │    │ - Latest feature values  │  │
│  │ - TB-scale       │    │ - <10ms P99              │  │
│  └─────────────────┘    └─────────────────────────┘  │
│                                                       │
│  Feature definitions:                                 │
│  - user_purchase_count_7d (computed daily)            │
│  - user_avg_session_duration (computed hourly)        │
│  - item_view_count_24h (computed hourly)              │
│  - user_embedding_v2 (computed weekly)                │
└─────────────────────────────────────────────────────┘
```

---

## 6. Designing an MLOps Pipeline

### Full Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                      MLOPS PIPELINE                               │
│                                                                    │
│  DATA LAYER                                                        │
│  ┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐  │
│  │ Data    │─▶│ Data     │─▶│ Feature  │─▶│ Data Versioning  │  │
│  │ Sources │  │ Validation│  │ Engineer │  │ (DVC)            │  │
│  └─────────┘  │(Great    │  │(Feast/   │  └──────────────────┘  │
│               │Expectat.)│  │ custom)  │                         │
│               └──────────┘  └──────────┘                         │
│                                   │                               │
│  TRAINING LAYER                   ▼                               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ Experiment│─▶│ Training │─▶│ Model    │─▶│ Model Registry   │ │
│  │ Tracking │  │ Pipeline │  │ Evaluation│  │ (MLflow)         │ │
│  │ (W&B/   │  │ (Kubeflow│  │ (test set│  │                  │ │
│  │  MLflow) │  │  /Vertex)│  │  + eval) │  │ v1.0 → v1.1     │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │
│                                                    │              │
│  DEPLOYMENT LAYER                                  ▼              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ CI/CD    │─▶│ Model    │─▶│ Canary / │─▶│ Production       │ │
│  │ Pipeline │  │ Validation│  │ A/B Test │  │ Serving          │ │
│  │(GitHub   │  │ Gate     │  │          │  │                  │ │
│  │ Actions) │  │          │  │          │  │                  │ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │
│                                                    │              │
│  MONITORING LAYER                                  ▼              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────────┐ │
│  │ Data     │  │ Model    │  │ Infra    │  │ Alerting         │ │
│  │ Drift    │  │ Perf     │  │ Metrics  │  │ (PagerDuty)      │ │
│  │ Monitor  │  │ Monitor  │  │ (GPU,    │  │                  │ │
│  │(Evidently│  │          │  │  latency)│  │ Drift > threshold│ │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

### 6.1 Training Pipeline

**Data Versioning (DVC):**
```
project/
├── data/
│   ├── raw/          ← tracked by DVC (not Git)
│   ├── processed/    ← tracked by DVC
│   └── data.dvc      ← tracked by Git (pointer file)
├── models/
│   └── model.pkl.dvc ← tracked by Git
├── dvc.yaml          ← pipeline definition
└── params.yaml       ← hyperparameters
```

**Experiment Tracking (Weights & Biases):**
```python
import wandb

wandb.init(project="document-classifier", config={
    "model": "bert-base",
    "learning_rate": 2e-5,
    "epochs": 10,
    "batch_size": 32
})

# During training
wandb.log({"train_loss": loss, "val_accuracy": acc, "epoch": epoch})

# After training
wandb.log({"test_accuracy": 0.96, "test_f1": 0.94})
wandb.save("model.pt")
```

### 6.2 CI/CD for ML

```
┌─────────────────────────────────────────────────────────────┐
│                  ML CI/CD PIPELINE                           │
│                                                              │
│  PR Opened                                                   │
│    ├── Run unit tests                                        │
│    ├── Run data validation (Great Expectations)              │
│    ├── Run model training on sample data                     │
│    └── Run model evaluation on test set                      │
│                                                              │
│  PR Merged to main                                           │
│    ├── Full training on production data                      │
│    ├── Model evaluation + comparison to current production   │
│    ├── PERFORMANCE GATE:                                     │
│    │   ├── accuracy >= current_prod - 0.01? ✅               │
│    │   ├── latency <= current_prod * 1.1?   ✅               │
│    │   └── no data quality issues?           ✅               │
│    ├── Register model in MLflow (Staging)                    │
│    ├── Deploy canary (10% traffic)                           │
│    ├── Monitor for 2 hours                                   │
│    └── Promote to Production (100% traffic) or Rollback     │
└─────────────────────────────────────────────────────────────┘
```

### 6.3 Monitoring for ML

**Data Drift Detection:**
```
┌──────────────────────────────────────────────────┐
│ Drift Monitor (Evidently AI)                      │
│                                                    │
│ Feature: user_age                                  │
│ ├── Training distribution: mean=35, std=12         │
│ ├── Production (today):    mean=42, std=8          │
│ ├── PSI (Population Stability Index): 0.18         │
│ └── Status: ⚠️ WARNING (PSI > 0.1)                │
│                                                    │
│ Feature: text_length                               │
│ ├── Training distribution: mean=150, std=80        │
│ ├── Production (today):    mean=145, std=75        │
│ ├── PSI: 0.02                                      │
│ └── Status: ✅ STABLE                              │
│                                                    │
│ Actions:                                           │
│ - PSI > 0.1: Alert team                            │
│ - PSI > 0.25: Trigger retraining pipeline          │
└──────────────────────────────────────────────────┘
```

**Model Performance Monitoring:**

| Metric | Check Frequency | Alert Threshold |
|--------|----------------|-----------------|
| Prediction accuracy | Hourly (sample) | Drop > 5% from baseline |
| Latency P99 | Real-time | > 2x baseline |
| Error rate | Real-time | > 1% |
| Data drift (PSI) | Daily | PSI > 0.1 |
| Concept drift | Weekly | Accuracy drop > 3% |
| Feature availability | Real-time | Any feature missing |

---

## 7. Production Considerations

### 7.1 Cost Management

**LLM API Cost Estimation (per 100K queries/day):**

| Model | Input Cost | Output Cost | Total/Day | Total/Month |
|-------|-----------|-------------|-----------|-------------|
| GPT-4o-mini | $1.50 | $6.00 | $7.50 | $225 |
| GPT-4o | $25.00 | $100.00 | $125.00 | $3,750 |
| Claude Sonnet 4.5 | $30.00 | $150.00 | $180.00 | $5,400 |
| Claude Haiku | $8.00 | $40.00 | $48.00 | $1,440 |
| Llama 3.1 70B (self-hosted) | ~$50 GPU/day | — | $50 | $1,500 |

*Assumes avg 500 input tokens + 200 output tokens per query.*

**Embedding Cost Estimation (per 1M documents):**

| Component | Cost |
|-----------|------|
| Embedding (text-embedding-3-small) | ~$4 |
| Vector DB storage (Pinecone, 1M vectors) | ~$70/mo |
| Re-ranking (Cohere, 1M queries) | ~$1,000/mo |

### 7.2 Latency Budgets

| Component | Target P50 | Target P99 | Optimization |
|-----------|-----------|-----------|-------------|
| Embedding (query) | 30ms | 80ms | Batch, cache |
| Vector search | 10ms | 50ms | Index tuning, replicas |
| Re-ranking | 50ms | 150ms | Top-K limit, smaller model |
| LLM generation (streaming) | 200ms TTFT | 500ms TTFT | Model routing, cache |
| **Total RAG pipeline** | **300ms TTFT** | **800ms TTFT** | — |
| LLM generation (full) | 1.5s | 4s | Smaller model, max_tokens |
| **Total RAG (non-streaming)** | **2s** | **5s** | — |

### 7.3 Scaling Patterns

**Horizontal scaling (most common):**
- Multiple API server replicas behind load balancer
- Scale on request queue depth or latency
- Each replica is stateless (state in Redis/DB)

**GPU scaling:**
- GPU instances are expensive ($2-8/hour)
- Use spot/preemptible instances for batch inference (70% savings)
- Right-size GPU: Don't use A100 for a small model
- Quantization: INT8 reduces memory 2x, INT4 reduces 4x with ~1% quality loss

**Queue-based scaling:**
- Put inference requests in a queue (SQS, RabbitMQ)
- Workers pull from queue, scale workers with KEDA
- Good for: batch processing, handling traffic spikes

### 7.4 Observability for AI

**LLM-Specific Metrics to Track:**

| Metric | Why | Tool |
|--------|-----|------|
| Token usage (input/output) | Cost control | LangSmith, custom |
| Latency per model call | Performance | LangSmith, Prometheus |
| Cache hit rate | Cost optimization | Custom metrics |
| Hallucination rate | Quality | Eval pipeline |
| User satisfaction (thumbs up/down) | Quality | Custom |
| Retrieval relevance score | RAG quality | RAGAS |
| Error rate by model | Reliability | Prometheus |
| Cost per query | Budget | Custom |

**LLM Tracing (LangSmith):**
```
Trace: user_query_12345
├── Input: "What is our refund policy?"
├── Retrieval
│   ├── Query embedding: 45ms
│   ├── Vector search: 12ms (5 results)
│   ├── Re-ranking: 89ms (top 3)
│   └── Total retrieval: 146ms
├── Generation
│   ├── Model: gpt-4o
│   ├── Input tokens: 1,234
│   ├── Output tokens: 187
│   ├── TTFT: 198ms
│   ├── Total: 1,890ms
│   └── Cost: $0.0043
├── Guardrails
│   ├── PII check: 5ms (clean)
│   └── Toxicity: 3ms (clean)
├── Total latency: 2,044ms
└── User feedback: 👍
```

---

## 8. Common Trade-offs

### RAG vs Fine-tuning vs Prompt Engineering

| Factor | Prompt Engineering | RAG | Fine-tuning |
|--------|-------------------|-----|-------------|
| **Cost to start** | $0 | $100-500 | $500-10,000 |
| **Data needed** | 0 examples | 100+ documents | 1,000+ examples |
| **Time to deploy** | Hours | Days | Weeks |
| **Update frequency** | Instant | Minutes (re-index) | Hours-days (retrain) |
| **Accuracy on domain data** | Low-Medium | High | Highest |
| **Hallucination risk** | High | Low (grounded) | Medium |
| **Latency impact** | None | +100-500ms | None (or faster) |
| **Best for** | Prototyping, simple tasks | Knowledge-heavy apps | Style/format, classification |

**Decision flowchart:**
```
Need domain-specific knowledge?
├── No → Prompt Engineering
└── Yes → Knowledge changes frequently?
    ├── Yes → RAG
    └── No → Need specific output format/style?
        ├── Yes → Fine-tuning (+ RAG if also knowledge-heavy)
        └── No → RAG
```

### Accuracy vs Latency vs Cost Triangle

```
                    ACCURACY
                       ▲
                      / \
                     /   \
                    /     \
                   / Pick  \
                  /  Two    \
                 /           \
                ▼─────────────▼
            LATENCY ◀───────▶ COST
```

**You can optimize for two, but the third suffers:**
- **High Accuracy + Low Latency** = High Cost (GPT-4o, multiple retrieval passes, re-ranking)
- **High Accuracy + Low Cost** = High Latency (batch processing, open-source models, extensive eval)
- **Low Latency + Low Cost** = Lower Accuracy (small models, no re-ranking, simple retrieval)

### Open-Source vs Proprietary Models

| Factor | Proprietary (GPT-4o, Claude) | Open-Source (Llama, Mistral) |
|--------|------------------------------|------------------------------|
| **Quality** | Best (for now) | Close for many tasks |
| **Cost at scale** | $3-15/1M tokens | $0.50-2/1M (self-hosted) |
| **Data privacy** | Data sent to provider | Full control |
| **Latency** | 200-500ms TTFT | 50-200ms TTFT (self-hosted) |
| **Reliability** | 99.9% SLA | You own uptime |
| **Customization** | Prompt only | Fine-tune, quantize |
| **Setup effort** | Minutes | Days-weeks |
| **GPU needed** | No | Yes ($2-8/hr) |

**Decision: Start with proprietary APIs. Move to open-source when:**
- Cost exceeds $5K/month
- Data privacy requirements prevent API usage
- You need <100ms latency
- You need custom model behavior that prompting can't achieve

---

## 9. Interview Questions with Answers

### Q1: "Design a Customer Support Chatbot with RAG"

**Clarifying questions:**
- How many support articles? → ~10,000, updated weekly
- Daily users? → ~5,000 conversations/day
- Need to handle ticket creation? → Yes, escalate to human if can't resolve
- Latency requirement? → First response <2 seconds
- Languages? → English only for now

**High-level architecture:**
```
┌──────┐    ┌──────────┐    ┌────────────────────────────────────┐
│ User │───▶│ Chat UI  │───▶│ Chat Service (FastAPI)              │
│      │◀───│(React/WS)│◀───│                                    │
└──────┘    └──────────┘    │  ┌─────────────┐  ┌─────────────┐ │
                             │  │ Intent      │  │ RAG Pipeline│ │
                             │  │ Classifier  │──▶│             │ │
                             │  │(simple/faq/ │  │ Retrieve →  │ │
                             │  │ complex/    │  │ Rerank →    │ │
                             │  │ escalate)   │  │ Generate    │ │
                             │  └─────────────┘  └─────────────┘ │
                             │                                    │
                             │  ┌─────────────┐  ┌─────────────┐ │
                             │  │ Ticket      │  │ Conversation│ │
                             │  │ Service     │  │ Memory      │ │
                             │  │ (Zendesk)   │  │ (Redis)     │ │
                             │  └─────────────┘  └─────────────┘ │
                             └────────────────────────────────────┘
```

**Key decisions:**
- **Intent classifier** (GPT-4o-mini, $0.15/1M): Route simple FAQs to cached answers, complex to RAG, frustrated users to human
- **RAG retrieval**: Hybrid search (vector + BM25) with Cohere re-ranking
- **Escalation**: If confidence < 0.7 or user says "talk to human" → create Zendesk ticket with conversation summary
- **Memory**: Redis for session state, last 5 turns in context

**Metrics to monitor:**
- Resolution rate (% resolved without human)
- CSAT score (post-chat survey)
- Escalation rate
- Average response latency
- Cost per conversation

---

### Q2: "Design a Document Search Engine for a Law Firm"

**Key requirements:** 500K legal documents, 200 lawyers, exact legal citations critical, privileged documents with strict access control.

**Architecture highlights:**
- **ACL-aware RAG**: Filter vector search by user's access permissions
- **Hybrid search**: Lawyers need exact case citations (BM25) + semantic understanding
- **Re-ranking**: Cross-encoder re-ranking — accuracy is critical for legal
- **Chunking**: Document-structure-aware (respect sections, paragraphs, footnotes)
- **Citation**: Every response must cite paragraph numbers, document IDs
- **Audit log**: Every query and result logged for compliance

**Trade-offs:**
- Higher latency acceptable (lawyers expect 3-5s for complex queries)
- Zero tolerance for hallucination (use faithfulness score > 0.95 gate)
- On-premise deployment for data sovereignty

---

### Q3: "Design a Real-Time Content Moderation System"

**Key requirements:** 1M posts/day, <500ms, text + images, minimize false positives.

**Architecture:**
```
Post → [Fast Classifier (50ms)] → Safe (95%) → Publish
                                 → Uncertain (4%) → [LLM Review (300ms)] → Safe/Block
                                 → Toxic (1%) → Block + Queue for Human Review
```

**Key decisions:**
- **Two-stage pipeline**: Fast classifier (DistilBERT) catches obvious cases, LLM handles edge cases
- **Why not LLM for everything?** At 1M posts/day, GPT-4o would cost ~$5K/day. Fast classifier costs ~$50/day
- **Image moderation**: CLIP or Google Vision API for image classification
- **Human-in-the-loop**: Uncertain cases go to moderation queue, human decisions feed back into training

---

### Q4: "Design an AI-Powered Code Review Tool"

**Key requirements:** Integrate with GitHub PRs, review for bugs, security, style, <30s per PR.

**Architecture highlights:**
- **Trigger**: GitHub webhook on PR creation/update
- **Context gathering**: Pull diff, full file context, repo conventions (from existing PRs)
- **Multi-pass review**:
  1. Security scan (static analysis + LLM for logic bugs)
  2. Bug detection (LLM with full function context)
  3. Style/readability (smaller model, compare to repo conventions)
- **Comment generation**: Post inline GitHub comments with suggestions
- **Learning**: Track which suggestions are accepted/rejected to improve

---

### Q5: "Design a Multi-Tenant LLM Platform"

**Key requirements:** 100 enterprise clients, each with different models, rate limits, data isolation.

**Architecture:**
```
Client A ─┐
Client B ─┼──▶ API Gateway ──▶ Tenant Router ──▶ [Model Pool]
Client C ─┘    (API key →      (config lookup)    ├── OpenAI
                tenant ID)                         ├── Anthropic
                                                   ├── Self-hosted Llama
                                                   └── Custom fine-tuned
```

**Key decisions:**
- **Tenant isolation**: Separate API keys, rate limits, model configs per tenant
- **Data isolation**: No cross-tenant data leakage. Separate vector stores per tenant
- **Usage tracking**: Log every API call with tenant ID, model, tokens, cost
- **Billing**: Calculate cost per tenant, support different pricing tiers
- **Guardrails per tenant**: Each tenant can configure their own content policies

---

## Quick Reference Card

### The 5-Phase Framework
```
1. CLARIFY (5 min)   → Users, scale, latency, accuracy, constraints
2. HIGH-LEVEL (10m)  → Draw boxes and arrows, name components
3. DEEP DIVE (20m)   → Detail 2-3 components with specifics
4. OPERATIONS (10m)  → Monitoring, cost, scaling, failures
5. TRADE-OFFS (5m)   → What would you change? Biggest risk?
```

### Numbers to Know
```
Embedding latency:     30-80ms per query
Vector search:         10-50ms per query
LLM TTFT (streaming):  200-500ms
LLM full response:     1-4 seconds
Re-ranking:            50-150ms

GPT-4o:          $2.50 input / $10 output per 1M tokens
GPT-4o-mini:     $0.15 input / $0.60 output per 1M tokens
Claude Sonnet:   $3 input / $15 output per 1M tokens
Claude Haiku:    $0.80 input / $4 output per 1M tokens
Embeddings:      $0.02-0.13 per 1M tokens

Pinecone:        ~$70/mo per 1M vectors
Qdrant:          ~$25/mo self-hosted per 1M vectors
```

### Magic Phrases for Interviews
- "Let me start by clarifying the requirements..."
- "The trade-off here is between X and Y..."
- "At this scale, we'd need to consider..."
- "For monitoring, I'd track these key metrics..."
- "If we needed to scale 10x, I'd change..."
- "The biggest risk in this design is..."
- "I'd start simple with X, then migrate to Y as we scale..."
