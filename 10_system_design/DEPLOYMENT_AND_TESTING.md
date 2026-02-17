# Deploying & Testing AI Applications in Production

> Most AI tutorials stop at `model.predict()`. This guide explains everything after — how production AI apps are packaged, deployed, tested, and kept running. It focuses on **concepts and decision-making**, not just config files.

---

## Table of Contents

1. [AI Apps vs Normal Apps — What's Different?](#1-ai-apps-vs-normal-apps--whats-different)
2. [The Gap Between Demo and Production](#2-the-gap-between-demo-and-production)
3. [Structuring an AI App for Production](#3-structuring-an-ai-app-for-production)
4. [Containerization — Why Docker Matters for AI](#4-containerization--why-docker-matters-for-ai)
5. [Choosing a Deployment Strategy](#5-choosing-a-deployment-strategy)
6. [CI/CD for AI — What's Different](#6-cicd-for-ai--whats-different)
7. [Testing AI Applications — The Four Layers](#7-testing-ai-applications--the-four-layers)
8. [Deployment Patterns for AI](#8-deployment-patterns-for-ai)
9. [Monitoring AI in Production](#9-monitoring-ai-in-production)
10. [Security for AI Applications](#10-security-for-ai-applications)
11. [The Production Readiness Checklist](#11-the-production-readiness-checklist)

---

## 1. AI Apps vs Normal Apps — What's Different?

Before diving into deployment, it's important to understand what makes an AI application fundamentally different from a traditional software application — and where they overlap.

### The Core Difference

A **normal software app** is deterministic — the same input always gives the same output. A REST API takes a request, runs some logic, queries a database, and returns a predictable response. You can test it with exact assertions: `assert response == expected`.

An **AI app** is non-deterministic — the same input can produce different outputs. It depends on external model APIs, retrieved context, and probabilistic generation. You can't test with exact assertions. Instead, you ask: "Is this answer *good enough*?" This single difference affects almost everything about how you build, test, deploy, and monitor.

### What's the Same (You Already Know ~70%)

The good news: most of the deployment stack is identical. If you know how to deploy a web app, you're 70% of the way to deploying an AI app.

| Aspect | Same Tools? | Details |
|--------|------------|---------|
| **Language / Framework** | Yes | Python + FastAPI is the standard for both |
| **Containerization** | Yes | Same Docker, same Dockerfile patterns |
| **CI/CD Platform** | Yes | GitHub Actions, GitLab CI, Jenkins — all work the same |
| **Deployment Target** | Yes | Cloud Run, ECS, Kubernetes — identical |
| **Load Balancer** | Yes | ALB, nginx, Cloudflare — no difference |
| **Secrets Management** | Yes | Same Secret Manager / Vault (just more API keys to manage) |
| **Caching** | Yes | Same Redis — but AI apps add *semantic* caching |
| **Auto-scaling** | Yes | Same HPA / auto-scaling groups — different bottleneck though |

### What's Different (The ~30% That's New)

| Aspect | Normal App | AI App |
|--------|-----------|--------|
| **Testing** | Exact assertions: `assert result == expected` | Quality metrics: "Is faithfulness >= 0.85?" |
| **Cost model** | Fixed infrastructure cost ($200/mo) | Per-request API cost ($0.001-0.05 per query) that can spike |
| **New infrastructure** | Just databases (PostgreSQL, Redis) | Add **vector databases** (Qdrant, Pinecone) for semantic search |
| **What gets deployed** | Code | Code + prompts + models + data (each can change independently) |
| **Monitoring** | "Is it up? Is it fast?" | Also: "Is the AI output still good? Is it hallucinating?" |
| **Security threats** | SQL injection, XSS, CSRF | Add: prompt injection, data leakage, PII exposure, cost attacks |
| **External dependency** | Optional (maybe 1-2 APIs) | Central — your entire app depends on an LLM API |
| **Failure mode** | Obvious (500 error, wrong data) | Subtle (plausible-sounding but wrong answers) |

### The 7 Things That Are Genuinely New

**1. Your most important dependency is an external API**

A normal app might call Stripe or Twilio occasionally. An AI app's **entire value** comes from the LLM API. If OpenAI is down, your app is useless. This means you need fallback models (OpenAI → Anthropic), aggressive retry logic, and your latency budget is dominated by the LLM call (1-3 seconds) rather than your code (50ms).

**2. Testing requires a new layer: evaluation**

Normal tests check "does this function return the right value?" AI tests check "is this answer good enough?" You need an **eval dataset** (50+ curated question-answer pairs) and metrics like retrieval recall, faithfulness, and hallucination rate. This eval layer doesn't exist in traditional software.

**3. Cost is per-request, not just infrastructure**

A normal app costs ~$200/month for servers regardless of traffic. An AI app paying $0.01 per GPT-4o request at 100K requests/day spends **$1,000/day** just on API fees. You need per-request cost tracking, budget alerts, model routing (cheap model for simple queries), and caching to control this.

**4. You need vector databases (new infrastructure)**

Normal apps use PostgreSQL and Redis. AI apps add a **vector database** (Qdrant, Pinecone, Weaviate, pgvector) to store and search document embeddings. This is an entirely new piece of infrastructure to deploy, manage, monitor, and back up.

**5. "Deployment" means more than just code**

In a normal app, deployment = ship new code. In an AI app, there are four things that can change independently, each requiring different testing:

| What Changes | Example | Needs Eval? | Needs Code Deploy? |
|-------------|---------|-------------|-------------------|
| Code | New endpoint, bug fix | Sometimes | Yes |
| Prompt | Different system prompt | **Yes** — behavior changes dramatically | No (config change) |
| Model | Swap GPT-4o-mini for Claude Haiku | **Yes** — output quality changes | No (config change) |
| Data | New documents indexed | **Yes** — retrieval quality changes | No |

A prompt change that looks trivial can completely change response quality. Every type of change needs its own eval gate.

**6. Monitoring has an extra dimension: output quality**

Normal monitoring asks: "Is the server healthy?" AI monitoring adds: "Is the AI still giving good answers?" You need to sample production responses and evaluate them regularly. A 2% increase in hallucination rate won't trigger error alerts — it looks perfectly healthy from an infrastructure perspective.

**7. New security attack vectors**

| Normal App Threat | AI App Equivalent |
|-------------------|-------------------|
| SQL injection (manipulate database) | Prompt injection (manipulate AI behavior) |
| Unauthorized data access | AI reveals other users' documents (data leakage) |
| Data breach | AI includes PII (SSN, emails) in responses |
| DDoS attack | Cost attack (send expensive queries to drain budget) |

### The Deployment Stack Side-by-Side

```
NORMAL WEB APP:                       AI / RAG APP:

Code (Python/Node)                    Code (Python/FastAPI)
    │                                     │
Docker                                Docker                     ← SAME
    │                                     │
GitHub Actions                        GitHub Actions             ← SAME
  ├── Lint                              ├── Lint                  ← SAME
  ├── Unit Tests                        ├── Unit Tests            ← SAME
  ├── Integration Tests                 ├── Integration Tests     ← SAME
  │                                     ├── Eval Tests            ← NEW
  │                                     ├── Cost Check            ← NEW
  ├── Build                             ├── Build                 ← SAME
  └── Deploy                            └── Deploy (canary)       ← SAME
    │                                     │
Cloud Run / ECS / K8s                 Cloud Run / ECS / K8s      ← SAME
    │                                     │
PostgreSQL + Redis                    PostgreSQL + Redis          ← SAME
                                      + Vector DB (Qdrant)       ← NEW
                                      + LLM APIs (OpenAI)        ← NEW
    │                                     │
Prometheus + Grafana                  Prometheus + Grafana        ← SAME
                                      + LLM Tracing (LangSmith)  ← NEW
                                      + Cost Dashboard            ← NEW
                                      + Quality Monitoring        ← NEW
```

### The Bottom Line

If you know how to deploy a normal web app, you already understand Docker, CI/CD, cloud platforms, monitoring, and scaling. **That foundation carries over directly.** What you need to learn on top is:

- **Eval testing** (the AI quality gate in your CI/CD pipeline)
- **Vector databases** (new infrastructure to manage)
- **Cost management** (per-request tracking, model routing, caching)
- **Quality monitoring** (output quality, not just uptime)
- **AI-specific security** (prompt injection, data leakage)

The rest of this guide covers each of these in detail.

---

## 2. The Gap Between Demo and Production

A Jupyter notebook that answers questions is **not** a production app. Here's what changes:

| Concern | Demo/Notebook | Production |
|---------|--------------|------------|
| **Runs where** | Your laptop | Cloud server, available 24/7 |
| **Handles errors** | Crashes, you restart | Must recover automatically |
| **API keys** | Hardcoded in cell | Securely stored in secret manager |
| **Multiple users** | Just you | Hundreds or thousands concurrently |
| **Updates** | Re-run the notebook | Zero-downtime deployment |
| **Cost tracking** | You check your OpenAI dashboard | Per-request logging, daily budgets, alerts |
| **Quality** | "It seems to work" | Automated eval suite with pass/fail gates |
| **Scaling** | Can't | Auto-scales with demand |

### The Production Journey

```
RESEARCH           →  ENGINEERING         →  OPERATIONS

Jupyter Notebook   →  Structured App      →  Deployed Service
"It works on my     "It's packaged,        "It runs 24/7, auto-scales,
 laptop"             tested, and             self-heals, and we know
                     containerized"          when it's degrading"
```

The three transformations you need to make:

1. **Notebook → Structured App**: Separate concerns (routes, services, config), add health checks, handle errors
2. **App → Container**: Package everything reproducibly so it runs identically everywhere
3. **Container → Production Service**: CI/CD, monitoring, scaling, security, cost controls

---

## 3. Structuring an AI App for Production

### Why Structure Matters

In a notebook, everything is in one file. In production, you separate concerns so that:
- **Configs can change** without code changes (model name, temperature, chunk size)
- **Services are testable** independently (test chunking without calling the LLM)
- **Endpoints are clear** (health check, chat, admin routes)
- **Startup is controlled** (load models once, not per-request)

### The Standard Layout

```
my-ai-app/
├── app/
│   ├── main.py           # FastAPI app — creates the server, registers routes
│   ├── config.py          # All settings from environment variables
│   ├── routers/           # HTTP endpoints (chat, health, admin)
│   ├── services/          # Business logic (RAG pipeline, LLM calls, embedding)
│   ├── models/            # Pydantic schemas for request/response validation
│   └── utils/             # Prompt templates, helpers
├── tests/                 # Unit, integration, and eval tests
├── Dockerfile             # How to package the app
├── docker-compose.yml     # Local dev environment (app + vector DB + Redis)
├── .github/workflows/     # CI/CD pipelines
├── .env.example           # Template showing required environment variables
└── requirements.txt       # Dependencies
```

### Key Design Decisions

**1. Health Checks — Why You Need Two**

Every production app needs at least two health endpoints:

- `/health` — **Liveness probe**: "Is the process running?" Returns 200 if the app hasn't crashed. Kubernetes uses this to know when to restart a pod.
- `/ready` — **Readiness probe**: "Can it handle traffic?" Checks that the vector DB is connected, the LLM API is reachable, and any models are loaded. Kubernetes uses this to know when to send traffic.

Why both? An app might be alive (process running) but not ready (database connection lost). Without a readiness probe, Kubernetes would send traffic to a broken instance.

**2. Configuration — Environment Variables, Not Hardcoded Values**

Every setting that might change between environments (dev, staging, prod) should come from environment variables:

- API keys (`OPENAI_API_KEY`)
- Model names (`LLM_MODEL=gpt-4o-mini`)
- Tuning parameters (`CHUNK_SIZE=512`, `TOP_K=5`, `LLM_TEMPERATURE=0.1`)
- Service URLs (`VECTOR_DB_URL=http://qdrant:6333`)

**Why?** You can change the model from `gpt-4o-mini` to `claude-haiku` by updating one environment variable — no code change, no redeploy. This is critical for A/B testing and cost optimization.

Use a library like `pydantic-settings` to validate that all required variables are present at startup (fail fast, not when the first request arrives).

**3. Startup vs Per-Request — Load Expensive Things Once**

Loading a model, connecting to a vector DB, or initializing an embedding pipeline takes seconds. Don't do it on every request.

Use FastAPI's **lifespan** pattern:
- **At startup**: Connect to vector DB, load embedding models, warm up caches
- **Per request**: Use the pre-loaded resources to answer queries
- **At shutdown**: Close connections gracefully, flush logs

This is the difference between a 2-second first response and a 200ms first response.

---

## 4. Containerization — Why Docker Matters for AI

### The Problem Docker Solves

"It works on my machine" is the #1 deployment problem. AI apps are especially fragile because they depend on:
- Specific Python versions
- ML library versions (PyTorch, Transformers, etc.)
- System libraries (for OCR, PDF parsing, etc.)
- Downloaded model files

Docker creates an **identical environment** everywhere — your laptop, CI server, and production cloud.

### How Docker Works for AI Apps

```
Dockerfile = Recipe:
  "Start with Python 3.11
   Install these Python packages
   Copy my application code
   Run the server on port 8000"

Docker Image = Snapshot:
  Everything packaged into a single file (~500MB-2GB for AI apps)

Docker Container = Running Instance:
  The image actually running, serving requests
```

### AI-Specific Docker Considerations

| Consideration | What It Means | Solution |
|---------------|---------------|----------|
| **Large images** | ML libraries make images 2-5 GB | Use multi-stage builds: install in stage 1, copy only needed files to stage 2 |
| **Model files** | Downloaded models add 500MB-5GB | Download during build (cached in layers) or load from cloud storage at startup |
| **GPU support** | Some models need NVIDIA GPUs | Use `nvidia/cuda` base images, deploy to GPU instances |
| **Slow builds** | `pip install` takes 5-10 min | Copy `requirements.txt` before source code → Docker caches the install layer |
| **Security** | Container shouldn't run as root | Add `USER appuser` — limits damage if the container is compromised |
| **Secrets** | API keys must not be in the image | Pass via environment variables at runtime, never `COPY .env` |
| **Health checks** | Container platform needs to know when app is ready | Add `HEALTHCHECK` instruction pointing to your `/health` endpoint |

### Multi-Stage Builds Explained

Multi-stage builds solve the "image too big" problem:

```
Stage 1 (Builder):                    Stage 2 (Production):
┌─────────────────────┐               ┌─────────────────────┐
│ Full Python + GCC   │               │ Slim Python only    │
│ + all build tools   │    Copy only  │                     │
│ + source code       │ ─────────────▶│ + installed packages│
│ + pip install       │  what's needed│ + app code          │
│ SIZE: ~2.5 GB       │               │ SIZE: ~800 MB       │
└─────────────────────┘               └─────────────────────┘
```

Stage 1 installs everything (including build tools like GCC for C extensions). Stage 2 starts fresh with a slim image and only copies the installed packages and app code. Build tools, source distributions, and caches are left behind.

### Docker Compose for Local Development

In production, your AI app depends on multiple services:

```
┌────────────┐     ┌────────────┐     ┌────────────┐
│  Your App  │────▶│ Vector DB  │     │   Redis    │
│  (FastAPI) │     │ (Qdrant)   │     │  (Cache)   │
│  Port 8000 │     │ Port 6333  │     │ Port 6379  │
└────────────┘     └────────────┘     └────────────┘
```

Docker Compose defines all services in one file and starts them together with `docker compose up`. It handles networking (services find each other by name), volume persistence (vector DB data survives restarts), and dependency ordering (app waits for Qdrant to be healthy before starting).

This gives every developer the same local environment — no "install Qdrant on your machine" instructions.

---

## 5. Choosing a Deployment Strategy

### The Options

| Approach | Best For | Monthly Cost | Ops Effort | When to Choose |
|----------|----------|-------------|------------|----------------|
| **Serverless Containers** (Cloud Run, Azure Container Apps) | Most AI apps | $30-200 | Very Low | You want to focus on the AI, not infra. Auto-scales to zero when idle |
| **Managed Containers** (ECS Fargate) | Mid-size apps | $100-500 | Low-Medium | You need more control than serverless but don't want K8s |
| **Kubernetes** (EKS, GKE) | Large, multi-service platforms | $200+ | High | Multiple services, strict requirements, GPU orchestration |
| **PaaS** (Railway, Render, Fly.io) | MVPs, side projects | $5-50 | Minimal | "Just deploy my Docker image" |
| **AI-Specific** (Modal, Replicate, Together AI) | Model serving | Pay-per-use | Minimal | Serving open-source models, GPU-heavy workloads |

### Decision Flowchart

```
Is this a side project or MVP?
├── Yes → Railway / Render / Fly.io (cheapest, simplest)
└── No → Do you need GPUs for model inference?
    ├── Yes → Modal / Replicate (GPU serverless) or K8s with GPU nodes
    └── No → How many services?
        ├── 1-2 services → Cloud Run / Azure Container Apps (serverless)
        ├── 3-5 services → ECS Fargate (managed containers)
        └── 5+ services → Kubernetes (full orchestration)
```

### What Happens During a Deployment

Regardless of platform, a deployment follows this flow:

```
1. BUILD       → Docker image from your code
2. PUSH        → Image to a container registry (ECR, GCR, Docker Hub)
3. DEPLOY      → Platform pulls image, starts new containers
4. HEALTH CHECK → Platform verifies /health and /ready return 200
5. TRAFFIC SHIFT → Platform routes traffic from old → new containers
6. SCALE DOWN  → Old containers are stopped
```

**Key concept: Zero-downtime deployment.** The platform runs both old and new versions simultaneously during step 5. Old containers keep serving until new ones pass health checks. Users never see an error.

### Secrets Management

API keys (OpenAI, Anthropic, etc.) are the #1 security risk in AI apps. The rule is simple: **secrets should never be in your code, Docker image, or git repository.**

| Environment | Where Secrets Live |
|-------------|-------------------|
| Local dev | `.env` file (added to `.gitignore`) |
| CI/CD | GitHub Secrets (encrypted, injected at runtime) |
| Cloud Run | GCP Secret Manager → mounted as env vars |
| Kubernetes | K8s Secrets or External Secrets Operator |
| ECS | AWS Secrets Manager → referenced in task definition |

The pattern is always: store secrets in a dedicated secret manager → inject them as environment variables at container startup → your app reads them from `os.environ`.

---

## 6. CI/CD for AI — What's Different

### Traditional CI/CD vs AI CI/CD

Traditional software CI/CD asks: **"Does the code work?"**
AI CI/CD adds: **"Does the AI perform well enough?"**

```
TRADITIONAL CI/CD PIPELINE:

  Push Code → Lint → Unit Tests → Integration Tests → Build → Deploy
                                                         │
                                                    All pass?
                                                    Yes → Ship it

AI CI/CD PIPELINE (additional stages):

  Push Code → Lint → Unit Tests → Integration Tests
                                        │
                                        ▼
                              ┌─────────────────────┐
                              │  EVAL TESTS          │  ← NEW: AI-specific
                              │                      │
                              │  - Retrieval quality  │  "Did we find the right docs?"
                              │  - Answer quality     │  "Is the answer correct?"
                              │  - Hallucination rate │  "Did it make things up?"
                              │  - Latency check      │  "Is it fast enough?"
                              │  - Cost estimate      │  "Are we within budget?"
                              └─────────┬─────────────┘
                                        │
                                  All pass?
                                  Yes ▼
                              ┌─────────────────────┐
                              │  BUILD & DEPLOY      │
                              │                      │
                              │  Build Docker image   │
                              │  Deploy canary (10%) │
                              │  Monitor for 5 min   │
                              │  Promote to 100%     │
                              └─────────────────────┘
                                        │
                                        ▼
                              ┌─────────────────────┐
                              │  POST-DEPLOY         │  ← NEW: ongoing
                              │                      │
                              │  - Monitor drift      │
                              │  - Track cost/day     │
                              │  - Sample quality     │
                              └─────────────────────┘
```

### The Three CI/CD Pipelines You Need

**Pipeline 1: CI (runs on every pull request)**

This is fast and cheap. It catches code bugs before they reach `main`.

| Step | Purpose | Runtime | Cost |
|------|---------|---------|------|
| Lint + Format | Catch style issues | 30s | Free |
| Type Check | Catch type errors | 30s | Free |
| Unit Tests | Verify functions work | 1 min | Free |
| Integration Tests | Verify components work together | 3 min | Free (uses mocked LLM) |
| Docker Build | Verify the image builds | 2 min | Free |
| Eval Tests | Verify AI quality hasn't regressed | 5-10 min | ~$0.50-2 per run (real LLM calls) |

**Pipeline 2: Deploy (runs on merge to main)**

| Step | Purpose | How |
|------|---------|-----|
| Build Docker image | Create production image | `docker build` with Git SHA as tag |
| Push to registry | Store image | Push to GCR / ECR / Docker Hub |
| Deploy canary | Test in production with 10% traffic | Deploy tagged revision, split traffic |
| Health check | Verify canary is healthy | Monitor error rate and latency for 5 minutes |
| Promote or rollback | Full deployment or revert | If healthy → 100% traffic. If errors → roll back to previous version |
| Notify team | Visibility | Slack message with deploy status |

**Pipeline 3: Model Update (manually triggered)**

When you change models (e.g., swap `gpt-4o-mini` for `claude-haiku`, or update your embedding model), you need a separate pipeline because:

- It's not a code change — it's a config/model change
- The impact is different — the same code might produce very different outputs
- It needs its own eval comparison — "Is the new model at least as good?"

| Step | Purpose |
|------|---------|
| Run eval suite with new model | Generate quality scores |
| Compare with production model | New model must be >= current on all metrics |
| Estimate cost impact | Will this save or cost more money? |
| Deploy with canary | Test with real traffic |
| Monitor + promote | Same canary flow as code deploys |

### Eval Gates — The AI-Specific Quality Gate

The key innovation in AI CI/CD is the **eval gate**: an automated check that blocks deployment if AI quality drops below a threshold.

| Gate | Threshold | What It Prevents |
|------|-----------|-----------------|
| Retrieval Recall@5 >= 0.80 | Must find the correct document in top 5 results, 80% of the time | Broken retrieval (wrong chunks, bad embeddings) |
| Faithfulness >= 0.85 | 85% of answer claims must be grounded in retrieved context | Hallucination increase |
| Answer Relevance >= 0.80 | 80% of answers must address the actual question | Off-topic or generic responses |
| P99 Latency <= 5s | 99th percentile response time under 5 seconds | Performance regression |
| Estimated daily cost <= $X | Cost within budget | Accidentally using expensive model |

**How it works:**
1. You maintain a curated **eval dataset** — 50-100 question-answer pairs with known-correct sources
2. CI runs your RAG pipeline against this dataset
3. A script compares the results against thresholds
4. If any metric is below threshold → CI fails → deployment blocked

This is the AI equivalent of "all tests must pass before merging."

### Implementing in GitHub Actions

GitHub Actions is the most common CI/CD platform. Key concepts:

- **Workflows** (`.github/workflows/*.yml`): Define what runs and when
- **Triggers**: `on: pull_request` (every PR) or `on: push to main` (every merge)
- **Jobs**: Independent stages that run in parallel (lint, test, build)
- **Services**: Spin up Docker containers (Qdrant, Redis) for integration tests
- **Secrets**: Encrypted variables (API keys) injected at runtime via `${{ secrets.NAME }}`
- **Artifacts**: Save test results, coverage reports for later review

The integration test job is especially useful — GitHub Actions can spin up a real Qdrant and Redis container as "services" alongside your test runner, giving you a realistic environment without any external infrastructure.

---

## 7. Testing AI Applications — The Four Layers

### The Testing Pyramid for AI

```
                    ┌──────────────────┐
                    │   E2E Tests      │  Few, slow, expensive
                    │ "Full user flow" │  Run: before release
                    ├──────────────────┤
                    │   Eval Tests     │  AI-specific quality
                    │ "Is the AI good?"│  Run: on PRs (sampled)
                    ├──────────────────┤
                    │ Integration Tests│  Component interaction
                    │ "Do parts work   │  Run: on every PR
                    │  together?"      │
                    ├──────────────────┤
                    │   Unit Tests     │  Many, fast, free
                    │ "Does each       │  Run: on every commit
                    │  function work?" │
                    └──────────────────┘
```

### Layer 1: Unit Tests

**What they test:** Individual functions in isolation, with no external dependencies.

**What to unit test in an AI app:**

| Component | What to Test | Example |
|-----------|-------------|---------|
| Chunking | Correct split sizes, overlap works, handles empty text | "Does `chunk_text('long text', 512)` return chunks under 512 tokens?" |
| Prompt templates | Variables are inserted correctly, output format is right | "Does `build_prompt(question, context)` include both?" |
| Schema validation | Pydantic models accept valid input, reject invalid | "Does `ChatRequest(message='')` raise a validation error?" |
| Metadata extraction | Title, author, date parsed correctly | "Does `extract_metadata(doc)` return the expected fields?" |
| Cost calculation | Token counts → dollar amounts are correct | "Does `calculate_cost('gpt-4o', 1000, 500)` return $0.0075?" |
| Text preprocessing | Cleaning, normalization work | "Does `clean_text(html_string)` strip tags correctly?" |

**Key principle:** Unit tests must be **fast** (no API calls, no database, no network) and **deterministic** (same input always gives same output). You should have 50-200 of these. They run in seconds.

### Layer 2: Integration Tests

**What they test:** Multiple components working together.

**Strategy:** Use **real** infrastructure services (vector DB, Redis) but **mock** expensive LLM API calls.

| Test | Real Services | Mocked Services | What It Verifies |
|------|--------------|----------------|-----------------|
| RAG pipeline | Vector DB (Qdrant) | LLM (OpenAI) | Query → embedding → retrieval → re-ranking works end-to-end |
| API endpoints | FastAPI server | LLM | HTTP requests return correct status codes and response shapes |
| Ingestion pipeline | Vector DB, file parsers | — | Documents are parsed, chunked, embedded, and stored correctly |
| Cache layer | Redis | LLM | Cache hits return cached response, cache misses call the pipeline |

**Why mock the LLM?** LLM API calls are:
- **Slow** (1-3 seconds each)
- **Expensive** ($0.001-0.01 per call)
- **Non-deterministic** (same prompt can give different answers)
- **Flaky** (API might be down or rate-limited)

By mocking the LLM, your integration tests run in 30 seconds instead of 10 minutes, cost $0 instead of $5, and never fail due to network issues.

### Layer 3: Eval Tests (AI-Specific)

**What they test:** The actual quality of your AI system's outputs.

This is what makes AI testing different from traditional software testing. You're not checking "does the function return the right type?" — you're checking "is the answer actually good?"

**Two categories:**

**A) Retrieval Evaluation** — Does the system find the right documents?

| Metric | What It Measures | How to Calculate | Good Score |
|--------|-----------------|-----------------|------------|
| **Recall@K** | "Is the correct doc in the top K results?" | (# of correct docs in top K) / (# of total correct docs) | >= 0.80 |
| **MRR** (Mean Reciprocal Rank) | "How high is the correct doc ranked?" | Average of 1/rank for the first correct result | >= 0.65 |
| **NDCG@K** | "Are results ordered by relevance?" | Normalized score comparing actual vs ideal ranking | >= 0.60 |
| **Context Precision** | "What % of retrieved chunks are actually relevant?" | (# relevant chunks) / (# total retrieved chunks) | >= 0.70 |

**B) Generation Evaluation** — Is the answer good?

| Metric | What It Measures | How to Calculate | Good Score |
|--------|-----------------|-----------------|------------|
| **Faithfulness** | "Is the answer grounded in the retrieved context?" | LLM-as-judge checks each claim against the context | >= 0.85 |
| **Answer Relevance** | "Does the answer address the question?" | LLM-as-judge scores relevance | >= 0.80 |
| **Hallucination Rate** | "Did the AI make things up?" | % of claims not supported by context | <= 5% |
| **Completeness** | "Did it answer the full question?" | Manual or LLM-as-judge | >= 0.70 |

**Tools for eval:**
- **RAGAS**: Open-source framework for RAG evaluation (faithfulness, relevance, precision, recall)
- **DeepEval**: Broader eval framework with hallucination detection
- **LangSmith**: Tracing + eval from LangChain
- **Custom**: Your own eval script comparing answers against expected keywords/sources

**The Eval Dataset — Your Most Important Asset**

Your eval dataset is a collection of question-answer-source triples:

```
Question: "What is our refund policy?"
Expected Source: "policies/refund-policy.md"
Expected Answer Contains: ["30 days", "full refund"]
Expected Answer Does NOT Contain: ["no refunds", "60 days"]
```

**How to build one:**
1. Start with 50 real or realistic questions
2. For each question, manually identify the correct source document
3. Write key phrases that should (and should not) appear in the answer
4. Include 5-10 "unanswerable" questions (the AI should say "I don't know")
5. Include edge cases: ambiguous questions, multi-part questions, questions requiring multiple documents

**Golden rule:** 50 well-curated examples > 500 sloppy ones. Quality over quantity. Add to it over time as you find failure cases.

### Layer 4: E2E and Manual Tests

For AI apps, automated eval catches most issues. But some things require human judgment:

- **Tone and style**: Is the response professional? Too wordy? Too terse?
- **Edge cases**: What happens with adversarial inputs? Prompt injection attempts?
- **User experience**: Is the streaming smooth? Do citations link to the right place?
- **Cultural sensitivity**: Are responses appropriate across cultures and contexts?

These are best done as part of a pre-release review, not on every PR.

### Load Testing — Can It Handle Real Traffic?

Before going live, you need to know: **how many concurrent users can this handle?**

**What to measure:**

| Metric | Target | Why |
|--------|--------|-----|
| P50 latency | < 2s | Typical user experience |
| P99 latency | < 5s | Worst-case experience — if P99 is 30s, 1 in 100 users waits 30 seconds |
| Max throughput | > expected peak QPS | Can you handle Black Friday? |
| Error rate under load | < 1% | Does the system degrade gracefully or crash? |
| Memory/CPU at peak | < 80% | Do you have headroom for spikes? |

Use a load testing tool (Locust, k6, or Artillery) to simulate concurrent users sending realistic queries. Start at 10 users, ramp to 100, and observe where things break.

**Common AI-specific bottlenecks:**
- LLM API rate limits (OpenAI: 10K RPM on Tier 3)
- Vector DB connection pool exhaustion
- Memory growth from loading too many documents
- Embedding API throughput (batching helps)

### Shadow Testing — Safe Model Swaps

When you want to switch models (e.g., `gpt-4o-mini` → `claude-haiku`), shadow testing lets you compare safely:

1. Both models process every request
2. Only the **current production model** response is served to users
3. The **shadow model** response is logged but never shown
4. After a few days, compare: quality, latency, cost
5. If the shadow model is equal or better → promote it

This eliminates the risk of switching models — you have real data before committing.

---

## 8. Deployment Patterns for AI

### Canary Deployments

The safest way to deploy AI changes. Instead of switching 100% of traffic at once:

```
Step 1: Deploy new version alongside old
Step 2: Route 10% of traffic to new version ("canary")
Step 3: Monitor for 5-30 minutes
         - Error rate stable?
         - Latency within bounds?
         - AI quality not degraded? (sample eval)
Step 4a: If healthy → gradually increase to 25% → 50% → 100%
Step 4b: If unhealthy → roll back to 0% → investigate
```

**Why canary matters for AI:** A code bug shows up instantly (500 errors). A bad prompt change or model regression might only show up as subtle quality degradation. The canary window gives you time to catch it.

### Blue-Green Deployments

Two identical environments:

```
BLUE  (current production) ← All traffic
GREEN (new version, idle)

1. Deploy new version to GREEN
2. Run smoke tests against GREEN
3. Switch load balancer: BLUE → GREEN
4. GREEN is now production
5. Keep BLUE running for 30 min (instant rollback if needed)
```

Simpler than canary but riskier — it's all-or-nothing. Good for small apps. Not recommended for AI apps where quality regression might be subtle.

### Rolling Updates (Kubernetes Default)

Kubernetes replaces pods one at a time:

```
Old: [Pod1] [Pod2] [Pod3]
         ↓ replace Pod1
Mid: [Pod1-new] [Pod2] [Pod3]    ← Pod1 new, health checked, receiving traffic
         ↓ replace Pod2
Mid: [Pod1-new] [Pod2-new] [Pod3]
         ↓ replace Pod3
New: [Pod1-new] [Pod2-new] [Pod3-new]
```

Fast and automatic. Good for code changes. For model changes, prefer canary (you need time to evaluate quality, not just health checks).

### Feature Flags for AI

Sometimes you want to test a new prompt, model, or retrieval strategy on a subset of users without deploying new code:

```
if feature_flag("use_claude_for_complex"):
    model = "claude-sonnet"
else:
    model = "gpt-4o-mini"
```

Tools: LaunchDarkly, Unleash, or a simple config in your database. Useful for A/B testing different AI strategies.

---

## 9. Monitoring AI in Production

### What's Different About Monitoring AI?

Traditional app monitoring tracks: **Is the server healthy? Are requests succeeding?**

AI monitoring adds: **Is the AI output still good? Are we spending too much?**

### The Three Pillars + AI Metrics

```
STANDARD INFRASTRUCTURE METRICS:
  ├── Latency (P50, P95, P99)
  ├── Error rate (4xx, 5xx)
  ├── Throughput (requests/sec)
  ├── CPU / Memory / GPU usage
  └── Pod restarts / health check failures

AI-SPECIFIC METRICS (what makes AI monitoring unique):
  ├── Token usage (input + output per request)
  ├── Cost per request and daily totals
  ├── Cache hit rate (semantic + exact)
  ├── Retrieval relevance score (sampled)
  ├── Hallucination rate (sampled)
  ├── User feedback (thumbs up/down ratio)
  ├── Model latency breakdown (embedding vs retrieval vs generation)
  └── Drift detection (are user queries changing?)
```

### Structured Logging — What to Log Per Request

Every request should produce a structured (JSON) log entry containing:

| Field | Why |
|-------|-----|
| `request_id` | Trace a single request through all components |
| `user_id` | Understand per-user patterns |
| `question` | Debug specific failures (careful: may contain PII) |
| `model` | Know which model was used |
| `tokens_in` / `tokens_out` | Cost tracking |
| `latency_ms` | Performance monitoring |
| `retrieval_count` | How many chunks were used |
| `cache_hit` | Was this a cached response? |
| `cost_usd` | Per-request cost |
| `sources` | Which documents were cited |

**Important: PII in logs.** User queries may contain personal information. Either:
- Mask PII before logging (using tools like Microsoft Presidio)
- Log to a restricted system with access controls
- Don't log the question at all (least useful, but safest)

### LLM Tracing — Understanding the Full Pipeline

Standard logging tells you "this request took 3 seconds." LLM tracing tells you **why**:

```
Request: "What is our refund policy?"
├── Query embedding:     45ms   ← fast
├── Vector search:       12ms   ← fast
├── Re-ranking:          89ms   ← 60% of retrieval time — optimize here?
├── Context assembly:    3ms    ← fast
├── LLM generation:      1,890ms ← 85% of total time (expected for LLM)
│   ├── Input tokens:    1,234
│   ├── Output tokens:   187
│   ├── Model:           gpt-4o
│   └── Cost:            $0.0043
├── Guardrail checks:    8ms    ← fast
└── Total:               2,047ms
```

Tools: **LangSmith** (from LangChain), **LangFuse** (open-source), **Arize** (ML observability).

### Alerting — When to Wake Someone Up

Not every metric needs an alert. Set up alerts only for conditions that require action:

| Alert | Threshold | Action |
|-------|-----------|--------|
| Error rate > 5% for 5 min | Immediate (PagerDuty) | Something is broken — investigate |
| P99 latency > 10s for 5 min | Immediate | LLM API might be slow, or system overloaded |
| LLM API errors > 10 in 1 min | Immediate | Switch to fallback model |
| Daily cost > 120% of budget | Warning (Slack) | Check for traffic spike or misconfiguration |
| Retrieval relevance < 0.6 (daily sample) | Warning | Possible data quality issue — check recent document updates |
| User satisfaction < 60% (weekly) | Advisory | Review recent changes to prompts, models, or retrieval |

**Anti-pattern: Alert fatigue.** If you get 20 alerts a day, you'll start ignoring them. Only alert on conditions that are actionable and require immediate attention.

### Cost Tracking Dashboard

AI apps can have runaway costs. A simple dashboard should show:

- **Daily spend** vs budget (bar chart with a red line at the budget)
- **Cost per request** over time (catch sudden increases)
- **Cost breakdown by model** (which model is the most expensive?)
- **Cache savings** (how much money is the cache saving?)
- **Projected monthly cost** based on current usage trend

### Drift Detection — When the World Changes

Your AI system was designed for specific types of queries. But user behavior changes:

- **Data drift**: Users start asking different kinds of questions than what you tested for
- **Concept drift**: The answers to existing questions change (policy updates, product changes)
- **Model drift**: The LLM provider changes something (model update, behavior change)

**How to detect drift:**
1. Log user queries and periodically cluster them — are new clusters appearing?
2. Run your eval suite weekly against production queries (sampled)
3. Track retrieval relevance scores — a slow decline signals something has changed
4. Monitor user feedback trends — sudden dip = something broke

---

## 10. Security for AI Applications

### AI-Specific Security Threats

| Threat | What Happens | Defense |
|--------|-------------|---------|
| **Prompt injection** | User crafts input that overrides system prompt: "Ignore previous instructions and..." | Input validation, prompt sandboxing, LLM-based detection |
| **Data leakage** | AI reveals information from other users' documents | ACL-aware retrieval, tenant isolation in vector DB |
| **PII exposure** | AI returns sensitive data (SSN, email) in responses | PII detection on output (Presidio), output filtering |
| **API key theft** | Attacker gets your OpenAI key from exposed env file or log | Secret manager, never log or commit keys, rotate regularly |
| **Cost attack** | Malicious user sends thousands of expensive queries | Rate limiting, per-user quotas, cost circuit breaker |
| **Model abuse** | User uses your AI to generate harmful content | Content policy guardrails, input/output filtering |

### Defense in Depth

```
Layer 1: INPUT
  ├── Rate limiting (max requests per user per minute)
  ├── Input validation (max length, allowed characters)
  ├── PII detection (scan for SSN, credit cards, etc.)
  └── Prompt injection detection (classifier or pattern matching)

Layer 2: PROCESSING
  ├── ACL enforcement (only search docs the user can access)
  ├── Token budgeting (max_tokens limit per request)
  ├── Model isolation (no cross-tenant context)
  └── Timeout enforcement (kill requests after 30s)

Layer 3: OUTPUT
  ├── PII scanning (remove any PII that slipped through)
  ├── Content policy check (no harmful, illegal, or off-topic content)
  ├── Source attribution (only cite documents from the user's search results)
  └── Response length limit

Layer 4: INFRASTRUCTURE
  ├── Secrets in secret manager (never in code or env files)
  ├── HTTPS everywhere
  ├── Authentication on all endpoints (except /health)
  ├── Audit logging (who accessed what)
  └── Network policies (containers can only talk to allowed services)
```

---

## 11. The Production Readiness Checklist

### Before Going Live

**Application:**
- [ ] Health endpoint (`/health`) returns 200
- [ ] Readiness endpoint (`/ready`) checks all dependencies
- [ ] All config from environment variables (no hardcoded secrets)
- [ ] Input validation on every endpoint (Pydantic schemas)
- [ ] Rate limiting configured per user/API key
- [ ] Request timeouts set (30s default, 60s for LLM calls)
- [ ] Graceful shutdown handles in-flight requests
- [ ] Error responses are informative but don't leak internals

**Docker:**
- [ ] Non-root user in Dockerfile
- [ ] `.dockerignore` excludes `.env`, `.git`, `__pycache__`, `venv`, `*.pyc`
- [ ] Base image version pinned (not `latest`)
- [ ] Health check instruction present
- [ ] Image size reasonable (<2 GB for most apps)

**CI/CD:**
- [ ] Lint + unit tests on every PR
- [ ] Integration tests on every PR
- [ ] Eval tests on PRs (with pass/fail thresholds)
- [ ] Docker build verified in CI
- [ ] Automated deployment on merge to main
- [ ] Canary deployment (not all-at-once)
- [ ] Rollback plan documented and tested

**Testing:**
- [ ] Unit tests cover: chunking, prompts, schemas, cost calculation, text processing
- [ ] Integration tests cover: API endpoints, RAG pipeline, cache behavior
- [ ] Eval dataset with 50+ curated Q&A pairs
- [ ] Eval gates: retrieval recall >= 0.80, faithfulness >= 0.85, hallucination <= 5%
- [ ] Load test confirms P99 < 5s at expected peak QPS

**Monitoring:**
- [ ] Structured JSON logging with request_id, model, tokens, cost
- [ ] Metrics dashboard: latency, error rate, throughput
- [ ] AI metrics: token usage, cost/request, cache hit rate
- [ ] LLM tracing enabled (LangSmith, LangFuse, or custom)
- [ ] Alerting rules configured (error rate, latency, cost, API failures)
- [ ] Cost dashboard with daily budget and alerts
- [ ] User feedback collection (thumbs up/down)

**Security:**
- [ ] API keys in secret manager
- [ ] PII detection on inputs and outputs
- [ ] Prompt injection defense
- [ ] No sensitive data in logs
- [ ] HTTPS enforced
- [ ] Authentication on all non-health endpoints
- [ ] Rate limiting to prevent cost attacks

**Reliability:**
- [ ] Auto-scaling configured (min 2 replicas for availability)
- [ ] LLM API fallback (OpenAI down → switch to Anthropic)
- [ ] Retry with exponential backoff for transient failures
- [ ] Circuit breaker for external dependencies
- [ ] Database/vector DB connection pooling

---

## The Maturity Ladder

Where does your deployment stand?

```
LEVEL 0 — Prototype
  "I share my ngrok URL with teammates"
  Code runs on laptop. No CI/CD, tests, or monitoring.

LEVEL 1 — Containerized
  "It runs in Docker on a VM"
  Dockerfile exists. Basic health check. Manual deploys.

LEVEL 2 — Automated
  "CI/CD deploys on push to main"
  GitHub Actions for tests + deploy. Auto-deploy to Cloud Run/ECS.
  Unit and integration tests. Still no AI-specific evaluation.

LEVEL 3 — Production-Ready
  "We test AI quality and deploy with canary"
  Eval tests with pass/fail gates. Canary deployments.
  LLM tracing. Structured logging. Cost tracking.

LEVEL 4 — Enterprise
  "We optimize cost, run A/B tests, and handle failure gracefully"
  Model routing by complexity. Shadow testing for model swaps.
  Automatic drift detection. Disaster recovery plan.

Most AI apps should target Level 2-3.
Level 4 when you're spending >$5K/month on AI APIs.
```

---

## Summary: The Key Takeaways

1. **Structure your app** for production from day one — separate config, services, and routes
2. **Docker** makes "it works on my machine" a thing of the past
3. **CI/CD for AI** adds eval gates — automated quality checks that block bad deployments
4. **Test at four layers** — unit (fast, free), integration (mock the LLM), eval (real quality), E2E (human review)
5. **Your eval dataset is your most valuable asset** — curate 50+ examples and grow it over time
6. **Deploy with canary** — 10% traffic first, monitor, then promote
7. **Monitor AI-specific metrics** — tokens, cost, cache hit rate, retrieval quality, user feedback
8. **Security matters more for AI** — prompt injection, PII leakage, and cost attacks are real threats
9. **Shadow test model swaps** — run both models in parallel before committing
10. **Start at Level 2, grow to Level 3** — automated tests + automated deploys covers 90% of needs
