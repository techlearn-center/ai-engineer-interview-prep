# Deploying & Testing AI Applications in Production

> **The missing guide.** Most AI tutorials stop at `model.predict()`. This guide covers everything after that — containerizing, deploying, setting up CI/CD, testing, and running AI apps in production.

---

## Table of Contents

1. [From Notebook to Production — The Full Journey](#1-from-notebook-to-production)
2. [Containerizing AI Apps with Docker](#2-containerizing-ai-apps-with-docker)
3. [Deploying to Production](#3-deploying-to-production)
4. [CI/CD Pipelines for AI Apps](#4-cicd-pipelines-for-ai-apps)
5. [Testing AI Applications](#5-testing-ai-applications)
6. [Monitoring & Observability in Production](#6-monitoring--observability-in-production)
7. [Production Checklist](#7-production-checklist)

---

## 1. From Notebook to Production

### The Journey

```
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ Jupyter  │──▶│ Python   │──▶│ Docker   │──▶│ CI/CD    │──▶│Production│
│ Notebook │   │ App      │   │ Container│   │ Pipeline │   │ (Cloud)  │
│          │   │ (FastAPI) │   │          │   │          │   │          │
│ Research │   │ Structure│   │ Package  │   │ Automate │   │ Deploy   │
└──────────┘   └──────────┘   └──────────┘   └──────────┘   └──────────┘
```

### Step 1: Structure Your Project

```
my-ai-app/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI entrypoint
│   ├── config.py             # Settings (env vars, model paths)
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py        # Pydantic request/response models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── llm_service.py    # LLM API calls
│   │   ├── rag_service.py    # RAG pipeline
│   │   └── embedding_service.py
│   ├── routers/
│   │   ├── __init__.py
│   │   ├── chat.py           # /api/chat endpoints
│   │   └── health.py         # /health endpoint
│   └── utils/
│       ├── __init__.py
│       └── prompts.py        # Prompt templates
├── tests/
│   ├── __init__.py
│   ├── conftest.py           # Shared fixtures
│   ├── unit/
│   │   ├── test_prompts.py
│   │   ├── test_chunking.py
│   │   └── test_schemas.py
│   ├── integration/
│   │   ├── test_rag_pipeline.py
│   │   └── test_api_endpoints.py
│   └── eval/
│       ├── test_retrieval_quality.py
│       ├── test_generation_quality.py
│       └── eval_dataset.json
├── Dockerfile
├── docker-compose.yml
├── .github/
│   └── workflows/
│       ├── ci.yml            # Run tests on every PR
│       └── deploy.yml        # Deploy on merge to main
├── pyproject.toml             # Dependencies
├── .env.example              # Environment variables template
└── README.md
```

### Step 2: The FastAPI App

```python
# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.routers import chat, health
from app.config import settings
from app.services.rag_service import RAGService

# Initialize expensive resources once at startup
rag_service = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models and indexes at startup, clean up at shutdown."""
    global rag_service
    rag_service = RAGService(
        vector_db_url=settings.VECTOR_DB_URL,
        embedding_model=settings.EMBEDDING_MODEL,
        llm_model=settings.LLM_MODEL,
    )
    await rag_service.initialize()
    yield  # App runs here
    await rag_service.shutdown()

app = FastAPI(title="My AI App", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"])
app.include_router(health.router)
app.include_router(chat.router, prefix="/api")
```

```python
# app/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """All config comes from environment variables."""
    OPENAI_API_KEY: str
    VECTOR_DB_URL: str = "http://localhost:6333"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_MODEL: str = "gpt-4o-mini"
    LLM_TEMPERATURE: float = 0.1
    LLM_MAX_TOKENS: int = 1024
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 50
    TOP_K: int = 5
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"

settings = Settings()
```

```python
# app/routers/health.py
from fastapi import APIRouter

router = APIRouter()

@router.get("/health")
async def health_check():
    return {"status": "healthy"}

@router.get("/ready")
async def readiness_check():
    """Check that all dependencies are available."""
    # Check vector DB connection
    # Check LLM API reachability
    return {"status": "ready", "vector_db": "connected", "llm": "reachable"}
```

```python
# app/routers/chat.py
from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from app.models.schemas import ChatRequest, ChatResponse
from app.main import rag_service

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    response = await rag_service.query(
        question=request.message,
        conversation_history=request.history,
    )
    return ChatResponse(
        answer=response.answer,
        sources=response.sources,
        tokens_used=response.tokens_used,
    )

@router.post("/chat/stream")
async def chat_stream(request: ChatRequest):
    async def generate():
        async for chunk in rag_service.stream_query(request.message):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"
    return StreamingResponse(generate(), media_type="text/event-stream")
```

---

## 2. Containerizing AI Apps with Docker

### Basic Dockerfile for an AI App

```dockerfile
# Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (for some ML libraries)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files first (Docker layer caching)
COPY pyproject.toml .
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ app/

# Create non-root user (security best practice)
RUN useradd --create-home appuser
USER appuser

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run with uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Docker Compose for Local Development

```yaml
# docker-compose.yml
version: "3.8"

services:
  app:
    build: .
    ports:
      - "8000:8000"
    env_file:
      - .env
    depends_on:
      qdrant:
        condition: service_healthy
    volumes:
      - ./app:/app/app  # Hot reload in dev

  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
    volumes:
      - qdrant_data:/qdrant/storage
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/healthz"]
      interval: 10s
      timeout: 5s
      retries: 3

  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"

volumes:
  qdrant_data:
```

### Docker Best Practices for AI Apps

| Practice | Why | Example |
|----------|-----|---------|
| Multi-stage builds | Smaller image (no build tools in prod) | `FROM python:3.11 AS builder` then `FROM python:3.11-slim` |
| Layer caching | Faster rebuilds when only code changes | Copy `requirements.txt` before `COPY . .` |
| Non-root user | Security — limit container privileges | `RUN useradd appuser && USER appuser` |
| .dockerignore | Don't copy unnecessary files | Ignore `.git`, `__pycache__`, `.env`, `venv` |
| Pin versions | Reproducible builds | `FROM python:3.11.7-slim` not `python:latest` |
| Health checks | K8s/ECS knows when app is ready | `HEALTHCHECK CMD curl -f /health` |
| No secrets in image | Security — use env vars at runtime | Never `COPY .env` into the image |

### Multi-Stage Build (For Apps with ML Model Files)

```dockerfile
# Stage 1: Build and download model
FROM python:3.11 AS builder
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Download model files during build (cached in layer)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Stage 2: Production image (smaller)
FROM python:3.11-slim
WORKDIR /app
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /root/.cache/torch /root/.cache/torch
COPY app/ app/
RUN useradd --create-home appuser && chown -R appuser /app
USER appuser
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 3. Deploying to Production

### Deployment Options Comparison

| Option | Best For | Cost | Complexity | Scale |
|--------|----------|------|------------|-------|
| **Railway / Render** | MVPs, side projects | $5-20/mo | Very Low | Low |
| **AWS ECS / Fargate** | Containers without K8s | $50-500/mo | Medium | Medium-High |
| **AWS EKS / GKE** | Large-scale, multi-service | $200+/mo | High | Very High |
| **GCP Cloud Run** | Serverless containers | Pay-per-use | Low | Medium-High |
| **Azure Container Apps** | Serverless containers | Pay-per-use | Low | Medium-High |
| **Fly.io** | Global edge deployment | $10-100/mo | Low | Medium |
| **Modal** | ML/AI-specific serverless | Pay-per-use | Low | High |
| **Replicate** | Model serving only | Pay-per-use | Very Low | High |

### Option A: Deploy to GCP Cloud Run (Recommended for Most AI Apps)

```bash
# 1. Build and push Docker image to Google Container Registry
gcloud builds submit --tag gcr.io/MY_PROJECT/my-ai-app

# 2. Deploy to Cloud Run
gcloud run deploy my-ai-app \
  --image gcr.io/MY_PROJECT/my-ai-app \
  --platform managed \
  --region us-central1 \
  --memory 2Gi \
  --cpu 2 \
  --min-instances 1 \          # Keep warm (avoid cold starts)
  --max-instances 10 \         # Auto-scale up to 10
  --set-env-vars "OPENAI_API_KEY=sk-..." \
  --allow-unauthenticated

# 3. Get the URL
gcloud run services describe my-ai-app --format='value(status.url)'
# → https://my-ai-app-xxxx-uc.a.run.app
```

**Cost estimate:** ~$30-100/month for a RAG app with moderate traffic.

### Option B: Deploy to AWS ECS with Fargate

```bash
# 1. Create ECR repository
aws ecr create-repository --repository-name my-ai-app

# 2. Build and push
aws ecr get-login-password | docker login --username AWS --password-stdin ACCOUNT.dkr.ecr.REGION.amazonaws.com
docker build -t my-ai-app .
docker tag my-ai-app:latest ACCOUNT.dkr.ecr.REGION.amazonaws.com/my-ai-app:latest
docker push ACCOUNT.dkr.ecr.REGION.amazonaws.com/my-ai-app:latest

# 3. Create ECS service (via console, CDK, or Terraform)
# Task definition → Service → Load Balancer → Domain
```

### Option C: Deploy to Kubernetes (EKS/GKE)

```yaml
# k8s/deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: ai-app
  labels:
    app: ai-app
spec:
  replicas: 3
  selector:
    matchLabels:
      app: ai-app
  template:
    metadata:
      labels:
        app: ai-app
    spec:
      containers:
        - name: ai-app
          image: gcr.io/my-project/my-ai-app:v1.2.0
          ports:
            - containerPort: 8000
          env:
            - name: OPENAI_API_KEY
              valueFrom:
                secretKeyRef:
                  name: ai-app-secrets
                  key: openai-api-key
            - name: VECTOR_DB_URL
              value: "http://qdrant:6333"
          resources:
            requests:
              cpu: "500m"
              memory: "1Gi"
            limits:
              cpu: "2"
              memory: "4Gi"
          readinessProbe:
            httpGet:
              path: /ready
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 5
          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 15
            periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata:
  name: ai-app
spec:
  selector:
    app: ai-app
  ports:
    - port: 80
      targetPort: 8000
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: ai-app-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: ai-app
  minReplicas: 2
  maxReplicas: 20
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 70
```

### Secrets Management

**Never hardcode API keys.** Here's how to manage them by platform:

| Platform | How to Set Secrets |
|----------|-------------------|
| Cloud Run | `gcloud run deploy --set-env-vars` or Secret Manager |
| ECS | AWS Secrets Manager → Task Definition |
| Kubernetes | `kubectl create secret` or External Secrets Operator |
| Railway/Render | Dashboard → Environment Variables |
| Local | `.env` file (never commit to git) |

```bash
# Kubernetes: Create secret
kubectl create secret generic ai-app-secrets \
  --from-literal=openai-api-key=sk-xxx \
  --from-literal=qdrant-api-key=xxx

# GCP Secret Manager
echo -n "sk-xxx" | gcloud secrets create openai-api-key --data-file=-
```

---

## 4. CI/CD Pipelines for AI Apps

### What's Different About AI CI/CD?

Traditional apps test: "Does the code work?"
AI apps also test: "Does the model perform well enough?"

```
TRADITIONAL CI/CD:
  Code → Lint → Unit Tests → Integration Tests → Deploy

AI CI/CD (additional steps):
  Code → Lint → Unit Tests → Integration Tests
    → Eval Tests (retrieval quality, LLM output quality)
    → Cost Check (estimated API costs within budget?)
    → Performance Gate (latency within bounds?)
    → Deploy (canary → full)
    → Post-deploy monitoring (drift, quality)
```

### GitHub Actions: Complete CI Pipeline

```yaml
# .github/workflows/ci.yml
name: CI Pipeline

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

env:
  PYTHON_VERSION: "3.11"

jobs:
  lint:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - name: Install linters
        run: pip install ruff mypy
      - name: Run ruff (linting + formatting)
        run: ruff check . && ruff format --check .
      - name: Run mypy (type checking)
        run: mypy app/ --ignore-missing-imports

  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - name: Install dependencies
        run: pip install -r requirements.txt -r requirements-test.txt
      - name: Run unit tests
        run: pytest tests/unit/ -v --tb=short --junitxml=results/unit.xml
      - name: Upload test results
        uses: actions/upload-artifact@v4
        with:
          name: unit-test-results
          path: results/unit.xml

  integration-tests:
    runs-on: ubuntu-latest
    services:
      qdrant:
        image: qdrant/qdrant:latest
        ports:
          - 6333:6333
      redis:
        image: redis:7-alpine
        ports:
          - 6379:6379
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - name: Install dependencies
        run: pip install -r requirements.txt -r requirements-test.txt
      - name: Run integration tests
        env:
          VECTOR_DB_URL: http://localhost:6333
          REDIS_URL: redis://localhost:6379
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: pytest tests/integration/ -v --tb=short

  eval-tests:
    runs-on: ubuntu-latest
    # Only run eval on PRs (expensive, uses real LLM API)
    if: github.event_name == 'pull_request'
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - name: Install dependencies
        run: pip install -r requirements.txt -r requirements-test.txt
      - name: Run evaluation tests
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
        run: pytest tests/eval/ -v --tb=short
      - name: Check eval thresholds
        run: python scripts/check_eval_results.py
        # Fails if: retrieval_relevance < 0.7 or hallucination_rate > 0.1

  docker-build:
    runs-on: ubuntu-latest
    needs: [lint, unit-tests]
    steps:
      - uses: actions/checkout@v4
      - name: Build Docker image
        run: docker build -t my-ai-app:${{ github.sha }} .
      - name: Run container smoke test
        run: |
          docker run -d --name test-app -p 8000:8000 \
            -e OPENAI_API_KEY=test \
            -e VECTOR_DB_URL=http://localhost:6333 \
            my-ai-app:${{ github.sha }}
          sleep 5
          curl -f http://localhost:8000/health || exit 1
          docker stop test-app
```

### GitHub Actions: Deploy Pipeline

```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    # Only deploy if CI passed
    needs: []
    steps:
      - uses: actions/checkout@v4

      - name: Authenticate to GCP
        uses: google-github-actions/auth@v2
        with:
          credentials_json: ${{ secrets.GCP_SA_KEY }}

      - name: Set up Cloud SDK
        uses: google-github-actions/setup-gcloud@v2

      - name: Build and push to Container Registry
        run: |
          gcloud builds submit \
            --tag gcr.io/${{ secrets.GCP_PROJECT }}/my-ai-app:${{ github.sha }}

      - name: Deploy to Cloud Run (Canary - 10%)
        run: |
          gcloud run deploy my-ai-app \
            --image gcr.io/${{ secrets.GCP_PROJECT }}/my-ai-app:${{ github.sha }} \
            --region us-central1 \
            --tag canary \
            --no-traffic

          # Send 10% of traffic to canary
          gcloud run services update-traffic my-ai-app \
            --region us-central1 \
            --to-tags canary=10

      - name: Wait and check canary health (5 min)
        run: |
          sleep 300
          # Check error rate on canary
          python scripts/check_canary_health.py \
            --service my-ai-app \
            --tag canary \
            --max-error-rate 0.05

      - name: Promote canary to 100%
        run: |
          gcloud run services update-traffic my-ai-app \
            --region us-central1 \
            --to-latest

      - name: Notify team
        uses: slackapi/slack-github-action@v1
        with:
          payload: |
            {"text": "Deployed my-ai-app ${{ github.sha }} to production"}
```

### CI/CD for Model Updates (Not Just Code)

When you retrain or swap models, CI/CD needs additional gates:

```yaml
# .github/workflows/model-update.yml
name: Model Update Pipeline

on:
  workflow_dispatch:
    inputs:
      model_version:
        description: "New model version to deploy"
        required: true
      model_type:
        description: "Type of model update"
        required: true
        type: choice
        options:
          - embedding_model
          - llm_model
          - reranker_model

jobs:
  validate-model:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Run eval suite with new model
        env:
          NEW_MODEL: ${{ inputs.model_version }}
        run: |
          # Run the full eval suite with the new model
          python scripts/run_eval.py \
            --model-type ${{ inputs.model_type }} \
            --model-version ${{ inputs.model_version }} \
            --output results/eval_new.json

      - name: Compare with current production model
        run: |
          python scripts/compare_eval.py \
            --current results/eval_current.json \
            --new results/eval_new.json \
            --fail-if-regression

      - name: Check cost impact
        run: |
          python scripts/estimate_cost.py \
            --model ${{ inputs.model_version }} \
            --traffic-estimate 50000

  deploy-model:
    needs: validate-model
    runs-on: ubuntu-latest
    steps:
      - name: Update model config
        run: |
          # Update the model version in config/environment
          gcloud run services update my-ai-app \
            --update-env-vars "${{ inputs.model_type }}=${{ inputs.model_version }}"

      - name: Verify deployment
        run: |
          python scripts/smoke_test.py --url https://my-ai-app.run.app/api/chat
```

---

## 5. Testing AI Applications

### The Testing Pyramid for AI Apps

```
                    ┌───────────────┐
                    │   E2E Tests   │  ← Few, expensive, slow
                    │  (full flow)  │     Test: "User asks question → gets good answer"
                    ├───────────────┤
                    │  Eval Tests   │  ← AI-specific quality tests
                    │(retrieval,    │     Test: "Is the retrieval relevant? Is the answer faithful?"
                    │ generation)   │
                    ├───────────────┤
                    │ Integration   │  ← Test component interactions
                    │  Tests        │     Test: "API → RAG pipeline → returns response"
                    │               │
                    ├───────────────┤
                    │  Unit Tests   │  ← Many, fast, cheap
                    │               │     Test: "Chunker splits correctly" "Prompt formats correctly"
                    └───────────────┘
```

### 5.1 Unit Tests (Fast, Run on Every Commit)

Unit tests verify individual functions work correctly — **no LLM API calls needed**.

```python
# tests/unit/test_chunking.py
import pytest
from app.services.chunking import chunk_text, RecursiveChunker

class TestChunking:
    def test_chunk_respects_max_size(self):
        text = "Hello world. " * 1000  # Long text
        chunks = chunk_text(text, chunk_size=512, overlap=50)
        for chunk in chunks:
            assert len(chunk.split()) <= 520  # Some tolerance

    def test_chunk_overlap(self):
        text = "Sentence one. Sentence two. Sentence three. Sentence four."
        chunks = chunk_text(text, chunk_size=20, overlap=10)
        # Verify overlap exists between consecutive chunks
        for i in range(len(chunks) - 1):
            overlap = set(chunks[i].split()) & set(chunks[i + 1].split())
            assert len(overlap) > 0

    def test_empty_text_returns_empty(self):
        chunks = chunk_text("", chunk_size=512, overlap=50)
        assert chunks == []

    def test_short_text_returns_single_chunk(self):
        chunks = chunk_text("Short text.", chunk_size=512, overlap=50)
        assert len(chunks) == 1
        assert chunks[0] == "Short text."
```

```python
# tests/unit/test_prompts.py
from app.utils.prompts import build_rag_prompt, format_sources

class TestPromptBuilding:
    def test_prompt_includes_context(self):
        prompt = build_rag_prompt(
            question="What is our refund policy?",
            context_chunks=["Refunds are processed within 30 days."],
            history=[]
        )
        assert "refund" in prompt.lower()
        assert "30 days" in prompt

    def test_prompt_includes_history(self):
        prompt = build_rag_prompt(
            question="Tell me more",
            context_chunks=["Some context"],
            history=[
                {"role": "user", "content": "What is RAG?"},
                {"role": "assistant", "content": "RAG is Retrieval-Augmented Generation."}
            ]
        )
        assert "What is RAG?" in prompt

    def test_format_sources_creates_citations(self):
        sources = format_sources([
            {"title": "Policy Guide", "url": "/docs/policy", "relevance": 0.95},
            {"title": "FAQ", "url": "/docs/faq", "relevance": 0.82},
        ])
        assert "Policy Guide" in sources
        assert "/docs/policy" in sources

    def test_prompt_has_max_context_limit(self):
        """Ensure we don't exceed context window."""
        huge_context = ["x" * 10000] * 100
        prompt = build_rag_prompt("question", huge_context, [])
        # Should truncate to fit within token limit
        assert len(prompt) < 100000  # Rough character limit
```

```python
# tests/unit/test_schemas.py
import pytest
from pydantic import ValidationError
from app.models.schemas import ChatRequest, ChatResponse

class TestSchemas:
    def test_valid_chat_request(self):
        req = ChatRequest(message="Hello", history=[])
        assert req.message == "Hello"

    def test_empty_message_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(message="", history=[])

    def test_message_too_long_rejected(self):
        with pytest.raises(ValidationError):
            ChatRequest(message="x" * 50001, history=[])

    def test_chat_response_includes_sources(self):
        resp = ChatResponse(
            answer="The refund policy is 30 days.",
            sources=[{"title": "Policy", "url": "/docs"}],
            tokens_used=150
        )
        assert len(resp.sources) == 1
```

### 5.2 Integration Tests (Test Component Interactions)

Integration tests verify that components work together. Use **real services** (vector DB, Redis) but **mock expensive LLM calls**.

```python
# tests/integration/test_rag_pipeline.py
import pytest
from unittest.mock import AsyncMock, patch
from app.services.rag_service import RAGService

@pytest.fixture
async def rag_service():
    """Create a RAG service with real vector DB but mocked LLM."""
    service = RAGService(
        vector_db_url="http://localhost:6333",
        embedding_model="text-embedding-3-small",
        llm_model="gpt-4o-mini",
    )
    await service.initialize()
    # Seed test data
    await service.ingest_documents([
        {"text": "Our refund policy allows returns within 30 days.", "source": "policy.md"},
        {"text": "Shipping takes 3-5 business days.", "source": "shipping.md"},
        {"text": "Customer support hours are 9am-5pm EST.", "source": "support.md"},
    ])
    yield service
    await service.shutdown()

class TestRAGPipeline:
    @patch("app.services.llm_service.call_llm")
    async def test_retrieval_returns_relevant_chunks(self, mock_llm, rag_service):
        """Test that the retriever finds the right documents."""
        mock_llm.return_value = "Mocked response"

        result = await rag_service.query("What is the refund policy?")

        # Check that retrieved chunks are relevant
        sources = [s["source"] for s in result.sources]
        assert "policy.md" in sources

    @patch("app.services.llm_service.call_llm")
    async def test_irrelevant_query_returns_no_good_matches(self, mock_llm, rag_service):
        mock_llm.return_value = "I don't have information about that."

        result = await rag_service.query("What is the weather on Mars?")

        # Should have low relevance scores
        assert all(s["relevance"] < 0.5 for s in result.sources)
```

```python
# tests/integration/test_api_endpoints.py
import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app

@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c

class TestAPIEndpoints:
    async def test_health_endpoint(self, client):
        response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "healthy"

    async def test_chat_endpoint_returns_answer(self, client):
        response = await client.post("/api/chat", json={
            "message": "What is the refund policy?",
            "history": []
        })
        assert response.status_code == 200
        data = response.json()
        assert "answer" in data
        assert "sources" in data
        assert isinstance(data["tokens_used"], int)

    async def test_chat_rejects_empty_message(self, client):
        response = await client.post("/api/chat", json={
            "message": "",
            "history": []
        })
        assert response.status_code == 422  # Validation error

    async def test_chat_stream_returns_sse(self, client):
        response = await client.post("/api/chat/stream", json={
            "message": "Hello",
            "history": []
        })
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
```

### 5.3 Eval Tests (AI-Specific Quality Tests)

Eval tests measure **retrieval and generation quality** using real LLM calls. These are expensive — run them on PRs, not every commit.

```python
# tests/eval/test_retrieval_quality.py
import pytest
import json
from app.services.rag_service import RAGService

# Load evaluation dataset
with open("tests/eval/eval_dataset.json") as f:
    EVAL_DATA = json.load(f)

@pytest.fixture(scope="module")
async def rag_service():
    service = RAGService(...)
    await service.initialize()
    yield service

class TestRetrievalQuality:
    @pytest.mark.parametrize("item", EVAL_DATA["retrieval_tests"])
    async def test_retrieval_finds_relevant_doc(self, rag_service, item):
        """For each test question, verify the expected document is retrieved."""
        results = await rag_service.retrieve(item["question"], top_k=5)
        retrieved_sources = [r["source"] for r in results]

        # The expected source must be in the top 5
        assert item["expected_source"] in retrieved_sources, \
            f"Expected '{item['expected_source']}' in results for: {item['question']}"

    async def test_overall_recall_at_5(self, rag_service):
        """Measure recall@5 across the entire eval set."""
        hits = 0
        total = len(EVAL_DATA["retrieval_tests"])

        for item in EVAL_DATA["retrieval_tests"]:
            results = await rag_service.retrieve(item["question"], top_k=5)
            retrieved_sources = [r["source"] for r in results]
            if item["expected_source"] in retrieved_sources:
                hits += 1

        recall = hits / total
        assert recall >= 0.80, f"Recall@5 is {recall:.2f}, expected >= 0.80"

    async def test_mrr(self, rag_service):
        """Mean Reciprocal Rank — is the best result ranked first?"""
        reciprocal_ranks = []

        for item in EVAL_DATA["retrieval_tests"]:
            results = await rag_service.retrieve(item["question"], top_k=5)
            for rank, r in enumerate(results, 1):
                if r["source"] == item["expected_source"]:
                    reciprocal_ranks.append(1.0 / rank)
                    break
            else:
                reciprocal_ranks.append(0.0)

        mrr = sum(reciprocal_ranks) / len(reciprocal_ranks)
        assert mrr >= 0.65, f"MRR is {mrr:.2f}, expected >= 0.65"
```

```python
# tests/eval/test_generation_quality.py
import pytest
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision

class TestGenerationQuality:
    async def test_faithfulness(self, rag_service):
        """Test that answers are grounded in the retrieved context (no hallucinations)."""
        results = []
        for item in EVAL_DATA["generation_tests"]:
            response = await rag_service.query(item["question"])
            results.append({
                "question": item["question"],
                "answer": response.answer,
                "contexts": [s["text"] for s in response.sources],
            })

        # Use RAGAS to evaluate
        scores = evaluate(results, metrics=[faithfulness])
        avg_faithfulness = scores["faithfulness"].mean()
        assert avg_faithfulness >= 0.85, \
            f"Faithfulness is {avg_faithfulness:.2f}, expected >= 0.85"

    async def test_answer_relevancy(self, rag_service):
        """Test that answers actually address the question asked."""
        results = []
        for item in EVAL_DATA["generation_tests"]:
            response = await rag_service.query(item["question"])
            results.append({
                "question": item["question"],
                "answer": response.answer,
                "contexts": [s["text"] for s in response.sources],
            })

        scores = evaluate(results, metrics=[answer_relevancy])
        avg_relevancy = scores["answer_relevancy"].mean()
        assert avg_relevancy >= 0.80, \
            f"Answer relevancy is {avg_relevancy:.2f}, expected >= 0.80"

    async def test_no_hallucination_on_unknown_question(self, rag_service):
        """When asked about something not in the knowledge base, should say 'I don't know'."""
        response = await rag_service.query("What is the GDP of Jupiter?")
        refusal_phrases = ["don't have", "no information", "not sure", "cannot find"]
        assert any(phrase in response.answer.lower() for phrase in refusal_phrases), \
            f"Expected refusal, got: {response.answer}"
```

### 5.4 The Eval Dataset

```json
// tests/eval/eval_dataset.json
{
  "retrieval_tests": [
    {
      "question": "What is the refund policy?",
      "expected_source": "policies/refund-policy.md",
      "expected_keywords": ["30 days", "refund"]
    },
    {
      "question": "How do I reset my password?",
      "expected_source": "support/account-help.md",
      "expected_keywords": ["password", "reset", "email"]
    },
    {
      "question": "What programming languages does the API support?",
      "expected_source": "docs/api-reference.md",
      "expected_keywords": ["Python", "JavaScript", "SDK"]
    }
  ],
  "generation_tests": [
    {
      "question": "What is the refund policy?",
      "expected_answer_contains": ["30 days"],
      "expected_answer_not_contains": ["60 days", "no refunds"]
    },
    {
      "question": "How do I contact support?",
      "expected_answer_contains": ["email", "support@"],
      "expected_answer_not_contains": []
    }
  ]
}
```

**How to build your eval dataset:**
1. Start with 20-50 real user questions (or make them up based on your docs)
2. Manually label the correct source document for each
3. Write expected keywords/phrases for answers
4. Grow the dataset over time — add edge cases as you find them
5. **Golden rule: 50 well-curated examples > 500 sloppy ones**

### 5.5 Load Testing

```python
# scripts/load_test.py
# Using locust for load testing
from locust import HttpUser, task, between

class AIAppUser(HttpUser):
    wait_time = between(1, 3)

    @task(10)
    def chat(self):
        self.client.post("/api/chat", json={
            "message": "What is the refund policy?",
            "history": []
        })

    @task(1)
    def health_check(self):
        self.client.get("/health")
```

```bash
# Run load test: 50 concurrent users for 2 minutes
pip install locust
locust -f scripts/load_test.py --host http://localhost:8000 \
  --users 50 --spawn-rate 5 --run-time 2m --headless
```

**Key metrics to measure:**
| Metric | Target | What It Tells You |
|--------|--------|-------------------|
| P50 latency | <2s | Typical user experience |
| P99 latency | <5s | Worst-case experience |
| Throughput | >50 QPS | Can you handle the load? |
| Error rate | <1% | Is the system stable under load? |
| CPU/Memory | <80% | Do you have headroom? |

### 5.6 Shadow Testing (Before Swapping Models)

When switching models (e.g., GPT-4o-mini → Claude Haiku), run both in parallel:

```python
# Shadow test: run new model alongside production, compare results
async def shadow_test(question: str):
    # Run both models
    prod_response = await llm_call(model="gpt-4o-mini", prompt=question)
    shadow_response = await llm_call(model="claude-haiku", prompt=question)

    # Log both for comparison (don't serve shadow to user)
    log_comparison({
        "question": question,
        "prod_answer": prod_response,
        "shadow_answer": shadow_response,
        "prod_latency": prod_response.latency,
        "shadow_latency": shadow_response.latency,
        "prod_tokens": prod_response.tokens,
        "shadow_tokens": shadow_response.tokens,
    })

    # Serve only production response to user
    return prod_response
```

### 5.7 Testing Checklist

```
BEFORE EVERY PR:
  ✅ Unit tests pass (pytest tests/unit/)
  ✅ Linting passes (ruff check .)
  ✅ Type checking passes (mypy app/)
  ✅ Docker builds successfully

BEFORE MERGING TO MAIN:
  ✅ Integration tests pass
  ✅ Eval tests pass (retrieval recall >= 0.80, faithfulness >= 0.85)
  ✅ No new security vulnerabilities (pip audit)

BEFORE MODEL CHANGES:
  ✅ Eval comparison: new model >= current model on all metrics
  ✅ Cost estimate is within budget
  ✅ Latency is within bounds
  ✅ Shadow test results reviewed
```

---

## 6. Monitoring & Observability in Production

### What to Monitor for AI Apps

```
┌─────────────────────────────────────────────────────────────────┐
│                   MONITORING STACK                               │
│                                                                  │
│  INFRASTRUCTURE METRICS (Prometheus + Grafana)                   │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │ CPU /    │  │ Request  │  │ Error    │  │ Pod Count /  │   │
│  │ Memory   │  │ Latency  │  │ Rate     │  │ Restarts     │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
│                                                                  │
│  AI-SPECIFIC METRICS (LangSmith / Custom)                       │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐   │
│  │ Token    │  │ Cache    │  │ Retrieval│  │ User         │   │
│  │ Usage /  │  │ Hit Rate │  │ Relevance│  │ Satisfaction │   │
│  │ Cost     │  │          │  │ Score    │  │ (thumbs)     │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘   │
│                                                                  │
│  LOGGING (structured JSON → centralized log system)             │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Every request logged:                                     │   │
│  │ {timestamp, request_id, user_id, question, model,        │   │
│  │  tokens_in, tokens_out, latency_ms, retrieval_count,     │   │
│  │  cache_hit, cost_usd, sources[], status_code}            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
│  ALERTING (PagerDuty / Slack)                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Alert if:                                                 │   │
│  │ - Error rate > 5% for 5 minutes                          │   │
│  │ - P99 latency > 10s for 5 minutes                        │   │
│  │ - LLM API errors > 10 in 1 minute                        │   │
│  │ - Daily cost exceeds budget by 20%                        │   │
│  │ - Retrieval relevance drops below 0.6 (sampled)          │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Structured Logging Example

```python
# app/middleware/logging.py
import structlog
import time
from fastapi import Request

logger = structlog.get_logger()

async def log_request(request: Request, call_next):
    start = time.time()
    request_id = str(uuid4())

    response = await call_next(request)

    duration = time.time() - start
    logger.info(
        "request_completed",
        request_id=request_id,
        method=request.method,
        path=request.url.path,
        status_code=response.status_code,
        duration_ms=round(duration * 1000, 2),
        # AI-specific fields added by the RAG service:
        # tokens_in, tokens_out, model, cache_hit, cost_usd
    )
    return response
```

### Cost Tracking Dashboard

```python
# Track cost per request
def calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = {
        "gpt-4o":        {"input": 2.50 / 1_000_000, "output": 10.00 / 1_000_000},
        "gpt-4o-mini":   {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
        "claude-sonnet": {"input": 3.00 / 1_000_000, "output": 15.00 / 1_000_000},
        "claude-haiku":  {"input": 0.80 / 1_000_000, "output": 4.00 / 1_000_000},
    }
    p = pricing.get(model, pricing["gpt-4o-mini"])
    return input_tokens * p["input"] + output_tokens * p["output"]
```

---

## 7. Production Checklist

### Pre-Launch Checklist

```
APPLICATION
  □ Health check endpoint (/health) returns 200
  □ Readiness probe checks all dependencies (/ready)
  □ Environment variables for all secrets (no hardcoded keys)
  □ Input validation on all endpoints (Pydantic)
  □ Rate limiting configured
  □ CORS configured for production domains only
  □ Request timeout set (30s default, longer for LLM calls)
  □ Graceful shutdown handles in-flight requests

DOCKER
  □ Non-root user in Dockerfile
  □ .dockerignore includes .env, .git, __pycache__, venv
  □ Pinned base image version
  □ Health check in Dockerfile
  □ Multi-stage build (if model files are large)

CI/CD
  □ Lint + unit tests run on every PR
  □ Integration tests run on every PR
  □ Eval tests run on PRs (with pass/fail gates)
  □ Docker build verified in CI
  □ Automated deployment on merge to main
  □ Canary/staged rollout (not all-at-once)
  □ Rollback plan documented and tested

TESTING
  □ Unit tests for: chunking, prompts, schemas, utils
  □ Integration tests for: API endpoints, RAG pipeline
  □ Eval tests for: retrieval quality, generation quality
  □ Eval dataset with 50+ curated question-answer pairs
  □ Load test confirms P99 < target at expected QPS

MONITORING
  □ Structured logging (JSON) to centralized system
  □ Metrics: latency, error rate, token usage, cost
  □ LLM tracing (LangSmith or equivalent)
  □ Alerting rules configured
  □ Cost dashboard with daily/weekly budget alerts
  □ User feedback collection (thumbs up/down)

SECURITY
  □ API keys stored in secret manager (not env files)
  □ PII detection on inputs and outputs
  □ Prompt injection defense
  □ No sensitive data in logs
  □ HTTPS enforced
  □ Authentication on all non-health endpoints

RELIABILITY
  □ Auto-scaling configured (min 2, max based on budget)
  □ LLM API fallback (e.g., OpenAI → Anthropic)
  □ Retry logic with exponential backoff for API calls
  □ Circuit breaker for external dependencies
  □ Database connection pooling
  □ Cache layer (Redis) for frequent queries
```

### The Deployment Maturity Ladder

```
Level 0: YOLO
  "I run it on my laptop and share the URL with ngrok"
  → No CI/CD, no tests, no monitoring

Level 1: Containerized
  "It runs in Docker on a cloud VM"
  → Dockerfile, basic health check, manual deployment

Level 2: Automated
  "CI/CD deploys on push to main"
  → GitHub Actions, unit tests, auto-deploy to Cloud Run/ECS

Level 3: Production-Ready
  "We have tests, monitoring, and canary deploys"
  → Eval tests, LLM tracing, structured logging, staged rollouts

Level 4: Enterprise
  "We handle failures gracefully and optimize cost"
  → A/B testing, model routing, cost optimization, disaster recovery

Most AI apps should aim for Level 2-3. Level 4 when you're at scale.
```

---

## Quick Reference: Commands You'll Use Daily

```bash
# LOCAL DEVELOPMENT
docker compose up -d                           # Start local stack
pytest tests/unit/ -v                          # Run unit tests
pytest tests/integration/ -v                   # Run integration tests
pytest tests/eval/ -v                          # Run eval tests (uses API)
ruff check . && ruff format --check .          # Lint and format check

# DOCKER
docker build -t my-ai-app .                    # Build image
docker run -p 8000:8000 --env-file .env my-ai-app  # Run container
docker compose logs -f app                     # View logs

# DEPLOYMENT (GCP Cloud Run)
gcloud builds submit --tag gcr.io/PROJECT/my-ai-app
gcloud run deploy my-ai-app --image gcr.io/PROJECT/my-ai-app

# DEPLOYMENT (Kubernetes)
kubectl apply -f k8s/
kubectl rollout status deployment/ai-app
kubectl logs -f deployment/ai-app

# MONITORING
curl http://localhost:8000/health              # Health check
curl http://localhost:8000/ready               # Readiness check
locust -f scripts/load_test.py                 # Load test
```
