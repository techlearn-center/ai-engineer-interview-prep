# AI System Design Practice Scenarios

Five detailed, interview-style system design walkthroughs. Each scenario follows
the same rigorous structure you would use in a real interview: clarify, scope,
design, deep-dive, and discuss trade-offs.

---

## Scenario 1: Production RAG Chatbot for Enterprise Knowledge Base

### Interview Prompt

> "Design an AI-powered chatbot that lets employees ask natural-language
> questions about internal company knowledge. The system must retrieve answers
> from ~500K documents spread across Confluence, Slack, and Google Drive. It
> should cite its sources, support multi-turn conversations, and respect
> document-level access permissions so users only see information they are
> authorized to view. Expect around 5,000 daily active users."

### Clarifying Questions to Ask

1. What is the acceptable end-to-end latency for a response? (Target: < 5s for
   first token, streaming thereafter.)
2. Are documents mostly English, or do we need multilingual support?
3. How frequently do documents change? (Drives re-indexing cadence.)
4. Is there an existing identity provider (Okta, Azure AD) we integrate with
   for ACLs?
5. Do we need to support structured data (tables, spreadsheets) or only prose?
6. What is the sensitivity classification? Can we use a third-party LLM API, or
   must everything stay on-prem?
7. How many concurrent conversations do we expect at peak?
8. Should the system be able to say "I don't know" rather than hallucinate?

### Requirements Summary

| Category | Requirement |
|----------|------------|
| Functional | Natural-language Q&A over 500K docs |
| Functional | Source citations with links |
| Functional | Multi-turn conversation with context |
| Functional | ACL-aware retrieval (per-user permissions) |
| Non-functional | < 5s time-to-first-token |
| Non-functional | 5,000 DAU, ~50K queries/day |
| Non-functional | 99.9% availability |
| Non-functional | Documents re-indexed within 1 hour of change |
| Scale | 500K docs, avg 2 pages each = ~1M chunks |

### High-Level Architecture

```
+------------------+       +------------------+       +-------------------+
|   Data Sources   |       |   Ingestion      |       |   Vector Store    |
|                  |       |   Pipeline        |       |                   |
|  Confluence API -+------>|  Crawl/Extract   -+----->|  Pinecone /       |
|  Slack Export   -+------>|  Chunk           -+----->|  Weaviate /       |
|  Google Drive   -+------>|  Embed           -+----->|  pgvector         |
|                  |       |  Store ACL meta   |       |  (with ACL meta)  |
+------------------+       +------------------+       +-------------------+
                                                             |
                                                             | query
                                                             v
+------------------+       +------------------+       +-------------------+
|   Frontend       |       |   API Gateway    |       |   RAG Service     |
|                  |       |                  |       |                   |
|  React Chat UI  -+------>|  Auth (JWT/OIDC) -+----->|  Query Rewrite    |
|  Streaming SSE  <+-------+  Rate Limiting   <+------+  Retrieve + Filter|
|                  |       |  Session Mgmt    |       |  Rerank           |
+------------------+       +------------------+       |  Generate (LLM)   |
                                                       |  Citation Map     |
                                                       +-------------------+
                                                             |
                                                             v
                                                       +-------------------+
                                                       |   LLM Provider    |
                                                       |                   |
                                                       |  GPT-4 / Claude   |
                                                       |  (or self-hosted  |
                                                       |   Llama 3)        |
                                                       +-------------------+

Supporting Services:
+------------------+   +------------------+   +-------------------+
| Conversation     |   | ACL Sync         |   | Observability     |
| Store (Redis +   |   | Service          |   | (LangSmith /      |
| PostgreSQL)      |   | (IdP -> ACL DB)  |   |  Datadog + custom)|
+------------------+   +------------------+   +-------------------+
```

### Detailed Component Design

**1. Ingestion Pipeline**

- Built with Apache Airflow (or Prefect) DAGs, one per source.
- Confluence connector uses the REST API; pages are extracted as Markdown.
- Slack connector pulls messages + threads from the Conversations API.
- Google Drive connector uses the Drive v3 API with watch notifications for
  near-real-time updates.
- Each document is chunked using a recursive character splitter (LangChain
  `RecursiveCharacterTextSplitter`, chunk size 512 tokens, overlap 64 tokens).
- Chunks are embedded with `text-embedding-3-small` (1536 dims) or a
  self-hosted `e5-large-v2` model for on-prem deployments.
- Each chunk record stores: `chunk_id`, `doc_id`, `source_url`, `text`,
  `embedding`, `acl_groups[]`, `last_updated`, `doc_title`.
- ACL groups are extracted at crawl time from each source's permission model
  and stored alongside the vector.
- A deduplication step hashes chunk content (SHA-256) to avoid re-embedding
  unchanged documents.

**2. ACL Sync Service**

- A nightly batch job (with incremental change-feed processing) pulls group
  memberships from the identity provider (e.g., Azure AD groups via
  Microsoft Graph API).
- Maintains a mapping table: `user_id -> [group_ids]`.
- At query time, the user's groups are looked up and passed as a metadata
  filter to the vector store (`WHERE acl_groups OVERLAP user_groups`).

**3. RAG Service (Core Query Path)**

- Step 1 -- Query Rewrite: Use the conversation history (last 5 turns) to
  rewrite the current question into a standalone query. Example: user says
  "What about their pricing?" -> rewritten to "What is Vendor X's pricing
  model for the Enterprise tier?"
- Step 2 -- Retrieval: Embed the rewritten query, search the vector store
  with a metadata filter on the user's ACL groups. Retrieve top-50 candidates.
- Step 3 -- Reranking: Pass the 50 candidates through a cross-encoder reranker
  (e.g., Cohere Rerank or `bge-reranker-v2-m3`). Keep top-8.
- Step 4 -- Generation: Send the top-8 chunks plus conversation history to
  the LLM with a system prompt that requires inline citations like [1], [2].
- Step 5 -- Citation Mapping: Post-process the LLM output to convert citation
  markers into clickable links with doc title and source URL.
- Step 6 -- Stream the response token-by-token via Server-Sent Events (SSE).

**4. Conversation Store**

- Redis stores active conversation context (TTL 30 minutes of inactivity).
- PostgreSQL stores full conversation history for analytics and fine-tuning
  data collection.
- Each conversation has a `conversation_id`, `user_id`, list of
  `(role, content, citations[], timestamp)` turns.

**5. Frontend**

- React chat interface with streaming display.
- Each message shows inline citations as numbered footnotes; clicking a
  footnote opens the source document in a new tab.
- Thumbs up/down feedback buttons on each response for RLHF data collection.

### Data Flow Walkthrough

```
User types: "What is our parental leave policy for US employees?"

1. Frontend sends POST /api/chat with { conversation_id, message }
2. API Gateway validates JWT, extracts user_id, looks up user_groups
3. RAG Service receives request with user_groups = ["all-employees", "us-hr"]
4. Query Rewrite: no prior context -> query stays as-is
5. Embed query -> [0.012, -0.034, ...] (1536 dims)
6. Vector search: top-50 chunks WHERE acl_groups overlaps user_groups
7. Reranker scores 50 chunks, returns top-8:
   - Chunk from "US Benefits Handbook 2025" (score 0.94)
   - Chunk from "HR FAQ - Leave Policies" (score 0.91)
   - ... 6 more
8. LLM prompt:
   System: "Answer using only the provided context. Cite sources as [1], [2]."
   Context: [8 chunks with source metadata]
   User: "What is our parental leave policy for US employees?"
9. LLM streams: "US employees receive 16 weeks of paid parental leave [1].
   Both birth and non-birth parents are eligible [2]. ..."
10. Citation mapper converts [1] -> { title: "US Benefits Handbook", url: "..." }
11. Response streamed to frontend via SSE
12. Conversation stored in Redis + PostgreSQL
```

### Trade-offs Discussion

| Decision | Chosen | Alternative | Why |
|----------|--------|-------------|-----|
| Vector DB | Pinecone | pgvector | Pinecone has native metadata filtering; pgvector is cheaper but slower at scale with complex filters |
| Embedding model | text-embedding-3-small | e5-large-v2 | Lower latency via API; self-hosted e5 for on-prem |
| Chunk size | 512 tokens | 256 or 1024 | 512 balances precision and context; 256 loses context, 1024 dilutes relevance |
| Reranker | Cross-encoder | None | 15% relevance improvement justifies the 200ms added latency |
| ACL filtering | Pre-retrieval filter | Post-retrieval filter | Pre-filter prevents data leakage even in logs; post-filter risks exposing titles |
| LLM | GPT-4o / Claude 3.5 | Self-hosted Llama 3 | Quality matters most for knowledge Q&A; self-hosted for regulated industries |

### Operations and Monitoring

- **Retrieval quality**: Track `retrieval_precision@k` using human-labeled
  query-relevance pairs. Dashboard in Grafana.
- **Answer quality**: Thumbs up/down ratio per week; human evaluation of
  random 50 answers/week using a rubric (accuracy, citation correctness,
  completeness).
- **Latency**: P50/P95/P99 for each stage (embed, retrieve, rerank, generate).
  Alert if P95 > 8s.
- **ACL correctness**: Weekly audit -- sample 100 queries, verify that no
  returned chunk violates the user's permission boundary.
- **Freshness**: Monitor ingestion lag (time from doc update to re-index).
  Alert if lag > 2 hours.
- **Cost**: Track LLM token usage per user/department. Set per-user daily
  token budgets.
- **Hallucination detection**: Log cases where the LLM generates text not
  grounded in any retrieved chunk (use NLI model for automated detection).

### Scaling Considerations

- **Horizontal scaling**: RAG service is stateless; scale pods behind a load
  balancer. Conversation context is in Redis.
- **Vector store scaling**: Pinecone scales automatically. For pgvector, use
  partitioning by source type and IVFFlat indexes.
- **Embedding throughput**: Batch embed during ingestion. For query-time
  embedding, the single-vector latency (~20ms) is not a bottleneck.
- **LLM throughput**: Use multiple API keys with round-robin, or deploy
  self-hosted models on multiple GPU nodes with vLLM.
- **Caching**: Cache identical queries (same user group set + query hash)
  in Redis with 15-minute TTL. Expect ~20% cache hit rate.
- **Multi-region**: Deploy in multiple regions with separate vector store
  replicas. Route users to nearest region.

---

## Scenario 2: AI-Powered Document Processing Pipeline

### Interview Prompt

> "Design an AI-powered document processing system for an insurance company
> that processes 10,000 claims per day. Each claim arrives as a set of
> unstructured PDF documents (medical reports, invoices, police reports, etc.).
> The system must extract structured data from these PDFs, classify each
> document by type, route it to the correct department, and include
> human-in-the-loop review for low-confidence extractions. Design the full
> pipeline with quality gates."

### Clarifying Questions to Ask

1. How many pages is a typical claim package? (Assume 5-20 pages across
   multiple PDFs.)
2. What are the document types we need to classify? (Medical report, invoice,
   police report, ID document, claim form, correspondence -- ~10 types.)
3. What structured fields need to be extracted? (Patient name, date of
   incident, diagnosis codes, dollar amounts, policy number, etc.)
4. What is the acceptable error rate? (< 2% for critical fields like dollar
   amounts and policy numbers.)
5. What is the end-to-end SLA? (Claims should be triaged within 2 hours of
   receipt.)
6. Are documents scanned images, digital PDFs, or both?
7. Are there existing legacy systems we need to integrate with?
8. What volume of human reviewers is available? (Assume 50 reviewers.)

### Requirements Summary

| Category | Requirement |
|----------|------------|
| Functional | OCR scanned documents with > 99% character accuracy |
| Functional | Classify documents into 10+ types |
| Functional | Extract 20+ structured fields per document type |
| Functional | Route claims to correct department |
| Functional | Human review for extractions below confidence threshold |
| Non-functional | 10,000 claims/day = ~7 claims/minute sustained |
| Non-functional | End-to-end triage within 2 hours |
| Non-functional | < 2% error rate on critical fields post-pipeline |
| Non-functional | Audit trail for every extraction decision |

### High-Level Architecture

```
+-------------------+     +-------------------+     +-------------------+
|   Intake Layer    |     |   Processing      |     |   Extraction      |
|                   |     |   Pipeline        |     |   Engine          |
|  Email Ingest    -+---->|                   |     |                   |
|  Portal Upload   -+---->|  1. PDF Split    -+---->|  4. LLM Extract  |
|  Fax Gateway     -+---->|  2. OCR          -+---->|     (GPT-4V /    |
|  API Endpoint    -+---->|  3. Classify     -+---->|      Claude)     |
|                   |     |                   |     |  5. Validate      |
+-------------------+     +-------------------+     +-------------------+
                                                           |
                                              +------------+------------+
                                              |                         |
                                              v                         v
                                    +-------------------+     +-------------------+
                                    | High Confidence   |     | Low Confidence    |
                                    | (auto-approve)    |     | (human review)    |
                                    |                   |     |                   |
                                    | 6a. Write to      |     | 6b. Queue to      |
                                    |     Claims DB     |     |     Review UI     |
                                    +-------------------+     +-------------------+
                                              |                         |
                                              v                         v
                                    +-------------------+     +-------------------+
                                    |   Routing Engine  |     |   Human Review    |
                                    |                   |     |   Dashboard       |
                                    |  7. Department    |     |                   |
                                    |     assignment    |     |  Correct/Confirm  |
                                    |  8. Priority      |     |  extractions      |
                                    |     scoring       |     |                   |
                                    +-------------------+     +-------------------+
                                              |                         |
                                              +------------+------------+
                                                           |
                                                           v
                                                  +-------------------+
                                                  |   Claims System   |
                                                  |   (downstream)    |
                                                  +-------------------+

Supporting:
+-------------------+   +-------------------+   +-------------------+
| Object Storage    |   | Orchestrator      |   | Model Registry    |
| (S3 / Azure Blob) |   | (Temporal /       |   | & Retraining      |
| - Original PDFs   |   |  Step Functions)  |   | Pipeline          |
| - OCR output      |   |                   |   |                   |
+-------------------+   +-------------------+   +-------------------+
```

### Detailed Component Design

**1. Intake Layer**

- Multi-channel ingestion: email (via SendGrid inbound parse), web portal
  (direct upload), fax (via eFax API), and REST API for partner systems.
- Each submission creates a `Claim` record in PostgreSQL with status
  `RECEIVED` and stores all PDFs in S3 under `s3://claims/{claim_id}/raw/`.
- A message is published to an SQS queue (or Kafka topic) to trigger
  processing.

**2. PDF Splitting and Pre-processing**

- Multi-page PDFs are split into individual pages using `PyMuPDF` (fitz).
- Each page is classified as either digital-text or scanned-image by checking
  if extractable text length > 100 characters.
- Digital pages: text extracted directly with PyMuPDF.
- Scanned pages: sent to OCR engine.
- Page images are rendered at 300 DPI for downstream vision model consumption.

**3. OCR Engine**

- Primary: AWS Textract (for tables, forms, and handwriting).
- Fallback: Google Document AI for complex layouts.
- OCR output stored as structured JSON with bounding boxes and confidence
  scores per word.
- Post-OCR text correction: a small fine-tuned model fixes common OCR errors
  in medical terminology (e.g., "d1agnosis" -> "diagnosis").

**4. Document Classification**

- A fine-tuned `DeBERTa-v3-base` classifier trained on 50K labeled document
  pages.
- Input: first 512 tokens of OCR text + page metadata (page count, position
  in PDF).
- Output: document type + confidence score.
- Documents with confidence < 0.85 are flagged for human classification.
- Model retrained monthly on newly labeled data from human review.

**5. Structured Data Extraction**

- Uses a multimodal LLM (GPT-4o or Claude 3.5 Sonnet) with structured
  output (JSON mode / tool calling).
- Each document type has a specific extraction schema defined in JSON Schema.
- Example schema for Medical Report:
  ```json
  {
    "patient_name": "string",
    "date_of_service": "date",
    "diagnosis_codes": ["string"],
    "procedures": [{"code": "string", "description": "string"}],
    "provider_name": "string",
    "total_charges": "decimal"
  }
  ```
- The LLM receives: (a) the OCR text, (b) the page image, (c) the target
  schema, and (d) few-shot examples for that document type.
- Each extracted field gets a confidence score derived from LLM log-probs
  and cross-validation against OCR bounding-box data.

**6. Validation and Quality Gates**

- Rule-based validators for each field type:
  - Policy numbers match regex `^POL-\d{10}$`.
  - Dates are valid and not in the future.
  - Dollar amounts are positive and within expected ranges.
  - Diagnosis codes exist in ICD-10 lookup table.
- Cross-document consistency checks:
  - Patient name consistent across all documents in a claim.
  - Dates of service fall within policy coverage period.
- Confidence thresholds:
  - All fields > 0.90 confidence AND pass validation -> auto-approve.
  - Any field < 0.90 OR validation failure -> route to human review.
  - Expected auto-approval rate: ~70% of claims.

**7. Human Review Dashboard**

- React-based UI showing the original document image side-by-side with
  extracted fields.
- Low-confidence fields highlighted in yellow; validation failures in red.
- Reviewer can correct fields, confirm extractions, or re-classify documents.
- Average review time target: 3 minutes per claim.
- With 50 reviewers and 30% review rate (3,000 claims/day), each reviewer
  handles 60 claims/day = 7.5 claims/hour -- feasible.

**8. Routing Engine**

- Rule-based + ML hybrid routing:
  - Rule-based: claim type (auto, health, property) -> department.
  - ML-based: priority scoring model predicts claim complexity and urgency.
- Priority factors: dollar amount, injury severity, fraud risk score,
  customer tier.
- Output: department assignment + priority level (P1-P4).

### Data Flow Walkthrough

```
1. Claimant uploads 4 PDF files via web portal
2. Intake layer creates Claim #CLM-20260217-4523, stores PDFs in S3
3. SQS message triggers processing Lambda / ECS task
4. PDF splitter: 4 PDFs -> 12 total pages
5. Page analysis: 8 digital, 4 scanned
6. OCR runs on 4 scanned pages (Textract) -> JSON with bounding boxes
7. Classifier runs on all 12 pages:
   - Pages 1-3: "Medical Report" (0.97)
   - Pages 4-6: "Invoice" (0.93)
   - Pages 7-9: "Claim Form" (0.99)
   - Pages 10-12: "Police Report" (0.88) -- flagged for review
8. Extraction runs per document group:
   - Medical Report: all fields > 0.90, valid -> auto-approve
   - Invoice: total_charges = $14,328.50 (0.95), valid -> auto-approve
   - Claim Form: policy_number POL-0038847291 (0.98), valid -> auto-approve
   - Police Report: incident_date confidence 0.72 -> human review
9. Claim routed to human review queue (one low-confidence field)
10. Reviewer sees police report image + extracted fields
    - Corrects incident_date from "2026-01-15" to "2026-01-05"
    - Confirms all other fields
11. All extractions finalized, written to Claims DB
12. Routing engine: Auto claim, $14K, injury involved -> Auto Claims, P2
13. Claim appears in adjuster's queue within 45 minutes of upload
```

### Trade-offs Discussion

| Decision | Chosen | Alternative | Why |
|----------|--------|-------------|-----|
| OCR | AWS Textract | Tesseract OSS | Textract handles forms/tables natively; Tesseract needs custom training |
| Extraction | Multimodal LLM | Custom NER model | LLM generalizes to new doc types without retraining; NER is cheaper per-doc |
| Orchestration | Temporal | Step Functions | Temporal handles long-running workflows and retries better; Step Functions for simpler AWS-native flows |
| Classification | Fine-tuned DeBERTa | LLM zero-shot | DeBERTa is 100x cheaper per inference and 10x faster; fine-tuned accuracy exceeds zero-shot |
| Human review threshold | 0.90 confidence | 0.80 or 0.95 | 0.90 gives ~70% auto-approval; 0.80 risks errors, 0.95 overwhelms reviewers |
| Storage | S3 + PostgreSQL | Document DB | Relational model fits structured extractions; S3 for raw files |

### Operations and Monitoring

- **Throughput dashboard**: claims processed per hour, current queue depth,
  average processing time per stage.
- **Extraction accuracy**: Weekly sample of 200 auto-approved claims manually
  audited. Track accuracy per field type.
- **Human review metrics**: average review time, reviewer agreement rate
  (double-review 5% of claims), claims per reviewer per hour.
- **SLA tracking**: % of claims triaged within 2-hour SLA. Alert if drops
  below 95%.
- **Model drift detection**: monitor classifier confidence distribution weekly.
  Alert if mean confidence drops > 5%.
- **Cost tracking**: OCR cost per page, LLM tokens per extraction, total cost
  per claim.

### Scaling Considerations

- **Horizontal scaling**: Processing workers are stateless; scale ECS tasks
  or Kubernetes pods based on SQS queue depth.
- **Batch optimization**: Group pages for OCR into batches of 50 for Textract
  async API to reduce per-page cost.
- **LLM throughput**: Use multiple API keys with load balancing. For 10K
  claims x 4 doc groups = 40K LLM calls/day = ~28/minute -- well within
  rate limits.
- **Peak handling**: Insurance has seasonal peaks (storms, open enrollment).
  Design for 3x burst (30K claims/day) with auto-scaling.
- **Multi-region**: Deploy processing pipeline in the same region as S3
  buckets to minimize data transfer costs.

---

## Scenario 3: Real-Time Content Moderation System

### Interview Prompt

> "Design a real-time content moderation system for a social media platform
> that handles 1 million posts per day. The system must detect toxic content,
> spam, and misinformation across both text and images. Responses must be
> returned within 500ms. Minimize false positives to avoid over-censoring
> legitimate speech. Include a human review queue for edge cases and a model
> retraining pipeline that learns from human feedback."

### Clarifying Questions to Ask

1. What types of content? Text-only posts, images, or also video?
   (Assume text + images, no video for V1.)
2. What are the moderation actions? (Remove, flag for review, reduce
   distribution, warn user.)
3. What is the acceptable false positive rate? (Target < 1% for removals.)
4. What is the acceptable false negative rate? (Target < 5% for severe
   content like CSAM or violence threats.)
5. Do we need to support multiple languages? (Assume top 10 languages.)
6. Is there a legal/regulatory framework (DSA, CCPA) driving requirements?
7. How many human moderators are available? (Assume 200 moderators across
   time zones.)
8. Do we need to handle adversarial attacks (e.g., Unicode tricks,
   steganography)?

### Requirements Summary

| Category | Requirement |
|----------|------------|
| Functional | Classify posts as: clean, toxic, spam, misinformation |
| Functional | Handle text and image content |
| Functional | Support 10+ languages |
| Functional | Human review queue for borderline cases |
| Functional | Model retraining from human decisions |
| Non-functional | < 500ms P99 latency for moderation decision |
| Non-functional | 1M posts/day = ~12 posts/second avg, ~50/second peak |
| Non-functional | < 1% false positive rate for auto-removals |
| Non-functional | < 5% false negative rate for severe content |
| Scale | Must handle 3x traffic spikes during viral events |

### High-Level Architecture

```
+-------------------+     +-------------------+     +-------------------+
|   Post Service    |     |   Moderation      |     |   Fast-Path       |
|   (upstream)      |     |   Gateway         |     |   Models          |
|                   |     |                   |     |                   |
|  New post created-+---->|  Receive post    -+---->|  Text Classifier  |
|                   |     |  Extract features |     |  (BERT-based,     |
|                   |     |  Route to models  |     |   < 50ms)         |
|                   |     |                   |     |                   |
|                   |     |                   |     |  Image Classifier  |
|                   |     |                   |     |  (ResNet/ViT,     |
|                   |     |                   |     |   < 100ms)        |
|                   |     |                   |     |                   |
|                   |     |                   |     |  Spam Detector    |
|                   |     |                   |     |  (gradient-boosted |
|                   |     |                   |     |   features, <20ms)|
+-------------------+     +-------------------+     +-------------------+
                                                           |
                                              +------------+------------+
                                              |            |            |
                                              v            v            v
                                         +--------+  +--------+  +-----------+
                                         | CLEAN  |  | BLOCK  |  | UNCERTAIN |
                                         | (pass) |  |(remove)|  | (review)  |
                                         +--------+  +--------+  +-----------+
                                                                       |
                                                                       v
                                                          +-------------------+
                                                          |  Slow-Path        |
                                                          |  Analysis         |
                                                          |                   |
                                                          |  LLM reasoning    |
                                                          |  Context analysis |
                                                          |  User history     |
                                                          +-------------------+
                                                                  |
                                                       +----------+----------+
                                                       |                     |
                                                       v                     v
                                                  +--------+         +-----------+
                                                  | DECIDE |         | HUMAN     |
                                                  |(auto)  |         | REVIEW    |
                                                  +--------+         +-----------+
                                                                          |
                                                                          v
                                                                 +-------------------+
                                                                 | Feedback Loop     |
                                                                 | Label -> Retrain  |
                                                                 +-------------------+
```

### Detailed Component Design

**1. Moderation Gateway**

- Thin service (Go or Rust for low latency) that sits in the post-creation
  path synchronously.
- Receives the post (text, image URLs, user_id, metadata).
- Extracts features: text language, text length, user account age, user
  prior violation count, post metadata.
- Fans out to multiple classifiers in parallel.
- Aggregates scores, applies decision logic, returns verdict within 500ms.
- If models don't respond within 400ms, falls back to a conservative
  allow-and-queue-for-async-review policy.

**2. Text Classifier**

- Fine-tuned `distilbert-base-multilingual-cased` on 500K labeled examples
  from prior moderation decisions.
- Multi-label classification: toxicity, hate speech, harassment, sexual
  content, self-harm, spam.
- Deployed on GPU instances (NVIDIA T4) with ONNX Runtime for < 50ms
  inference.
- Outputs per-label confidence scores.

**3. Image Classifier**

- Two-stage pipeline:
  - Stage 1: ViT-based safety classifier (NSFW, violence, gore) -- < 80ms.
  - Stage 2: OCR on image text -> feed extracted text through text classifier.
- Deployed on GPU instances with TorchServe.
- For memes and screenshots with embedded text, the OCR path catches
  text-based violations that image-only models miss.

**4. Spam Detector**

- Gradient-boosted model (XGBoost) on handcrafted features:
  - Account age, follower count, posting frequency, URL count, duplicate
    content hash, known spam domain list.
- Extremely fast (< 5ms on CPU).
- High-precision model -- only auto-removes obvious spam.

**5. Decision Engine**

- Combines scores from all classifiers using a rule-based policy engine:
  ```
  IF any_severity_score > 0.95:  -> AUTO_REMOVE
  IF any_severity_score > 0.70:  -> SLOW_PATH_ANALYSIS
  IF spam_score > 0.90:          -> AUTO_REMOVE
  IF all_scores < 0.30:          -> ALLOW
  ELSE:                          -> SLOW_PATH_ANALYSIS
  ```
- Thresholds are configurable per content category and per region (different
  legal requirements).
- The decision, scores, and reasoning are logged for every post.

**6. Slow-Path Analysis**

- Asynchronous pipeline for borderline cases (~15% of posts).
- Uses an LLM (Claude 3.5 Haiku for speed) to analyze context:
  - "Is this sarcasm or genuine hate speech?"
  - "Does this image + caption combination constitute misinformation?"
- Pulls user history: prior violations, posting patterns, trust score.
- Checks against known misinformation claims database (claim-matching with
  embeddings against a fact-check database).
- If LLM confidence > 0.85, auto-decide. Otherwise, route to human review.

**7. Human Review Queue**

- Priority queue ordered by: severity x reach (follower count) x time.
- Moderators see: the post content, model scores, LLM reasoning, user
  history, similar past decisions.
- Moderators choose: approve, remove, reduce distribution, or escalate.
- Target: moderator reviews a case in < 60 seconds.
- 200 moderators x 8 hours x 60 cases/hour = 96,000 cases/day capacity.
  With ~5% of 1M posts needing human review = 50K cases -- well within
  capacity.

**8. Retraining Pipeline**

- Human decisions flow into a labeled dataset stored in a data lake.
- Weekly retraining of text and image classifiers:
  - New training data = last week's human decisions + hard negatives from
    false positives.
  - Train on GPU cluster (4x A100), evaluate on held-out test set.
  - Only promote new model if accuracy improves on test set AND false
    positive rate does not increase.
- A/B test new models on 5% of traffic for 48 hours before full rollout.
- Monthly recalibration of decision thresholds based on false positive/
  negative rates.

### Data Flow Walkthrough

```
1. User creates a post: "These people should be removed from our country"
   + an image of a political rally
2. Post Service sends to Moderation Gateway (sync call)
3. Gateway fans out in parallel:
   a. Text Classifier: toxicity=0.74, hate_speech=0.68 (uncertain)
   b. Image Classifier: violence=0.12, nsfw=0.02 (clean)
   c. Image OCR: extracts banner text "Unity Rally 2026" -> clean
   d. Spam Detector: spam=0.03 (clean)
4. Decision Engine: hate_speech 0.68 > 0.30 but < 0.95 -> SLOW_PATH
5. Gateway returns PENDING to Post Service (post is published but flagged)
6. Slow-path LLM analysis:
   - Context: political speech, no specific target named
   - User history: 2-year account, 0 prior violations, political commenter
   - LLM reasoning: "Ambiguous -- could be anti-immigration rhetoric or
     legitimate political commentary. Confidence: 0.55"
   - Decision: route to human review
7. Moderator reviews:
   - Sees post, model scores, LLM reasoning
   - Decides: APPROVE (legitimate political speech)
8. Decision logged, user trust score updated (+1)
9. This example added to training set as a negative (not toxic) for
   reducing false positives on political speech
```

### Trade-offs Discussion

| Decision | Chosen | Alternative | Why |
|----------|--------|-------------|-----|
| Sync vs async | Sync fast-path + async slow-path | Fully async | Users expect instant posting; sync fast-path allows most posts through quickly |
| Text model | DistilBERT | Full BERT / LLM | DistilBERT meets the 50ms latency budget; LLM is too slow for sync path |
| Decision on uncertain | Publish + flag | Block until reviewed | Blocking degrades UX for 15% of posts; publish + flag balances safety and UX |
| Retraining frequency | Weekly | Daily / Monthly | Weekly catches drift fast enough; daily is operationally expensive; monthly is too slow |
| Threshold tuning | Per-category per-region | Global | Legal requirements vary by jurisdiction; hate speech thresholds differ EU vs US |
| LLM in slow path | Claude 3.5 Haiku | GPT-4o | Haiku is faster and cheaper for this context-analysis task; GPT-4o for complex cases |

### Operations and Monitoring

- **Real-time dashboards**: posts/second, auto-remove rate, slow-path rate,
  human review queue depth, average decision latency.
- **False positive tracking**: sample 500 auto-removed posts weekly for human
  audit. Target < 1%.
- **False negative tracking**: sample 1000 auto-approved posts weekly. Target
  < 5% missed violations.
- **Moderator health**: track cases per moderator per hour, agreement rate
  between moderators (Cohen's kappa), moderator well-being check-ins (exposure
  to harmful content).
- **Adversarial monitoring**: track new evasion patterns (Unicode
  substitutions, image text overlays). Update rule-based pre-filters weekly.
- **Latency monitoring**: P50/P95/P99 per model. Alert if P99 > 400ms on any
  single model.
- **Incident response**: if a viral harmful post bypasses the system, have a
  rapid-response manual review process with < 15 minute SLA.

### Scaling Considerations

- **Auto-scaling**: GPU inference pods scale based on request queue depth.
  Pre-warm capacity for known high-traffic events.
- **Model serving**: Use Triton Inference Server for batching multiple
  requests on a single GPU (increases throughput 3-5x).
- **Geographic distribution**: Deploy moderation models in each region to
  minimize latency. Models are small enough to replicate.
- **Feature store**: Pre-compute user features (account age, violation count)
  in Redis for < 1ms lookup.
- **Spike handling**: During viral events, temporarily raise auto-allow
  threshold (accept more false negatives) to prevent queue overflow.
  Compensate with more aggressive async review.
- **Multi-modal scaling**: As video support is added, use frame sampling
  (1 frame/second) + audio transcription to keep latency manageable.

---

## Scenario 4: Multi-Tenant LLM Platform (LLM Gateway)

### Interview Prompt

> "Design a multi-tenant LLM platform that serves as an AI gateway for 100+
> enterprise clients. Each client has different model preferences, rate limits,
> and strict data isolation requirements. The platform must support routing to
> OpenAI, Anthropic, and open-source models. Include usage tracking for
> billing, prompt management with versioning, and configurable guardrails
> per tenant. Think of it as an internal LLM infrastructure product."

### Clarifying Questions to Ask

1. What is the expected total request volume across all tenants? (Assume
   10M requests/day, ~120 requests/second average.)
2. What latency overhead is acceptable from the gateway? (< 50ms added on
   top of LLM provider latency.)
3. Do tenants need streaming support? (Yes, SSE streaming is required.)
4. What level of data isolation? (Tenant data must never be visible to
   other tenants -- logical isolation minimum, physical isolation for
   premium tier.)
5. Do we need to support fine-tuned models per tenant?
6. What guardrails are needed? (PII detection, topic restrictions, output
   format enforcement, token limits.)
7. Is there a self-service portal or is configuration API-only?
8. Do we need prompt playground / testing tools?

### Requirements Summary

| Category | Requirement |
|----------|------------|
| Functional | Route requests to OpenAI, Anthropic, open-source models |
| Functional | Per-tenant configuration: models, rate limits, guardrails |
| Functional | Usage tracking and billing per tenant |
| Functional | Prompt template management with versioning |
| Functional | Input/output guardrails (PII, topic, format) |
| Functional | Tenant admin portal |
| Non-functional | < 50ms gateway overhead |
| Non-functional | 10M requests/day (120 RPS average, 500 RPS peak) |
| Non-functional | 99.95% uptime SLA |
| Non-functional | Strict data isolation between tenants |
| Scale | 100+ tenants, growing to 500+ |

### High-Level Architecture

```
+-------------------+     +-------------------+     +-------------------+
|   Tenant Apps     |     |   API Gateway     |     |   LLM Router      |
|                   |     |   (Kong / Envoy)  |     |                   |
|  Client SDK     --+---->|  Auth (API keys)  |---->|  Tenant Config    |
|  (Python, JS,    |     |  Rate Limiting     |     |  Lookup           |
|   Go SDKs)       |     |  Request Logging   |     |                   |
|                   |     |                   |     |  Model Selection   |
+-------------------+     +-------------------+     |                   |
                                                     |  Prompt Template  |
                                                     |  Resolution       |
                                                     +-------------------+
                                                            |
                                           +----------------+----------------+
                                           |                |                |
                                           v                v                v
                                  +-------------+  +-------------+  +---------------+
                                  |  OpenAI     |  |  Anthropic  |  |  Self-Hosted   |
                                  |  Adapter    |  |  Adapter    |  |  Adapter       |
                                  |             |  |             |  |  (vLLM /       |
                                  | GPT-4o      |  | Claude 3.5  |  |  Llama 3 70B)  |
                                  | GPT-4o-mini |  | Haiku       |  |                |
                                  +-------------+  +-------------+  +---------------+
                                           |                |                |
                                           +----------------+----------------+
                                                            |
                                                            v
                                                   +-------------------+
                                                   |  Output Pipeline  |
                                                   |                   |
                                                   |  Guardrail Check  |
                                                   |  PII Scrubbing    |
                                                   |  Usage Metering   |
                                                   |  Response Cache   |
                                                   |  Stream to Client |
                                                   +-------------------+

Supporting Services:
+-------------------+   +-------------------+   +-------------------+
| Tenant Config     |   | Billing Service   |   | Prompt Registry   |
| Store             |   |                   |   |                   |
| (PostgreSQL +     |   | Usage Aggregation |   | Version Control   |
|  Redis cache)     |   | Invoice Gen       |   | A/B Testing       |
+-------------------+   +-------------------+   +-------------------+

+-------------------+   +-------------------+
| Admin Portal      |   | Observability     |
| (React SPA)       |   | (Datadog /        |
|                   |   |  Grafana + custom) |
+-------------------+   +-------------------+
```

### Detailed Component Design

**1. API Gateway Layer**

- Kong or Envoy as the edge proxy.
- Authentication: API key per tenant, validated against Redis-cached key
  store. Keys are hashed (SHA-256) at rest.
- Rate limiting: token bucket per tenant, configurable per model. Stored in
  Redis. Example: Tenant A gets 1000 RPM for GPT-4o, 5000 RPM for
  GPT-4o-mini.
- Request logging: every request logged with `tenant_id`, `model`,
  `input_tokens`, `output_tokens`, `latency`, `status`. Logs go to Kafka
  for async processing.
- Streaming support: SSE pass-through with chunked transfer encoding.

**2. LLM Router**

- Core service (Python/FastAPI or Go for performance).
- Tenant Config Lookup: fetch from Redis (cache) -> PostgreSQL (source of
  truth). Config includes:
  ```json
  {
    "tenant_id": "acme-corp",
    "allowed_models": ["gpt-4o", "claude-3-5-sonnet", "llama-3-70b"],
    "default_model": "gpt-4o",
    "rate_limits": {"gpt-4o": 1000, "claude-3-5-sonnet": 500},
    "guardrails": {
      "pii_detection": true,
      "blocked_topics": ["competitor-names"],
      "max_output_tokens": 4096
    },
    "fallback_chain": ["gpt-4o", "claude-3-5-sonnet", "llama-3-70b"]
  }
  ```
- Model Selection: uses tenant's requested model (or default). If the
  requested model is not in `allowed_models`, returns 403.
- Fallback: if the primary model returns a 5xx or times out, automatically
  retry with the next model in the fallback chain.
- Prompt Template Resolution: if the request references a prompt template
  by ID, resolve it from the Prompt Registry and merge with user variables.

**3. Provider Adapters**

- Adapter pattern: each LLM provider has an adapter that normalizes the
  request/response format.
- Common interface:
  ```
  interface LLMAdapter {
    complete(request: NormalizedRequest): Stream<NormalizedChunk>
  }
  ```
- OpenAI Adapter: maps to Chat Completions API.
- Anthropic Adapter: maps to Messages API with system prompt handling.
- Self-Hosted Adapter: calls vLLM's OpenAI-compatible API endpoint.
- Adapters handle provider-specific retry logic, error mapping, and
  streaming format differences.
- Connection pooling: maintain persistent HTTP/2 connections to each
  provider to reduce connection overhead.

**4. Guardrails Engine**

- Pre-request guardrails (input):
  - PII Detection: regex + NER model (Presidio) scans input for SSNs,
    credit cards, emails. Action: redact or reject based on tenant config.
  - Topic blocking: embedding similarity against a blocklist of topic
    embeddings. If cosine similarity > 0.85, reject with explanation.
  - Prompt injection detection: classifier trained on known injection
    patterns. Flag suspicious inputs for logging.
  - Max input token enforcement.
- Post-response guardrails (output):
  - PII Detection on output: scan LLM response for leaked PII.
  - Content safety: lightweight toxicity classifier on output.
  - Output format validation: if tenant specifies JSON schema, validate
    response structure.
  - Max output token enforcement (stop generation at limit).
- Guardrails add < 30ms total (run in parallel where possible).

**5. Prompt Registry**

- Git-like versioning for prompt templates.
- Each template has: `template_id`, `version`, `tenant_id`, `model_hint`,
  `system_prompt`, `user_prompt_template`, `variables[]`, `metadata`.
- Templates are stored in PostgreSQL with full version history.
- API:
  - `POST /prompts` -- create new template
  - `PUT /prompts/{id}` -- create new version
  - `GET /prompts/{id}?version=latest` -- retrieve
  - `POST /prompts/{id}/test` -- test with sample variables
- A/B testing: multiple prompt versions can be active simultaneously with
  traffic splitting (e.g., 90% v3, 10% v4).

**6. Billing Service**

- Usage events flow from Kafka into a TimescaleDB (time-series PostgreSQL
  extension).
- Each event: `tenant_id`, `model`, `input_tokens`, `output_tokens`,
  `timestamp`, `cost`.
- Cost calculation: `cost = input_tokens * model_input_price +
  output_tokens * model_output_price + gateway_markup`.
- Real-time usage dashboard per tenant.
- Monthly invoice generation with line items per model.
- Budget alerts: tenants can set monthly spending limits; gateway rejects
  requests when budget exceeded (configurable: reject vs warn).

**7. Tenant Admin Portal**

- React SPA with tenant-scoped views.
- Features: model configuration, rate limit settings, guardrail
  configuration, API key management (rotate, revoke), usage dashboards,
  prompt management, billing history.
- Role-based access: tenant admin, tenant developer, platform admin.

### Data Flow Walkthrough

```
1. Acme Corp's app sends request:
   POST /v1/chat/completions
   Headers: { Authorization: "Bearer acme-key-xxx" }
   Body: {
     "model": "gpt-4o",
     "prompt_template": "customer-support-v3",
     "variables": { "customer_name": "Jane", "issue": "billing dispute" },
     "stream": true
   }

2. API Gateway:
   - Validates API key -> tenant_id = "acme-corp"
   - Checks rate limit: 847/1000 RPM used -> allowed
   - Logs request metadata to Kafka

3. LLM Router:
   - Loads tenant config from Redis
   - Confirms "gpt-4o" is in allowed_models
   - Resolves prompt template "customer-support-v3":
     System: "You are a helpful customer support agent for Acme Corp..."
     User: "Customer Jane has a billing dispute. Help resolve it."

4. Input Guardrails (parallel, 25ms total):
   - PII scan: no PII detected
   - Topic filter: no blocked topics
   - Injection detection: clean

5. OpenAI Adapter:
   - Formats request for Chat Completions API
   - Sends with Acme's allocated API key (or shared key with tenant isolation)
   - Receives streaming response

6. Output Guardrails (applied per chunk):
   - PII scan on accumulated response: clean
   - Token count: 342 output tokens (within 4096 limit)

7. Response streamed to client via SSE

8. Post-request:
   - Usage event: {tenant: "acme-corp", model: "gpt-4o", input: 156,
     output: 342, cost: $0.0043, latency: 2.3s, prompt_version: "v3"}
   - Written to Kafka -> TimescaleDB
   - Rate limit counter updated in Redis
```

### Trade-offs Discussion

| Decision | Chosen | Alternative | Why |
|----------|--------|-------------|-----|
| Gateway language | Go | Python (FastAPI) | Go's concurrency model and low GC pause times suit the < 50ms overhead requirement |
| Config store | PostgreSQL + Redis cache | DynamoDB | PostgreSQL for relational config; Redis for sub-1ms reads; DynamoDB adds vendor lock-in |
| Billing granularity | Per-request token-level | Hourly aggregates | Token-level enables precise billing and debugging; higher storage cost is justified |
| Data isolation | Logical (shared infra, tenant_id filtering) | Physical (separate DBs per tenant) | Logical scales to 500+ tenants; physical isolation offered as premium tier for regulated clients |
| Prompt storage | PostgreSQL with versioning | Git repo | DB is easier to query and integrate with API; Git for large prompt engineering teams |
| Guardrails | In-gateway | Separate microservice | In-gateway avoids network hop; separate service if guardrails become complex |

### Operations and Monitoring

- **Gateway health**: request rate, error rate, P50/P95/P99 latency (gateway
  overhead only, excluding LLM time).
- **Provider health**: per-provider success rate, latency, rate limit hits.
  Auto-failover if a provider's error rate > 5%.
- **Tenant health**: per-tenant request volume, error rate, budget usage.
  Alert tenant admins on anomalies.
- **Guardrail metrics**: PII detection rate, topic block rate, injection
  detection rate per tenant.
- **Cost optimization**: track cost per request per model. Suggest cheaper
  model alternatives to tenants if quality metrics allow.
- **Security monitoring**: API key usage patterns, detect key leakage (same
  key used from unexpected IPs), enforce key rotation policy.
- **SLA tracking**: per-tenant uptime and latency SLA compliance.

### Scaling Considerations

- **Stateless gateway**: scale horizontally behind a load balancer. Target
  < 10ms per gateway node.
- **Redis cluster**: for rate limiting and config cache. Use Redis Cluster
  mode for HA.
- **Kafka partitioning**: partition usage events by tenant_id for ordered
  processing within a tenant.
- **Provider rate limits**: pool API keys across tenants with fair scheduling.
  Use token bucket per provider to stay within aggregate limits.
- **Self-hosted models**: scale vLLM deployments per GPU cluster. Use
  Kubernetes with GPU node pools and autoscaling based on queue depth.
- **Multi-region**: deploy gateway in multiple regions. Provider routing
  uses the nearest provider endpoint.
- **Tenant isolation at scale**: for 500+ tenants, shard the config store
  by tenant_id range. Cache is per-gateway-node to avoid Redis hotspots.

---

## Scenario 5: Personalized Recommendation System with ML

### Interview Prompt

> "Design a personalized recommendation system for an e-commerce site with
> 10 million products and 50 million users. The system must provide real-time
> recommendations on product pages (similar items, 'customers also bought')
> and batch recommendations for daily email campaigns. Include a feature store
> for user and item features, an A/B testing framework for model experiments,
> and a strategy for cold start handling for new users and new products."

### Clarifying Questions to Ask

1. What is the latency requirement for real-time recommendations? (Target:
   < 100ms P99.)
2. How many recommendations per request? (10-20 items per widget.)
3. What signals do we have? (Views, clicks, add-to-cart, purchases, ratings,
   search queries, browse history.)
4. How often do product catalogs change? (1000 new products/day, 10K price
   updates/day.)
5. What is the email campaign scale? (10M personalized emails/day.)
6. Are there business rules? (Don't recommend out-of-stock items, promote
   high-margin items, diversity requirements.)
7. What is the current tech stack? (Assume AWS, Kubernetes, existing data
   warehouse in Snowflake.)
8. Do we need to handle seasonality? (Yes -- holiday shopping, back-to-school,
   etc.)

### Requirements Summary

| Category | Requirement |
|----------|------------|
| Functional | Real-time "similar items" on product pages |
| Functional | Real-time "customers also bought" on product pages |
| Functional | Batch "recommended for you" for email campaigns |
| Functional | Feature store for user/item features |
| Functional | A/B testing for model experiments |
| Functional | Cold start for new users and products |
| Non-functional | < 100ms P99 for real-time serving |
| Non-functional | 10M products, 50M users |
| Non-functional | 500M events/day (views, clicks, purchases) |
| Non-functional | 10M batch recommendations/day for emails |
| Scale | Handle 10x traffic during Black Friday |

### High-Level Architecture

```
+-------------------+     +-------------------+     +-------------------+
|   Event Sources   |     |   Stream          |     |   Feature Store   |
|                   |     |   Processing      |     |                   |
|  Clickstream     -+---->|  Kafka + Flink   -+---->|  Online: Redis    |
|  Purchase events -+---->|                   |     |  (user features,  |
|  Search queries  -+---->|  Real-time feature|     |   item features)  |
|  Ratings         -+---->|  computation      |     |                   |
|                   |     |                   |     |  Offline: Snowflake|
+-------------------+     +-------------------+     |  (historical      |
                                                     |   aggregates)     |
                                                     +-------------------+
                                                            |
                                                            v
+-------------------+     +-------------------+     +-------------------+
|   Training        |     |   Model Registry  |     |   Serving Layer   |
|   Pipeline        |     |                   |     |                   |
|                   |     |  MLflow           |     |  Candidate Gen    |
|  Candidate Gen   -+---->|  - model versions |---->|  (ANN index)      |
|  (Two-Tower/ALS) |     |  - metrics        |     |                   |
|                   |     |  - A/B assignments|     |  Ranking Model    |
|  Ranking Model   -+---->|                   |---->|  (real-time       |
|  (GBDT/DNN)      |     |                   |     |   inference)      |
|                   |     |                   |     |                   |
+-------------------+     +-------------------+     |  Business Rules   |
                                                     |  (filtering)      |
                                                     +-------------------+
                                                            |
                                              +-------------+-------------+
                                              |                           |
                                              v                           v
                                    +-------------------+       +-------------------+
                                    |  Real-Time API    |       |  Batch Pipeline   |
                                    |                   |       |                   |
                                    |  Product page     |       |  Spark job        |
                                    |  widgets          |       |  10M emails/day   |
                                    |  < 100ms P99      |       |  Pre-compute top  |
                                    |                   |       |  20 per user      |
                                    +-------------------+       +-------------------+

Supporting:
+-------------------+   +-------------------+
| A/B Test Service  |   | Monitoring        |
| (Experimentation  |   | (Grafana +        |
|  Platform)        |   |  custom metrics)  |
+-------------------+   +-------------------+
```

### Detailed Component Design

**1. Event Ingestion and Stream Processing**

- All user interactions (page views, clicks, add-to-cart, purchases, searches,
  ratings) are sent as events to Kafka.
- Event schema: `{user_id, event_type, item_id, timestamp, session_id,
  device, context}`.
- Apache Flink processes events in real-time:
  - Computes real-time user features: items viewed in last 30 minutes,
    categories browsed in current session, cart contents.
  - Computes real-time item features: trending score (views in last hour),
    current stock level.
  - Writes to Redis (online feature store) with TTLs.
- Flink also writes raw events to S3 (Parquet format) for batch processing.
- Daily Spark jobs aggregate historical features: purchase history (90 days),
  category affinity scores, price sensitivity, brand preferences, lifetime
  value.

**2. Feature Store**

- Two-tier architecture:
  - **Online store (Redis Cluster)**: serves features at < 5ms for real-time
    inference. Stores recent user features and pre-computed item features.
    Key patterns: `user:{user_id}:features`, `item:{item_id}:features`.
  - **Offline store (Snowflake)**: stores full feature history for training.
    Point-in-time correct joins to prevent data leakage during training.
- Feature definitions managed in a feature registry (Feast or Tecton):
  ```
  User features:
  - purchase_count_90d (int)
  - avg_order_value (float)
  - category_affinity (map<string, float>)
  - price_sensitivity (float, 0-1)
  - last_active_timestamp (datetime)
  - session_items_viewed (list<string>) [real-time]

  Item features:
  - category (string)
  - brand (string)
  - price (float)
  - avg_rating (float)
  - review_count (int)
  - sales_velocity_7d (float)
  - trending_score (float) [real-time]
  - embedding (float[128]) [from Two-Tower model]
  - stock_level (int) [real-time]
  ```
- Feature freshness SLAs: real-time features < 5 seconds, batch features
  < 24 hours.

**3. Candidate Generation**

- Two-Tower Model (retrieval model):
  - User tower: encodes user features into a 128-dim embedding.
  - Item tower: encodes item features into a 128-dim embedding.
  - Trained on (user, item, label) triples from purchase/click data.
  - Loss: sampled softmax or in-batch negatives.
  - Framework: TensorFlow / PyTorch, trained on GPU cluster.
- Item embeddings are pre-computed and stored in an Approximate Nearest
  Neighbor (ANN) index:
  - Technology: FAISS (IVF-PQ) or ScaNN.
  - Index size: 10M items x 128 dims = ~5GB. Fits in memory on a single
    node (replicated for HA).
  - Updated daily with new/changed items.
- At query time: compute user embedding from real-time features, query ANN
  index for top-500 candidates in < 10ms.
- Additional candidate sources:
  - Co-purchase graph: "customers who bought X also bought Y" using item-item
    collaborative filtering (precomputed, stored in Redis).
  - Content-based: items in the same category/brand with high ratings.
  - Trending: globally or category-level trending items.
- Merge candidates from all sources, deduplicate -> ~500-1000 candidates.

**4. Ranking Model**

- Two-stage ranking:
  - Stage 1 -- Lightweight ranker: XGBoost model scores 1000 candidates in
    < 20ms. Features: user-item affinity, price match, category match,
    item popularity, trending score. Outputs top-100.
  - Stage 2 -- Deep ranker: DNN model (MLP with 3 hidden layers) scores 100
    candidates in < 30ms. Additional features: user sequence (last 20
    interactions via transformer encoder), cross-features. Outputs top-20.
- Final scoring combines model score with business rules:
  - Boost high-margin items (configurable weight).
  - Penalize recently viewed items (user already saw them).
  - Enforce diversity: no more than 3 items from the same brand in top-10.
  - Filter out-of-stock items.
  - Apply promotional boosts for campaign items.

**5. Real-Time Serving API**

- Endpoint: `GET /recommendations?user_id=X&item_id=Y&type=similar&count=10`
- Flow (all within 100ms budget):
  1. Feature lookup from Redis: 5ms
  2. User embedding computation: 10ms
  3. ANN retrieval: 10ms
  4. Candidate merging: 5ms
  5. Stage 1 ranking: 20ms
  6. Stage 2 ranking: 30ms
  7. Business rules + filtering: 5ms
  8. A/B test assignment: 2ms
  9. Response serialization: 3ms
  Total: ~90ms P95
- Deployed on Kubernetes with HPA (Horizontal Pod Autoscaler) based on CPU
  and request latency.
- Response cached in CDN for anonymous users (cache key = item_id + rec_type,
  TTL 5 minutes).

**6. Batch Recommendation Pipeline**

- Daily Spark job running on EMR:
  1. Load all 50M user feature vectors from offline store.
  2. For each user, compute top-20 recommendations using the same candidate
     generation + ranking pipeline (but batch-optimized).
  3. Store results in DynamoDB: `{user_id: [item_id_1, item_id_2, ...]}`.
  4. Email service reads from DynamoDB when composing personalized emails.
- Optimization: partition users into segments, run in parallel across 100+
  Spark executors. Full run completes in ~4 hours.
- Freshness: results are computed overnight and reflect yesterday's data.
- Fallback: if batch job fails, use previous day's recommendations (still
  valid for 48 hours).

**7. Cold Start Handling**

- **New Users (no interaction history)**:
  - Use contextual signals: device type, referral source, landing page,
    geographic location, time of day.
  - Serve popularity-based recommendations segmented by context (e.g.,
    trending items in user's country/category they landed on).
  - After 3-5 interactions, transition to personalized recommendations
    using a "warm-up" model trained specifically on sparse-interaction users.
  - Explicit preference collection: optional onboarding quiz ("What
    categories interest you?").

- **New Items (no interaction data)**:
  - Content-based embedding: compute item embedding from title, description,
    images, category, and attributes using a pre-trained model.
  - Place new items in the ANN index immediately based on content embedding.
  - Boost new items with an "exploration bonus" in the ranking model
    (decays over 7 days).
  - Use multi-armed bandit (Thompson Sampling) to allocate exploration
    traffic: show new items to a small % of users and rapidly learn their
    appeal.
  - After 100+ interactions, the collaborative signal dominates and the
    content-based embedding is blended out.

**8. A/B Testing Framework**

- Experimentation service assigns users to experiments at request time.
- Assignment: deterministic hash `hash(user_id + experiment_id) % 100`
  for consistent bucketing.
- Supports:
  - Model A/B tests: different ranking models.
  - Feature A/B tests: include/exclude a feature.
  - Algorithm A/B tests: different candidate generation strategies.
  - Business rule A/B tests: different diversity/boost configurations.
- Metrics tracked per experiment:
  - Click-through rate (CTR) on recommendations.
  - Add-to-cart rate.
  - Conversion rate (purchase).
  - Revenue per session.
  - Diversity (unique categories in shown recommendations).
  - Coverage (% of catalog shown to at least one user).
- Statistical rigor: use sequential testing (always-valid p-values) to
  allow early stopping. Minimum 7-day run for seasonality.
- Guardrail metrics: if an experiment degrades revenue per session by > 2%,
  auto-kill the experiment.

### Data Flow Walkthrough

```
Real-time path (user viewing a product page):

1. User views Product #P-8834 (running shoes)
2. Frontend calls: GET /recommendations?user_id=U-442&item_id=P-8834
   &type=similar&count=10
3. Serving layer:
   a. Feature lookup (Redis, 5ms):
      - User U-442: category_affinity={shoes: 0.8, fitness: 0.6},
        price_sensitivity=0.4, session_items=[P-1200, P-8834]
      - Item P-8834: category=shoes, brand=Nike, price=$129, trending=0.7
   b. User embedding (10ms): encode features -> [0.12, -0.45, ...]
   c. ANN search (10ms): top-500 nearest items to P-8834's embedding
      + top-200 nearest to user embedding -> 650 unique candidates
   d. Co-purchase lookup (Redis, 3ms): users who bought P-8834 also bought
      [P-9012, P-7756, P-3344, ...] -> add 50 candidates
   e. Total: ~700 unique candidates
   f. Stage 1 ranking (XGBoost, 20ms): score 700 -> top-100
   g. Stage 2 ranking (DNN, 30ms): score 100 -> top-20
   h. Business rules (5ms):
      - Remove P-8834 (current item) and P-1200 (already viewed)
      - Remove 2 out-of-stock items
      - Enforce diversity: cap 3 Nike items
      - Boost P-5567 (promotional campaign)
      -> Final 10 recommendations
   i. A/B test: user assigned to experiment "ranking-v7-vs-v8",
      bucket: control (v7)
   j. Return 10 items with scores
4. Event logged: {user: U-442, type: "rec_shown", items: [...],
   experiment: "ranking-v7", latency: 87ms}
5. User clicks on recommendation P-9012
6. Click event -> Kafka -> Flink updates session features in Redis
7. Next recommendation request reflects the new click
```

### Trade-offs Discussion

| Decision | Chosen | Alternative | Why |
|----------|--------|-------------|-----|
| Candidate gen | Two-Tower + co-purchase | Matrix factorization (ALS) | Two-Tower handles cold start via content features; ALS is simpler but pure collaborative |
| ANN index | FAISS IVF-PQ | HNSW (hnswlib) | IVF-PQ is more memory-efficient for 10M items; HNSW has better recall but 3x memory |
| Ranking | Two-stage (XGBoost + DNN) | Single DNN | Two-stage reduces DNN inference from 1000 to 100 items, saving 70ms |
| Online feature store | Redis Cluster | DynamoDB | Redis has < 1ms latency; DynamoDB has ~5ms and is pricier at this read volume |
| Batch pipeline | Spark on EMR | Airflow + Python | Spark handles 50M users in 4 hours; pure Python would take days |
| A/B testing | Custom platform | LaunchDarkly | Custom gives ML-specific metrics (CTR, revenue); LaunchDarkly better for feature flags |
| Cold start items | Content embedding + exploration bonus | Random placement | Content embedding gives a reasonable starting signal; random wastes impressions |

### Operations and Monitoring

- **Recommendation quality**: daily dashboard tracking CTR, conversion rate,
  revenue per recommendation widget, by rec type (similar, also-bought,
  personalized).
- **Model staleness**: alert if model is not retrained within 48 hours.
  Track prediction drift using PSI (Population Stability Index) on score
  distributions.
- **Feature freshness**: monitor feature update timestamps. Alert if
  real-time features are > 1 minute stale or batch features are > 26 hours
  stale.
- **Serving latency**: P50/P95/P99 per component (feature lookup, embedding,
  ANN, ranking). Alert if P99 > 150ms.
- **ANN index quality**: sample 1000 queries/day, compare ANN results to
  exact brute-force search. Track recall@100.
- **A/B test health**: automated checks for Sample Ratio Mismatch (SRM)
  to detect assignment bugs. Alert if observed ratio deviates > 1% from
  expected.
- **Coverage and diversity**: track what % of catalog appears in
  recommendations (target > 30%). Monitor average intra-list diversity.
- **Cold start monitoring**: separately track CTR for new users (< 5
  interactions) and new items (< 100 interactions). Compare to established
  user/item baselines.
- **Business rule compliance**: verify 0% of recommendations are out-of-stock
  or violate diversity constraints.

### Scaling Considerations

- **Black Friday scaling**: pre-warm 10x serving capacity. Pre-compute
  recommendations for top 10% of users (by predicted activity). Cache
  aggressively with shorter TTLs.
- **ANN index updates**: use a shadow index pattern -- build new index in
  background, swap atomically. No downtime during daily updates.
- **Feature store scaling**: Redis Cluster supports 100K+ reads/second per
  shard. Partition by user_id hash.
- **Training scaling**: distributed training across 8 GPUs for Two-Tower
  model. Incremental training on new data daily (full retrain weekly).
- **Batch pipeline scaling**: increase Spark executors for 50M -> 200M users.
  Partition by user_id range, each executor handles a shard independently.
- **Multi-region**: deploy serving layer in each region with local ANN index
  replicas. Feature store is global (Redis Global Tables or cross-region
  replication).
- **Cost optimization**: use spot instances for batch training and Spark jobs.
  Use reserved instances for serving layer (predictable load). Right-size
  GPU instances based on actual model inference throughput.

---

## Quick Reference: Common Patterns Across All Scenarios

| Pattern | Where Used |
|---------|-----------|
| Two-tier storage (hot/cold) | Feature Store (Scenario 5), Conversation Store (Scenario 1) |
| Human-in-the-loop | Document Processing (2), Content Moderation (3) |
| Async slow-path for complex cases | Content Moderation (3), Document Processing (2) |
| Confidence-based routing | Document Processing (2), Content Moderation (3) |
| A/B testing | Recommendations (5), Prompt versions (4) |
| Streaming responses | RAG Chatbot (1), LLM Gateway (4) |
| Adapter/provider abstraction | LLM Gateway (4) |
| Feature/config caching in Redis | All scenarios |
| Fallback chains | LLM Gateway (4), OCR (2) |
| Model retraining from production data | Content Moderation (3), Recommendations (5) |
| Rate limiting and usage metering | LLM Gateway (4) |
| ACL/data isolation | RAG Chatbot (1), LLM Gateway (4) |

---

## Interview Tips

1. **Always start with clarifying questions.** This shows you don't jump to
   solutions and that you understand real systems have constraints.

2. **State your assumptions explicitly.** "I'm assuming 500ms P99 is
   acceptable" is better than silently designing for 100ms.

3. **Draw the architecture first, then dive deep.** Interviewers want to see
   that you can think at multiple levels of abstraction.

4. **Discuss trade-offs, not just decisions.** "I chose Pinecone over pgvector
   because..." shows engineering maturity.

5. **Address failure modes.** What happens when the LLM provider goes down?
   What happens when the feature store is stale? Interviewers love seeing
   you think about the unhappy path.

6. **Mention monitoring and observability proactively.** Production ML systems
   fail silently (model drift, feature skew). Showing you know what to
   monitor differentiates you.

7. **Know your numbers.** "10K claims/day = 7/minute = not a throughput
   challenge, so the bottleneck is latency and accuracy" shows you can do
   back-of-envelope calculations.

8. **Connect to business impact.** "A 1% false positive rate means 10,000
   legitimate posts removed daily -- that's a PR risk" shows you think
   beyond pure engineering.
