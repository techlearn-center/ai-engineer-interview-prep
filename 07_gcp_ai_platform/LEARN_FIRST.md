# Learn: GCP for AI Engineers

This is a big topic. Read section by section - don't try to memorize everything at once.

---

## 1. GCP Services Map for AI Engineers

Here's what matters for an AI engineer interview. Don't worry about the 200+ other GCP services.

```
┌─────────────────────────────────────────────────────────┐
│                    YOUR ML WORKFLOW                       │
│                                                           │
│  [Raw Data]  →  [Process]  →  [Train]  →  [Serve]       │
│      │             │            │            │            │
│      v             v            v            v            │
│   Cloud         BigQuery    Vertex AI    Cloud Run        │
│   Storage       Dataflow    Training     Vertex AI        │
│   (GCS)         Pub/Sub     Notebooks    Endpoints        │
│                                                           │
│  ─────────── Supporting Services ───────────              │
│  IAM (permissions)    Artifact Registry (containers)      │
│  Secret Manager       Cloud Build (CI/CD)                 │
│  Cloud Logging        Cloud Monitoring                    │
└─────────────────────────────────────────────────────────┘
```

---

## 2. Cloud Storage (GCS) - Where Your Data Lives

GCS is like a giant folder in the cloud. Everything starts here:
training data, model artifacts, configs, logs.

**Key concepts:**
- **Bucket:** Top-level container (like a drive). Globally unique name.
- **Object/Blob:** A file inside a bucket.
- **URI format:** `gs://my-bucket/path/to/file.csv`

**Python SDK:**
```python
from google.cloud import storage

# Initialize client
client = storage.Client(project="my-project")

# Upload a file
bucket = client.bucket("my-ml-bucket")
blob = bucket.blob("data/training.csv")
blob.upload_from_filename("local_file.csv")

# Download a file
blob = bucket.blob("models/model.pkl")
blob.download_to_filename("local_model.pkl")

# List files
blobs = client.list_blobs("my-ml-bucket", prefix="data/")
for blob in blobs:
    print(blob.name)
```

**Interview tip:** Always mention GCS when discussing where to store:
- Training datasets
- Model artifacts / checkpoints
- Prediction logs
- Config files

---

## 3. BigQuery - SQL for Big Data

BigQuery is Google's data warehouse. You query terabytes of data in seconds using SQL.

**Why AI engineers care:**
- Your training data often lives in BigQuery
- Feature engineering with SQL is fast
- BigQuery ML lets you train models WITH SQL
- Connect BigQuery → Vertex AI pipelines

**Python SDK:**
```python
from google.cloud import bigquery

client = bigquery.Client(project="my-project")

# Run a query
query = """
    SELECT user_id, COUNT(*) as purchase_count, AVG(amount) as avg_amount
    FROM `my-project.sales.transactions`
    WHERE date >= '2024-01-01'
    GROUP BY user_id
"""
df = client.query(query).to_dataframe()  # Returns a pandas DataFrame!
print(df.head())
```

**BigQuery ML (train models with SQL!):**
```sql
-- Create a model
CREATE OR REPLACE MODEL `my_project.my_dataset.churn_model`
OPTIONS(
    model_type='LOGISTIC_REG',
    input_label_cols=['churned']
) AS
SELECT
    tenure_months,
    monthly_charges,
    total_charges,
    churned
FROM `my_project.my_dataset.customers`;

-- Make predictions
SELECT *
FROM ML.PREDICT(
    MODEL `my_project.my_dataset.churn_model`,
    (SELECT tenure_months, monthly_charges, total_charges
     FROM `my_project.my_dataset.new_customers`)
);
```

**Common interview question:** "How would you prepare training data at scale?"
Answer: "I'd use BigQuery to join, filter, and aggregate the raw data, then export
to GCS or load directly into a Vertex AI training pipeline."

---

## 4. Vertex AI - The ML Platform

Vertex AI is Google's unified ML platform. It handles the full ML lifecycle.

### Key Components:

| Component | What it does |
|-----------|-------------|
| **Workbench** | Managed Jupyter notebooks |
| **Training** | Train models (custom or AutoML) |
| **Model Registry** | Store and version trained models |
| **Endpoints** | Deploy models for online predictions |
| **Pipelines** | Orchestrate ML workflows |
| **Feature Store** | Centralized feature management |
| **Experiments** | Track training runs and metrics |

### Custom Training Job:
```python
from google.cloud import aiplatform

aiplatform.init(project="my-project", location="us-central1")

# Submit a training job
job = aiplatform.CustomTrainingJob(
    display_name="my-training-job",
    script_path="train.py",            # Your training script
    container_uri="us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-12:latest",
    requirements=["scikit-learn", "pandas"],
)

model = job.run(
    replica_count=1,
    machine_type="n1-standard-4",
    accelerator_type="NVIDIA_TESLA_T4",  # GPU
    accelerator_count=1,
)
```

### Deploy a Model:
```python
# Upload model to registry
model = aiplatform.Model.upload(
    display_name="my-model-v1",
    artifact_uri="gs://my-bucket/models/v1/",  # Where model files are
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
)

# Deploy to endpoint
endpoint = model.deploy(
    machine_type="n1-standard-4",
    min_replica_count=1,
    max_replica_count=3,  # Auto-scales!
)

# Make predictions
prediction = endpoint.predict(instances=[[1.0, 2.0, 3.0]])
print(prediction.predictions)
```

### Vertex AI Pipelines (Kubeflow):
```python
from kfp.v2 import dsl, compiler

@dsl.component
def preprocess_data(input_path: str, output_path: str):
    import pandas as pd
    df = pd.read_csv(input_path)
    df = df.dropna()
    df.to_csv(output_path, index=False)

@dsl.component
def train_model(data_path: str, model_path: str):
    from sklearn.linear_model import LogisticRegression
    import pandas as pd
    import joblib
    df = pd.read_csv(data_path)
    X, y = df.drop("target", axis=1), df["target"]
    model = LogisticRegression()
    model.fit(X, y)
    joblib.dump(model, model_path)

@dsl.pipeline(name="my-ml-pipeline")
def my_pipeline():
    preprocess_task = preprocess_data(
        input_path="gs://my-bucket/raw_data.csv",
        output_path="gs://my-bucket/clean_data.csv",
    )
    train_task = train_model(
        data_path="gs://my-bucket/clean_data.csv",
        model_path="gs://my-bucket/model.pkl",
    )
    train_task.after(preprocess_task)  # Order matters
```

---

## 5. Cloud Run - Deploy Your API

Cloud Run runs containers. You containerize your FastAPI app and deploy it.

**The workflow:**
```
Write FastAPI app → Dockerfile → Build container → Deploy to Cloud Run
```

**Dockerfile for ML API:**
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

**Deploy commands:**
```bash
# Build and push container
gcloud builds submit --tag gcr.io/my-project/ml-api

# Deploy to Cloud Run
gcloud run deploy ml-api \
    --image gcr.io/my-project/ml-api \
    --platform managed \
    --region us-central1 \
    --memory 2Gi \
    --allow-unauthenticated
```

**Interview tip:** When asked "How would you deploy this model?", mention:
- **Cloud Run** for simple REST API serving (stateless, auto-scales to zero)
- **Vertex AI Endpoints** for managed model serving (auto-scaling, A/B testing, monitoring)
- Explain the trade-off: Cloud Run = more control, Vertex AI = more managed

---

## 6. IAM - Permissions & Security

IAM (Identity and Access Management) controls WHO can do WHAT.

**Key concepts:**
- **Principal:** Who (user, service account, group)
- **Role:** What they can do (viewer, editor, admin)
- **Service Account:** An identity for your APPLICATION (not a human)

**Common roles for ML:**
| Role | What it grants |
|------|---------------|
| `roles/storage.objectViewer` | Read files from GCS |
| `roles/storage.objectAdmin` | Read/write/delete GCS files |
| `roles/bigquery.dataViewer` | Read BigQuery tables |
| `roles/bigquery.jobUser` | Run BigQuery queries |
| `roles/aiplatform.user` | Use Vertex AI services |
| `roles/run.invoker` | Call a Cloud Run service |

**Service accounts in code:**
```python
from google.cloud import storage

# Uses the default service account (automatic in GCP)
client = storage.Client()

# Or specify a service account key file (local development)
client = storage.Client.from_service_account_json("key.json")
```

**Interview tip:** Always mention "principle of least privilege" -
give only the minimum permissions needed.

---

## 7. Pub/Sub - Event-Driven Architecture

Pub/Sub is a messaging service. Useful for real-time ML pipelines.

```
[Data Producer] → publishes to → [Topic] → delivers to → [Subscriber]

Example:
[User clicks] → [click-events topic] → [ML pipeline that updates recommendations]
```

```python
from google.cloud import pubsub_v1

# Publish a message
publisher = pubsub_v1.PublisherClient()
topic_path = publisher.topic_path("my-project", "predictions")

data = '{"user_id": 123, "prediction": 0.95}'.encode("utf-8")
future = publisher.publish(topic_path, data)
print(f"Published: {future.result()}")

# Subscribe to messages
subscriber = pubsub_v1.SubscriberClient()
subscription_path = subscriber.subscription_path("my-project", "predictions-sub")

def callback(message):
    print(f"Received: {message.data}")
    message.ack()  # Acknowledge receipt

subscriber.subscribe(subscription_path, callback=callback)
```

---

## 8. Common Interview Architecture Questions

### "Design an ML pipeline on GCP"
```
BigQuery (raw data)
    → Dataflow / BigQuery SQL (preprocessing)
    → GCS (store processed data)
    → Vertex AI Training (train model)
    → Vertex AI Model Registry (store model)
    → Vertex AI Endpoint OR Cloud Run (serve predictions)
    → Cloud Monitoring (monitor drift & performance)
    → Pub/Sub (trigger retraining when drift detected)
```

### "How do you handle real-time predictions?"
```
Client → Cloud Run / Vertex AI Endpoint → Model → Response
         (auto-scales based on traffic)
```

### "How do you handle batch predictions?"
```
GCS (input data) → Vertex AI Batch Prediction → GCS (output predictions)
```

### "How do you monitor a model in production?"
- Vertex AI Model Monitoring for data drift and skew
- Cloud Logging for prediction logs
- Cloud Monitoring for latency and error rates
- Custom dashboard in Looker / Data Studio

---

## 9. GCP CLI Commands to Know

```bash
# Authentication
gcloud auth login
gcloud auth application-default login
gcloud config set project my-project

# GCS
gsutil ls gs://my-bucket/
gsutil cp local_file.csv gs://my-bucket/data/
gsutil cp gs://my-bucket/model.pkl ./

# BigQuery
bq query --use_legacy_sql=false 'SELECT COUNT(*) FROM `project.dataset.table`'
bq ls project:dataset

# Vertex AI
gcloud ai models list --region=us-central1
gcloud ai endpoints list --region=us-central1

# Cloud Run
gcloud run services list
gcloud run deploy SERVICE --image IMAGE --region REGION
```

---

## 10. Interview Vocabulary

| Term | Meaning |
|------|---------|
| **GCS** | Google Cloud Storage - object/file storage |
| **BigQuery** | Serverless data warehouse, SQL at scale |
| **Vertex AI** | Unified ML platform (train, deploy, monitor) |
| **Cloud Run** | Serverless container platform |
| **Pub/Sub** | Asynchronous messaging service |
| **IAM** | Identity and Access Management (permissions) |
| **Service Account** | Identity for applications (not humans) |
| **Artifact Registry** | Store Docker containers and packages |
| **Dataflow** | Apache Beam-based data processing (ETL) |
| **AutoML** | Train models without writing code |
| **Feature Store** | Centralized storage for ML features |
| **Model Monitoring** | Detect data drift and model degradation |
| **A/B Testing** | Route traffic between model versions |
| **Batch Prediction** | Run predictions on a large dataset at once |
| **Online Prediction** | Real-time predictions via API endpoint |
