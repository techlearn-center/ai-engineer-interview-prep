# GCP Cheatsheet for AI Engineers

Quick reference for the GCP services, CLI commands, and Python SDK patterns you need to know.

---

## Service Map (What to Use When)

| Task | GCP Service | Why |
|------|------------|-----|
| Store files (data, models) | **Cloud Storage (GCS)** | Cheap, scalable object storage |
| Query large datasets | **BigQuery** | Serverless SQL, handles petabytes |
| Train ML models | **Vertex AI Training** | Managed compute with GPUs |
| Store trained models | **Vertex AI Model Registry** | Version and track models |
| Serve predictions (managed) | **Vertex AI Endpoints** | Auto-scaling, A/B testing |
| Serve predictions (custom) | **Cloud Run** | Containerized APIs, scale to zero |
| Orchestrate ML pipelines | **Vertex AI Pipelines** | Kubeflow-based, managed |
| Stream data | **Pub/Sub** | Real-time messaging |
| Process data at scale | **Dataflow** | Apache Beam, batch + stream |
| Schedule jobs | **Cloud Scheduler** | Cron jobs in the cloud |
| Store secrets | **Secret Manager** | API keys, passwords |
| Manage permissions | **IAM** | Who can do what |
| Monitor everything | **Cloud Monitoring + Logging** | Alerts, dashboards, logs |

---

## GCS (Cloud Storage)

### CLI Commands
```bash
# List buckets
gsutil ls

# List files in a bucket
gsutil ls gs://my-bucket/data/

# Upload
gsutil cp local_file.csv gs://my-bucket/data/

# Download
gsutil cp gs://my-bucket/model.pkl ./local_model.pkl

# Copy between buckets
gsutil cp gs://bucket-a/file gs://bucket-b/file

# Sync a directory
gsutil -m rsync -r ./local_dir gs://my-bucket/remote_dir

# Delete
gsutil rm gs://my-bucket/old_file.csv
gsutil rm -r gs://my-bucket/old_folder/    # recursive
```

### Python SDK
```python
from google.cloud import storage

client = storage.Client(project="my-project")
bucket = client.bucket("my-bucket")

# Upload
blob = bucket.blob("data/train.csv")
blob.upload_from_filename("train.csv")

# Download
blob = bucket.blob("models/model.pkl")
blob.download_to_filename("model.pkl")

# Upload string/bytes
blob.upload_from_string(json.dumps(data), content_type="application/json")

# Read as text
text = blob.download_as_text()

# List files
for blob in client.list_blobs("my-bucket", prefix="data/"):
    print(blob.name, blob.size)
```

### URI Format
```
gs://bucket-name/path/to/file.csv
```

---

## BigQuery

### CLI Commands
```bash
# Run a query
bq query --use_legacy_sql=false 'SELECT COUNT(*) FROM `project.dataset.table`'

# List datasets
bq ls project:

# List tables in a dataset
bq ls project:dataset

# Export to GCS
bq extract project:dataset.table gs://bucket/export.csv
```

### Python SDK
```python
from google.cloud import bigquery

client = bigquery.Client(project="my-project")

# Run query → pandas DataFrame
query = """
SELECT user_id, COUNT(*) as orders, AVG(amount) as avg_amount
FROM `project.dataset.transactions`
WHERE date >= '2024-01-01'
GROUP BY user_id
ORDER BY orders DESC
LIMIT 1000
"""
df = client.query(query).to_dataframe()

# Load data from DataFrame
table_id = "project.dataset.new_table"
job = client.load_table_from_dataframe(df, table_id)
job.result()  # Wait for completion

# Load from GCS
job_config = bigquery.LoadJobConfig(source_format=bigquery.SourceFormat.CSV, skip_leading_rows=1)
uri = "gs://my-bucket/data.csv"
job = client.load_table_from_uri(uri, table_id, job_config=job_config)
```

### BigQuery ML
```sql
-- Train a model
CREATE OR REPLACE MODEL `project.dataset.my_model`
OPTIONS(model_type='LOGISTIC_REG', input_label_cols=['churned'])
AS SELECT * FROM `project.dataset.training_data`;

-- Predict
SELECT * FROM ML.PREDICT(MODEL `project.dataset.my_model`,
    (SELECT * FROM `project.dataset.new_data`));

-- Evaluate
SELECT * FROM ML.EVALUATE(MODEL `project.dataset.my_model`);

-- Feature importance
SELECT * FROM ML.FEATURE_IMPORTANCE(MODEL `project.dataset.my_model`);
```

**BQML Model Types:** `LOGISTIC_REG`, `LINEAR_REG`, `KMEANS`, `BOOSTED_TREE_CLASSIFIER`, `BOOSTED_TREE_REGRESSOR`, `DNN_CLASSIFIER`, `DNN_REGRESSOR`

---

## Vertex AI

### CLI Commands
```bash
# List models
gcloud ai models list --region=us-central1

# List endpoints
gcloud ai endpoints list --region=us-central1

# List training jobs
gcloud ai custom-jobs list --region=us-central1

# Submit training job
gcloud ai custom-jobs create \
    --region=us-central1 \
    --display-name=my-training \
    --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,container-image-uri=IMAGE
```

### Python SDK - Training
```python
from google.cloud import aiplatform

aiplatform.init(project="my-project", location="us-central1", staging_bucket="gs://my-bucket")

# Custom training job
job = aiplatform.CustomTrainingJob(
    display_name="my-job",
    script_path="train.py",
    container_uri="us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-3:latest",
    requirements=["pandas", "scikit-learn"],
)
model = job.run(
    replica_count=1,
    machine_type="n1-standard-4",
)
```

### Python SDK - Deploy
```python
# Upload model
model = aiplatform.Model.upload(
    display_name="my-model-v1",
    artifact_uri="gs://my-bucket/models/v1/",
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest",
)

# Deploy to endpoint
endpoint = model.deploy(
    machine_type="n1-standard-4",
    min_replica_count=1,
    max_replica_count=5,
    traffic_percentage=100,
)

# Predict
result = endpoint.predict(instances=[[1.0, 2.0, 3.0]])
```

### Pre-built Container URIs
```
# Training
us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-12:latest        # TensorFlow
us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-0:latest     # PyTorch
us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-3:latest     # Scikit-learn

# Serving
us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-12:latest      # TensorFlow
us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest   # Scikit-learn
us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-7:latest   # XGBoost
```

---

## Cloud Run

### Deploy a Container
```bash
# Build container
gcloud builds submit --tag gcr.io/PROJECT/IMAGE_NAME

# Deploy
gcloud run deploy SERVICE_NAME \
    --image gcr.io/PROJECT/IMAGE_NAME \
    --platform managed \
    --region us-central1 \
    --memory 2Gi \
    --cpu 2 \
    --min-instances 0 \
    --max-instances 10 \
    --allow-unauthenticated

# Update
gcloud run services update SERVICE_NAME --memory 4Gi
```

### Dockerfile for ML API
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
```

**Key:** Cloud Run requires port 8080 (set via `PORT` env var).

---

## IAM (Permissions)

### Key Roles for ML
```
roles/storage.objectViewer       # Read GCS files
roles/storage.objectAdmin        # Read/write/delete GCS files
roles/bigquery.dataViewer        # Read BigQuery tables
roles/bigquery.jobUser           # Run BigQuery queries
roles/bigquery.dataEditor        # Write BigQuery tables
roles/aiplatform.user            # Use Vertex AI
roles/run.invoker                # Call Cloud Run services
roles/secretmanager.secretAccessor  # Read secrets
```

### Service Accounts
```bash
# Create
gcloud iam service-accounts create ml-pipeline-sa \
    --display-name="ML Pipeline Service Account"

# Grant role
gcloud projects add-iam-policy-binding PROJECT \
    --member="serviceAccount:ml-pipeline-sa@PROJECT.iam.gserviceaccount.com" \
    --role="roles/storage.objectAdmin"
```

**Interview tip:** Always mention "principle of least privilege."

---

## Common Architecture Patterns

### Real-Time Prediction
```
Client → Cloud Run (FastAPI) → Model (in memory) → Response
         or
Client → Vertex AI Endpoint → Model → Response
```

### Batch Prediction
```
GCS (input) → Vertex AI Batch Prediction → GCS (output)
```

### ML Pipeline
```
BigQuery → Dataflow → GCS → Vertex AI Training → Model Registry → Endpoint
                              ↑
                    Cloud Scheduler (retrain weekly)
```

### Event-Driven ML
```
Pub/Sub → Cloud Function → Vertex AI Endpoint → Pub/Sub → Downstream
```

---

## Authentication

```bash
# Login (human)
gcloud auth login
gcloud auth application-default login

# Set project
gcloud config set project my-project

# In code (auto-detects credentials)
from google.cloud import storage
client = storage.Client()  # Uses GOOGLE_APPLICATION_CREDENTIALS env var

# Local dev with key file
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"
```
