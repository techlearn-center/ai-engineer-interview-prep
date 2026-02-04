# Hints - GCP AI Platform

## P1: GCS Operations

### upload_json_to_gcs
- `bucket = client.bucket(bucket_name)`
- `blob = bucket.blob(blob_path)`
- `json.dumps(data)` converts dict to JSON string
- `blob.upload_from_string(json_string)`
- Return `f"gs://{bucket_name}/{blob_path}"`

### download_json_from_gcs
- Same bucket/blob pattern
- `blob.download_as_text()` returns a string
- `json.loads(text)` converts JSON string back to dict

### upload_csv_to_gcs
- `output = io.StringIO()` - in-memory file
- `writer = csv.DictWriter(output, fieldnames=rows[0].keys())`
- `writer.writeheader()` then `writer.writerows(rows)`
- Upload `output.getvalue()` as string

### list_model_artifacts
- `bucket.list_blobs(prefix=model_prefix)` returns matching blobs
- Build dicts: `{"name": blob.name, "size": blob.size}`
- Sort by name: `sorted(result, key=lambda x: x["name"])`

### organize_training_data
- Reuse your upload_csv_to_gcs and upload_json_to_gcs functions!
- Augment metadata dict with train_samples, val_samples, dataset_name
- Base path: `f"datasets/{dataset_name}"`

## P2: BigQuery & BigQuery ML

### get_feature_stats
```sql
SELECT {group_col}, COUNT(*) AS count, AVG({feature_col}) AS avg_value
FROM `{table_id}`
GROUP BY {group_col}
ORDER BY count DESC
```
- Use an f-string to build the SQL
- `client.query(sql).to_dataframe()` runs it

### find_outliers
```sql
SELECT * FROM `{table_id}` WHERE {value_col} > {threshold} ORDER BY {value_col} DESC
```

### get_top_n
```sql
SELECT * FROM `{table_id}` ORDER BY {sort_col} DESC LIMIT {n}
```

### build_training_query
- Join feature_cols + [label_col] with commas for SELECT
- Wrap table_id in backticks
- Conditionally add TABLESAMPLE and WHERE clauses
- This returns a STRING, doesn't execute it

### build_bqml_create_model
- Pattern: `CREATE OR REPLACE MODEL \`{model_id}\` OPTIONS(...) AS SELECT ... FROM ...`
- OPTIONS must include: model_type, input_label_cols
- SELECT * if no feature_cols, otherwise list specific columns + label

## P3: Vertex AI Pipeline

### build_training_config
- Map framework to container URI with a dict/if-else
- GPU required → n1-standard-8 + NVIDIA_TESLA_T4
- No GPU → n1-standard-4, empty accelerator
- Return a TrainingConfig dataclass

### build_model_config
- display_name: f"{model_name}-{model_version}"
- Map framework to SERVING container URI (different from training!)
- Labels: dict with model_name, version, framework

### build_endpoint_config
- VALIDATE first: len(model_ids) == len(traffic_percentages), sum == 100
- traffic_split: `dict(zip(model_ids, traffic_percentages))`
- high_availability: more replicas, bigger machine

### design_ml_pipeline
- Return a list of dicts, one per step
- Each dict: step_name, component, inputs, outputs, pipeline_name
- 5 steps without deploy, 7 steps with deploy
- Follow the exact structure from the docstring
