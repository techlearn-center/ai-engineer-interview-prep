"""
SOLUTIONS - GCS Operations
=============================
Try to solve the problems yourself first!
"""
import json
import csv
import io

# Import mock classes (same as problem file)
from p1_gcs_operations import MockStorageClient, MockBucket, MockBlob


def upload_json_to_gcs(client, bucket_name, blob_path, data):
    """
    Key insight: JSON is just a string. Convert dict -> JSON string -> upload.
    In real GCP, this is identical except you use google.cloud.storage.Client.
    """
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    json_str = json.dumps(data)
    blob.upload_from_string(json_str)
    return f"gs://{bucket_name}/{blob_path}"


def download_json_from_gcs(client, bucket_name, blob_path):
    """Reverse of upload: download text -> parse JSON."""
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    text = blob.download_as_text()
    return json.loads(text)


def upload_csv_to_gcs(client, bucket_name, blob_path, rows):
    """
    Key insight: Use io.StringIO as an in-memory file.
    csv.DictWriter writes CSV from a list of dicts.
    """
    output = io.StringIO()
    if rows:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_path)
    blob.upload_from_string(output.getvalue())
    return f"gs://{bucket_name}/{blob_path}"


def list_model_artifacts(client, bucket_name, model_prefix="models/"):
    """Use list_blobs with prefix to find matching files."""
    bucket = client.bucket(bucket_name)
    blobs = bucket.list_blobs(prefix=model_prefix)
    result = [{"name": b.name, "size": b.size} for b in blobs]
    return sorted(result, key=lambda x: x["name"])


def organize_training_data(client, bucket_name, dataset_name, train_data, val_data, metadata):
    """
    Key insight: This is a common ML pattern - organize data in a consistent structure.
    Always store metadata alongside your data for reproducibility.
    """
    base_path = f"datasets/{dataset_name}"

    # Upload training CSV
    train_uri = upload_csv_to_gcs(client, bucket_name, f"{base_path}/train.csv", train_data)

    # Upload validation CSV
    val_uri = upload_csv_to_gcs(client, bucket_name, f"{base_path}/val.csv", val_data)

    # Augment and upload metadata
    metadata = {**metadata}  # copy to avoid mutating input
    metadata["train_samples"] = len(train_data)
    metadata["val_samples"] = len(val_data)
    metadata["dataset_name"] = dataset_name
    metadata_uri = upload_json_to_gcs(client, bucket_name, f"{base_path}/metadata.json", metadata)

    return {
        "train_uri": train_uri,
        "val_uri": val_uri,
        "metadata_uri": metadata_uri,
    }
