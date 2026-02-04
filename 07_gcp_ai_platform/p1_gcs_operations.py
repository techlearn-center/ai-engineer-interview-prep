"""
Problem 1: Google Cloud Storage Operations
============================================
Difficulty: Easy -> Medium

GCS is the foundation - all data and models live here.
This uses a MOCK client so you can run tests without a GCP account.

Run tests:
    pytest 07_gcp_ai_platform/tests/test_p1_gcs_operations.py -v
"""
import json
import csv
import io


class MockBlob:
    """Simulates a GCS blob (file)."""
    def __init__(self, name: str, data: bytes = b""):
        self.name = name
        self.data = data
        self.size = len(data)

    def upload_from_string(self, data: str | bytes):
        if isinstance(data, str):
            data = data.encode("utf-8")
        self.data = data
        self.size = len(data)

    def download_as_text(self) -> str:
        return self.data.decode("utf-8")


class MockBucket:
    """Simulates a GCS bucket."""
    def __init__(self, name: str):
        self.name = name
        self._blobs: dict[str, MockBlob] = {}

    def blob(self, name: str) -> MockBlob:
        if name not in self._blobs:
            self._blobs[name] = MockBlob(name)
        return self._blobs[name]

    def list_blobs(self, prefix: str = "") -> list[MockBlob]:
        return [b for name, b in self._blobs.items() if name.startswith(prefix)]


class MockStorageClient:
    """Simulates google.cloud.storage.Client."""
    def __init__(self):
        self._buckets: dict[str, MockBucket] = {}

    def bucket(self, name: str) -> MockBucket:
        if name not in self._buckets:
            self._buckets[name] = MockBucket(name)
        return self._buckets[name]


# ============================================================
# YOUR TASKS: Implement these functions using the mock client.
# They mirror exactly what you'd do with the real GCP SDK.
# ============================================================


def upload_json_to_gcs(client: MockStorageClient, bucket_name: str,
                       blob_path: str, data: dict) -> str:
    """
    Upload a Python dict as a JSON file to GCS.

    Steps:
        1. Get the bucket from the client
        2. Get/create a blob at the given path
        3. Convert the dict to a JSON string
        4. Upload the JSON string to the blob

    Return the GCS URI: "gs://{bucket_name}/{blob_path}"

    Example:
        upload_json_to_gcs(client, "my-bucket", "configs/model.json", {"lr": 0.01})
        -> "gs://my-bucket/configs/model.json"
    """
    # YOUR CODE HERE
    pass


def download_json_from_gcs(client: MockStorageClient, bucket_name: str,
                           blob_path: str) -> dict:
    """
    Download a JSON file from GCS and return it as a Python dict.

    Steps:
        1. Get the bucket
        2. Get the blob
        3. Download as text
        4. Parse JSON and return dict
    """
    # YOUR CODE HERE
    pass


def upload_csv_to_gcs(client: MockStorageClient, bucket_name: str,
                      blob_path: str, rows: list[dict]) -> str:
    """
    Upload a list of dicts as a CSV file to GCS.

    Each dict represents a row. Keys are column headers.
    Use csv.DictWriter with io.StringIO to build the CSV in memory.

    Return the GCS URI.

    Example:
        rows = [{"name": "Alice", "score": 90}, {"name": "Bob", "score": 85}]
        upload_csv_to_gcs(client, "bucket", "data/scores.csv", rows)
        -> "gs://bucket/data/scores.csv"
    """
    # YOUR CODE HERE
    pass


def list_model_artifacts(client: MockStorageClient, bucket_name: str,
                         model_prefix: str = "models/") -> list[dict]:
    """
    List all model artifacts under a prefix in GCS.

    Return a list of dicts with:
        - name: blob name
        - size: blob size in bytes

    Sorted by name.

    Example:
        list_model_artifacts(client, "my-bucket", "models/v1/")
        -> [{"name": "models/v1/model.pkl", "size": 1024}, ...]
    """
    # YOUR CODE HERE
    pass


def organize_training_data(client: MockStorageClient, bucket_name: str,
                           dataset_name: str,
                           train_data: list[dict],
                           val_data: list[dict],
                           metadata: dict) -> dict:
    """
    Organize training data in GCS following ML best practices.

    Create this structure:
        datasets/{dataset_name}/train.csv
        datasets/{dataset_name}/val.csv
        datasets/{dataset_name}/metadata.json

    The metadata dict should be augmented with:
        - "train_samples": number of training rows
        - "val_samples": number of validation rows
        - "dataset_name": the dataset name

    Return dict with URIs:
        {
            "train_uri": "gs://bucket/datasets/name/train.csv",
            "val_uri": "gs://bucket/datasets/name/val.csv",
            "metadata_uri": "gs://bucket/datasets/name/metadata.json",
        }
    """
    # YOUR CODE HERE
    pass
