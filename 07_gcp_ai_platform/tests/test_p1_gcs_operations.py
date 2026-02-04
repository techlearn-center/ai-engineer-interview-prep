import sys
import os
import json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from p1_gcs_operations import (
    MockStorageClient,
    upload_json_to_gcs,
    download_json_from_gcs,
    upload_csv_to_gcs,
    list_model_artifacts,
    organize_training_data,
)


class TestUploadJson:
    def test_returns_uri(self):
        client = MockStorageClient()
        uri = upload_json_to_gcs(client, "my-bucket", "config.json", {"lr": 0.01})
        assert uri == "gs://my-bucket/config.json"

    def test_data_stored(self):
        client = MockStorageClient()
        data = {"model": "linear", "epochs": 100}
        upload_json_to_gcs(client, "my-bucket", "config.json", data)
        blob = client.bucket("my-bucket").blob("config.json")
        stored = json.loads(blob.download_as_text())
        assert stored == data

    def test_nested_path(self):
        client = MockStorageClient()
        uri = upload_json_to_gcs(client, "bucket", "a/b/c.json", {"x": 1})
        assert uri == "gs://bucket/a/b/c.json"


class TestDownloadJson:
    def test_roundtrip(self):
        client = MockStorageClient()
        original = {"name": "test", "values": [1, 2, 3]}
        upload_json_to_gcs(client, "bucket", "data.json", original)
        result = download_json_from_gcs(client, "bucket", "data.json")
        assert result == original


class TestUploadCsv:
    def test_returns_uri(self):
        client = MockStorageClient()
        rows = [{"a": 1, "b": 2}]
        uri = upload_csv_to_gcs(client, "bucket", "data.csv", rows)
        assert uri == "gs://bucket/data.csv"

    def test_csv_content(self):
        client = MockStorageClient()
        rows = [{"name": "Alice", "score": "90"}, {"name": "Bob", "score": "85"}]
        upload_csv_to_gcs(client, "bucket", "data.csv", rows)
        blob = client.bucket("bucket").blob("data.csv")
        content = blob.download_as_text()
        assert "name" in content  # header row
        assert "Alice" in content
        assert "Bob" in content

    def test_multiple_rows(self):
        client = MockStorageClient()
        rows = [{"x": str(i)} for i in range(10)]
        upload_csv_to_gcs(client, "bucket", "data.csv", rows)
        blob = client.bucket("bucket").blob("data.csv")
        lines = blob.download_as_text().strip().split("\n")
        assert len(lines) == 11  # 1 header + 10 data rows


class TestListModelArtifacts:
    def test_lists_files(self):
        client = MockStorageClient()
        bucket = client.bucket("bucket")
        blob1 = bucket.blob("models/v1/model.pkl")
        blob1.upload_from_string(b"x" * 100)
        blob2 = bucket.blob("models/v1/config.json")
        blob2.upload_from_string(b"y" * 50)

        result = list_model_artifacts(client, "bucket", "models/v1/")
        assert len(result) == 2
        assert result[0]["name"] == "models/v1/config.json"
        assert result[1]["name"] == "models/v1/model.pkl"

    def test_includes_size(self):
        client = MockStorageClient()
        bucket = client.bucket("bucket")
        blob = bucket.blob("models/m.pkl")
        blob.upload_from_string(b"x" * 200)

        result = list_model_artifacts(client, "bucket", "models/")
        assert result[0]["size"] == 200

    def test_empty(self):
        client = MockStorageClient()
        result = list_model_artifacts(client, "bucket", "models/")
        assert result == []


class TestOrganizeTrainingData:
    def test_returns_uris(self):
        client = MockStorageClient()
        train = [{"f1": "1", "label": "0"}, {"f1": "2", "label": "1"}]
        val = [{"f1": "3", "label": "0"}]
        meta = {"version": "1.0"}

        result = organize_training_data(client, "bucket", "my_dataset", train, val, meta)

        assert result["train_uri"] == "gs://bucket/datasets/my_dataset/train.csv"
        assert result["val_uri"] == "gs://bucket/datasets/my_dataset/val.csv"
        assert result["metadata_uri"] == "gs://bucket/datasets/my_dataset/metadata.json"

    def test_metadata_augmented(self):
        client = MockStorageClient()
        train = [{"x": "1"}, {"x": "2"}, {"x": "3"}]
        val = [{"x": "4"}]
        meta = {"version": "2.0"}

        organize_training_data(client, "bucket", "ds1", train, val, meta)

        blob = client.bucket("bucket").blob("datasets/ds1/metadata.json")
        stored_meta = json.loads(blob.download_as_text())
        assert stored_meta["train_samples"] == 3
        assert stored_meta["val_samples"] == 1
        assert stored_meta["dataset_name"] == "ds1"
        assert stored_meta["version"] == "2.0"

    def test_train_csv_exists(self):
        client = MockStorageClient()
        train = [{"a": "1"}]
        val = [{"a": "2"}]

        organize_training_data(client, "bucket", "ds", train, val, {})

        blob = client.bucket("bucket").blob("datasets/ds/train.csv")
        content = blob.download_as_text()
        assert "a" in content  # header
