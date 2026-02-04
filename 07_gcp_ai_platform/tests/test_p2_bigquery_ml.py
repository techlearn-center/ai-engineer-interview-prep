import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from p2_bigquery_ml import (
    MockBigQueryClient,
    get_feature_stats,
    find_outliers,
    get_top_n,
    build_training_query,
    build_bqml_create_model,
)


class TestGetFeatureStats:
    def setup_method(self):
        self.client = MockBigQueryClient()
        self.client.load_table("project.data.users", [
            {"country": "US", "age": 30},
            {"country": "US", "age": 40},
            {"country": "UK", "age": 25},
            {"country": "US", "age": 35},
            {"country": "UK", "age": 30},
        ])

    def test_returns_results(self):
        result = get_feature_stats(self.client, "project.data.users", "age", "country")
        assert len(result) == 2

    def test_correct_counts(self):
        result = get_feature_stats(self.client, "project.data.users", "age", "country")
        us_row = next(r for r in result if r["country"] == "US")
        assert us_row["count"] == 3

    def test_sorted_by_count_desc(self):
        result = get_feature_stats(self.client, "project.data.users", "age", "country")
        assert result[0]["count"] >= result[1]["count"]


class TestFindOutliers:
    def setup_method(self):
        self.client = MockBigQueryClient()
        self.client.load_table("project.data.metrics", [
            {"endpoint": "/api/v1", "latency": 50},
            {"endpoint": "/api/v2", "latency": 2000},
            {"endpoint": "/api/v3", "latency": 150},
            {"endpoint": "/api/v4", "latency": 5000},
        ])

    def test_finds_outliers(self):
        result = find_outliers(self.client, "project.data.metrics", "latency", 1000)
        assert len(result) == 2

    def test_sorted_desc(self):
        result = find_outliers(self.client, "project.data.metrics", "latency", 100)
        latencies = [r["latency"] for r in result]
        assert latencies == sorted(latencies, reverse=True)

    def test_no_outliers(self):
        result = find_outliers(self.client, "project.data.metrics", "latency", 10000)
        assert len(result) == 0


class TestGetTopN:
    def setup_method(self):
        self.client = MockBigQueryClient()
        self.client.load_table("project.data.scores", [
            {"name": "Alice", "score": 90},
            {"name": "Bob", "score": 85},
            {"name": "Charlie", "score": 95},
            {"name": "Diana", "score": 70},
        ])

    def test_top_2(self):
        result = get_top_n(self.client, "project.data.scores", "score", n=2)
        assert len(result) == 2
        assert result[0]["name"] == "Charlie"

    def test_sorted_desc(self):
        result = get_top_n(self.client, "project.data.scores", "score", n=4)
        scores = [r["score"] for r in result]
        assert scores == sorted(scores, reverse=True)


class TestBuildTrainingQuery:
    def test_basic(self):
        sql = build_training_query(
            "project.dataset.features",
            ["age", "income"],
            "churned",
        )
        assert "age" in sql
        assert "income" in sql
        assert "churned" in sql
        assert "`project.dataset.features`" in sql

    def test_with_where(self):
        sql = build_training_query(
            "project.dataset.features",
            ["age"],
            "label",
            where_clause="age > 18",
        )
        assert "WHERE" in sql.upper()
        assert "age > 18" in sql

    def test_with_sample(self):
        sql = build_training_query(
            "project.dataset.features",
            ["age"],
            "label",
            sample_pct=10,
        )
        assert "TABLESAMPLE" in sql.upper()
        assert "10" in sql

    def test_select_columns(self):
        sql = build_training_query(
            "t",
            ["f1", "f2", "f3"],
            "target",
        )
        # All columns should be in the SELECT
        assert "f1" in sql
        assert "f2" in sql
        assert "f3" in sql
        assert "target" in sql


class TestBuildBQMLCreateModel:
    def test_basic_structure(self):
        sql = build_bqml_create_model(
            "project.dataset.my_model",
            "LOGISTIC_REG",
            "project.dataset.data",
            "churned",
        )
        assert "CREATE OR REPLACE MODEL" in sql.upper()
        assert "`project.dataset.my_model`" in sql
        assert "LOGISTIC_REG" in sql
        assert "churned" in sql

    def test_with_feature_cols(self):
        sql = build_bqml_create_model(
            "p.d.m",
            "LINEAR_REG",
            "p.d.t",
            "price",
            feature_cols=["sqft", "bedrooms"],
        )
        assert "sqft" in sql
        assert "bedrooms" in sql
        assert "price" in sql

    def test_select_all_when_no_features(self):
        sql = build_bqml_create_model(
            "p.d.m",
            "KMEANS",
            "p.d.t",
            "cluster",
        )
        assert "*" in sql

    def test_with_options(self):
        sql = build_bqml_create_model(
            "p.d.m",
            "BOOSTED_TREE_CLASSIFIER",
            "p.d.t",
            "label",
            options={"max_iterations": 20},
        )
        assert "max_iterations" in sql
        assert "20" in sql
