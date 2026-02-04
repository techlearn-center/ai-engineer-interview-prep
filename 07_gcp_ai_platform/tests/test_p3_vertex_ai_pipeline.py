import sys
import os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from p3_vertex_ai_pipeline import (
    build_training_config,
    build_model_config,
    build_endpoint_config,
    design_ml_pipeline,
    TrainingConfig,
    ModelConfig,
    EndpointConfig,
)


class TestBuildTrainingConfig:
    def test_tensorflow_gpu(self):
        config = build_training_config(
            job_name="train-bert",
            training_script="train.py",
            framework="tensorflow",
            gpu_required=True,
        )
        assert config.display_name == "train-bert"
        assert "tf-gpu" in config.container_uri
        assert config.machine_type == "n1-standard-8"
        assert config.accelerator_type == "NVIDIA_TESLA_T4"
        assert config.accelerator_count == 1

    def test_sklearn_cpu(self):
        config = build_training_config(
            job_name="train-lr",
            training_script="train.py",
            framework="sklearn",
            gpu_required=False,
        )
        assert "sklearn" in config.container_uri
        assert config.machine_type == "n1-standard-4"
        assert config.accelerator_count == 0
        assert config.accelerator_type == ""

    def test_pytorch(self):
        config = build_training_config(
            job_name="train-llm",
            training_script="train.py",
            framework="pytorch",
            gpu_required=True,
        )
        assert "pytorch" in config.container_uri

    def test_unsupported_framework(self):
        with pytest.raises(ValueError, match="Unsupported"):
            build_training_config("job", "train.py", "julia")

    def test_hyperparameters(self):
        config = build_training_config(
            job_name="job",
            training_script="train.py",
            framework="sklearn",
            hyperparameters={"lr": 0.01, "epochs": 100},
        )
        assert config.args == {"lr": 0.01, "epochs": 100}

    def test_returns_dataclass(self):
        config = build_training_config("j", "t.py", "sklearn")
        assert isinstance(config, TrainingConfig)


class TestBuildModelConfig:
    def test_basic(self):
        config = build_model_config(
            model_name="churn-model",
            model_version="v2",
            artifact_uri="gs://bucket/models/v2/",
            framework="sklearn",
        )
        assert config.display_name == "churn-model-v2"
        assert "sklearn" in config.serving_container_uri
        assert config.artifact_uri == "gs://bucket/models/v2/"

    def test_labels(self):
        config = build_model_config("m", "v1", "gs://b/m", "tensorflow")
        assert config.labels["model_name"] == "m"
        assert config.labels["version"] == "v1"
        assert config.labels["framework"] == "tensorflow"

    def test_xgboost(self):
        config = build_model_config("m", "v1", "gs://b/m", "xgboost")
        assert "xgboost" in config.serving_container_uri

    def test_unsupported(self):
        with pytest.raises(ValueError):
            build_model_config("m", "v1", "gs://b", "unknown_framework")

    def test_returns_dataclass(self):
        config = build_model_config("m", "v1", "gs://b", "sklearn")
        assert isinstance(config, ModelConfig)


class TestBuildEndpointConfig:
    def test_basic(self):
        config = build_endpoint_config(
            endpoint_name="prod-endpoint",
            model_ids=["model_a", "model_b"],
            traffic_percentages=[80, 20],
        )
        assert config.display_name == "prod-endpoint"
        assert config.traffic_split == {"model_a": 80, "model_b": 20}

    def test_high_availability(self):
        config = build_endpoint_config(
            "ha-endpoint", ["m1"], [100], high_availability=True
        )
        assert config.min_replicas == 2
        assert config.max_replicas == 10
        assert config.machine_type == "n1-standard-8"

    def test_standard(self):
        config = build_endpoint_config("std", ["m1"], [100])
        assert config.min_replicas == 1
        assert config.max_replicas == 3

    def test_traffic_must_sum_100(self):
        with pytest.raises(ValueError):
            build_endpoint_config("e", ["m1", "m2"], [50, 30])

    def test_mismatched_lengths(self):
        with pytest.raises(ValueError):
            build_endpoint_config("e", ["m1", "m2"], [100])

    def test_canary_deployment(self):
        config = build_endpoint_config(
            "canary",
            ["stable", "canary"],
            [95, 5],
        )
        assert config.traffic_split["stable"] == 95
        assert config.traffic_split["canary"] == 5


class TestDesignMLPipeline:
    def test_basic_steps(self):
        steps = design_ml_pipeline(
            "my-pipeline", "project.dataset.table", "LOGISTIC_REG", deploy=False
        )
        assert len(steps) == 5
        step_names = [s["step_name"] for s in steps]
        assert "data_extraction" in step_names
        assert "data_validation" in step_names
        assert "data_preprocessing" in step_names
        assert "model_training" in step_names
        assert "model_evaluation" in step_names

    def test_with_deployment(self):
        steps = design_ml_pipeline(
            "my-pipeline", "project.dataset.table", "LINEAR_REG", deploy=True
        )
        assert len(steps) == 7
        step_names = [s["step_name"] for s in steps]
        assert "model_upload" in step_names
        assert "model_deployment" in step_names

    def test_step_structure(self):
        steps = design_ml_pipeline("p", "t", "m", deploy=False)
        for step in steps:
            assert "step_name" in step
            assert "component" in step
            assert "inputs" in step
            assert "outputs" in step
            assert "pipeline_name" in step

    def test_pipeline_name_in_steps(self):
        steps = design_ml_pipeline("my-pipeline", "t", "m", deploy=False)
        for step in steps:
            assert step["pipeline_name"] == "my-pipeline"

    def test_correct_components(self):
        steps = design_ml_pipeline("p", "t", "m", deploy=True)
        components = {s["step_name"]: s["component"] for s in steps}
        assert components["data_extraction"] == "BigQuery"
        assert components["data_preprocessing"] == "Dataflow"
        assert "Vertex AI" in components["model_training"]
