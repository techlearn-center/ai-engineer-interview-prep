"""
Problem 3: Vertex AI Model Lifecycle
======================================
Difficulty: Medium -> Hard

This tests your understanding of the full ML lifecycle on GCP:
training config, model registry, endpoint deployment, and pipeline design.

Run tests:
    pytest 07_gcp_ai_platform/tests/test_p3_vertex_ai_pipeline.py -v
"""
from dataclasses import dataclass, field
from enum import Enum


class MachineType(Enum):
    """Common GCP machine types for ML."""
    N1_STANDARD_4 = "n1-standard-4"       # 4 vCPUs, 15GB RAM
    N1_STANDARD_8 = "n1-standard-8"       # 8 vCPUs, 30GB RAM
    N1_HIGHMEM_8 = "n1-highmem-8"         # 8 vCPUs, 52GB RAM
    A2_HIGHGPU_1G = "a2-highgpu-1g"       # 1 A100 GPU


class AcceleratorType(Enum):
    """GPU types available on GCP."""
    NVIDIA_TESLA_T4 = "NVIDIA_TESLA_T4"           # Budget GPU
    NVIDIA_TESLA_V100 = "NVIDIA_TESLA_V100"       # Mid-tier
    NVIDIA_A100 = "NVIDIA_A100_80GB"              # Top-tier


@dataclass
class TrainingConfig:
    """Configuration for a Vertex AI training job."""
    display_name: str = ""
    container_uri: str = ""
    machine_type: str = ""
    accelerator_type: str = ""
    accelerator_count: int = 0
    replica_count: int = 1
    args: dict = field(default_factory=dict)
    staging_bucket: str = ""


@dataclass
class ModelConfig:
    """Configuration for a deployed model."""
    display_name: str = ""
    artifact_uri: str = ""
    serving_container_uri: str = ""
    description: str = ""
    labels: dict = field(default_factory=dict)


@dataclass
class EndpointConfig:
    """Configuration for a Vertex AI endpoint."""
    display_name: str = ""
    machine_type: str = ""
    min_replicas: int = 1
    max_replicas: int = 1
    traffic_split: dict = field(default_factory=dict)


# ============================================================
# YOUR TASKS
# ============================================================


def build_training_config(
    job_name: str,
    training_script: str,
    framework: str,
    gpu_required: bool = False,
    hyperparameters: dict = None,
    staging_bucket: str = "gs://my-staging-bucket",
) -> TrainingConfig:
    """
    Build a Vertex AI training job configuration.

    Rules:
        - display_name: job_name
        - container_uri: Pick based on framework:
            "tensorflow" -> "us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-12:latest"
            "pytorch"    -> "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-0:latest"
            "sklearn"    -> "us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-3:latest"
            Otherwise    -> raise ValueError("Unsupported framework: {framework}")
        - machine_type:
            If gpu_required: "n1-standard-8"
            Else: "n1-standard-4"
        - accelerator_type:
            If gpu_required: "NVIDIA_TESLA_T4"
            Else: "" (empty string)
        - accelerator_count:
            If gpu_required: 1
            Else: 0
        - args: hyperparameters dict (or empty dict if None)
        - staging_bucket: as provided

    Return a TrainingConfig dataclass.
    """
    # YOUR CODE HERE
    pass


def build_model_config(
    model_name: str,
    model_version: str,
    artifact_uri: str,
    framework: str,
    description: str = "",
) -> ModelConfig:
    """
    Build a Vertex AI model configuration for the model registry.

    Rules:
        - display_name: "{model_name}-{model_version}" (e.g., "churn-model-v2")
        - artifact_uri: as provided (GCS path to model files)
        - serving_container_uri: Pick based on framework:
            "tensorflow" -> "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-12:latest"
            "pytorch"    -> "us-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.2-0:latest"
            "sklearn"    -> "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest"
            "xgboost"   -> "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-7:latest"
            Otherwise    -> raise ValueError
        - description: as provided
        - labels: {"model_name": model_name, "version": model_version, "framework": framework}

    Return a ModelConfig dataclass.
    """
    # YOUR CODE HERE
    pass


def build_endpoint_config(
    endpoint_name: str,
    model_ids: list[str],
    traffic_percentages: list[int],
    high_availability: bool = False,
) -> EndpointConfig:
    """
    Build a Vertex AI endpoint configuration for model serving.

    This is how you do A/B testing and canary deployments!

    Rules:
        - display_name: endpoint_name
        - machine_type:
            If high_availability: "n1-standard-8"
            Else: "n1-standard-4"
        - min_replicas:
            If high_availability: 2
            Else: 1
        - max_replicas:
            If high_availability: 10
            Else: 3
        - traffic_split: dict mapping model_id -> percentage
            e.g., {"model_1": 80, "model_2": 20}
        - Validate: traffic_percentages must sum to 100, raise ValueError if not
        - Validate: len(model_ids) must equal len(traffic_percentages), raise ValueError if not

    Return an EndpointConfig dataclass.
    """
    # YOUR CODE HERE
    pass


def design_ml_pipeline(
    pipeline_name: str,
    data_source: str,
    model_type: str,
    deploy: bool = True,
) -> list[dict]:
    """
    Design an ML pipeline as a list of steps.

    This is a SYSTEM DESIGN question - show you understand the full ML lifecycle.

    Return a list of pipeline step dicts, each with:
        - "step_name": descriptive name
        - "component": GCP service used
        - "inputs": list of input names/descriptions
        - "outputs": list of output names/descriptions

    Required steps (in order):
        1. "data_extraction" - component: "BigQuery"
           inputs: [data_source], outputs: ["raw_data"]

        2. "data_validation" - component: "Vertex AI"
           inputs: ["raw_data"], outputs: ["validated_data", "data_stats"]

        3. "data_preprocessing" - component: "Dataflow"
           inputs: ["validated_data"], outputs: ["train_data", "val_data", "test_data"]

        4. "model_training" - component: "Vertex AI Training"
           inputs: ["train_data", "val_data", model_type], outputs: ["trained_model", "metrics"]

        5. "model_evaluation" - component: "Vertex AI"
           inputs: ["trained_model", "test_data"], outputs: ["eval_metrics", "eval_report"]

    If deploy is True, also add:
        6. "model_upload" - component: "Vertex AI Model Registry"
           inputs: ["trained_model", "eval_metrics"], outputs: ["registered_model"]

        7. "model_deployment" - component: "Vertex AI Endpoints"
           inputs: ["registered_model"], outputs: ["endpoint_url"]

    Also add to each step:
        - "pipeline_name": the pipeline_name parameter

    Return the list of step dicts.
    """
    # YOUR CODE HERE
    pass
