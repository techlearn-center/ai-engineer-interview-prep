"""
SOLUTIONS - Vertex AI Pipeline
=================================
Try to solve the problems yourself first!
"""
from p3_vertex_ai_pipeline import TrainingConfig, ModelConfig, EndpointConfig


CONTAINER_URIS = {
    "training": {
        "tensorflow": "us-docker.pkg.dev/vertex-ai/training/tf-gpu.2-12:latest",
        "pytorch": "us-docker.pkg.dev/vertex-ai/training/pytorch-gpu.2-0:latest",
        "sklearn": "us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-3:latest",
    },
    "serving": {
        "tensorflow": "us-docker.pkg.dev/vertex-ai/prediction/tf2-cpu.2-12:latest",
        "pytorch": "us-docker.pkg.dev/vertex-ai/prediction/pytorch-cpu.2-0:latest",
        "sklearn": "us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-3:latest",
        "xgboost": "us-docker.pkg.dev/vertex-ai/prediction/xgboost-cpu.1-7:latest",
    },
}


def build_training_config(job_name, training_script, framework,
                          gpu_required=False, hyperparameters=None,
                          staging_bucket="gs://my-staging-bucket"):
    """
    Key insight: Know the pre-built container URIs.
    In an interview, you don't need to memorize exact URIs, but know they exist
    and explain the concept: "Vertex AI provides pre-built containers for
    common frameworks so you don't have to build your own Docker image."
    """
    if framework not in CONTAINER_URIS["training"]:
        raise ValueError(f"Unsupported framework: {framework}")

    return TrainingConfig(
        display_name=job_name,
        container_uri=CONTAINER_URIS["training"][framework],
        machine_type="n1-standard-8" if gpu_required else "n1-standard-4",
        accelerator_type="NVIDIA_TESLA_T4" if gpu_required else "",
        accelerator_count=1 if gpu_required else 0,
        args=hyperparameters or {},
        staging_bucket=staging_bucket,
    )


def build_model_config(model_name, model_version, artifact_uri,
                       framework, description=""):
    """
    Key insight: The model registry stores metadata about your model,
    not the model itself. The model files live in GCS (artifact_uri).
    Labels help you search and filter models later.
    """
    if framework not in CONTAINER_URIS["serving"]:
        raise ValueError(f"Unsupported framework: {framework}")

    return ModelConfig(
        display_name=f"{model_name}-{model_version}",
        artifact_uri=artifact_uri,
        serving_container_uri=CONTAINER_URIS["serving"][framework],
        description=description,
        labels={
            "model_name": model_name,
            "version": model_version,
            "framework": framework,
        },
    )


def build_endpoint_config(endpoint_name, model_ids, traffic_percentages,
                          high_availability=False):
    """
    Key insight: Traffic splitting is how you do A/B testing and canary deploys.
    - Canary: new model gets 5%, stable gets 95%
    - A/B test: model_a gets 50%, model_b gets 50%
    - Full rollout: new model gets 100%
    """
    if len(model_ids) != len(traffic_percentages):
        raise ValueError(
            f"model_ids ({len(model_ids)}) and traffic_percentages "
            f"({len(traffic_percentages)}) must have same length"
        )
    if sum(traffic_percentages) != 100:
        raise ValueError(
            f"Traffic percentages must sum to 100, got {sum(traffic_percentages)}"
        )

    traffic_split = dict(zip(model_ids, traffic_percentages))

    return EndpointConfig(
        display_name=endpoint_name,
        machine_type="n1-standard-8" if high_availability else "n1-standard-4",
        min_replicas=2 if high_availability else 1,
        max_replicas=10 if high_availability else 3,
        traffic_split=traffic_split,
    )


def design_ml_pipeline(pipeline_name, data_source, model_type, deploy=True):
    """
    Key insight: This is a SYSTEM DESIGN answer.
    In an interview, draw this on a whiteboard and explain each step.

    The pipeline follows MLOps best practices:
    1. Extract data from source
    2. Validate data quality
    3. Preprocess and split
    4. Train model
    5. Evaluate model
    6. (Optional) Register and deploy
    """
    steps = [
        {
            "step_name": "data_extraction",
            "component": "BigQuery",
            "inputs": [data_source],
            "outputs": ["raw_data"],
            "pipeline_name": pipeline_name,
        },
        {
            "step_name": "data_validation",
            "component": "Vertex AI",
            "inputs": ["raw_data"],
            "outputs": ["validated_data", "data_stats"],
            "pipeline_name": pipeline_name,
        },
        {
            "step_name": "data_preprocessing",
            "component": "Dataflow",
            "inputs": ["validated_data"],
            "outputs": ["train_data", "val_data", "test_data"],
            "pipeline_name": pipeline_name,
        },
        {
            "step_name": "model_training",
            "component": "Vertex AI Training",
            "inputs": ["train_data", "val_data", model_type],
            "outputs": ["trained_model", "metrics"],
            "pipeline_name": pipeline_name,
        },
        {
            "step_name": "model_evaluation",
            "component": "Vertex AI",
            "inputs": ["trained_model", "test_data"],
            "outputs": ["eval_metrics", "eval_report"],
            "pipeline_name": pipeline_name,
        },
    ]

    if deploy:
        steps.extend([
            {
                "step_name": "model_upload",
                "component": "Vertex AI Model Registry",
                "inputs": ["trained_model", "eval_metrics"],
                "outputs": ["registered_model"],
                "pipeline_name": pipeline_name,
            },
            {
                "step_name": "model_deployment",
                "component": "Vertex AI Endpoints",
                "inputs": ["registered_model"],
                "outputs": ["endpoint_url"],
                "pipeline_name": pipeline_name,
            },
        ])

    return steps
