"""
Problem 1: Building an ML Inference API with FastAPI
=====================================================
Difficulty: Medium

AI engineers need to serve models. FastAPI is the standard.
You'll be asked to build endpoints for model inference.

Run tests:
    pytest 05_api_serving/tests/test_p1_fastapi_inference.py -v
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np

app = FastAPI(title="ML Inference API")

# ---------------------
# TASK 1: Define Pydantic models for request/response validation
# ---------------------


class PredictionRequest(BaseModel):
    """
    Define a request model with:
        - features: list of floats (the input features)
        - model_name: string with default "linear_regression"

    Add validation: features must have at least 1 element.
    """
    # YOUR CODE HERE
    pass


class PredictionResponse(BaseModel):
    """
    Define a response model with:
        - prediction: float
        - model_name: str
        - confidence: float (between 0 and 1)
    """
    # YOUR CODE HERE
    pass


# ---------------------
# TASK 2: Implement a simple in-memory model registry
# ---------------------

class ModelRegistry:
    """
    A simple model registry that stores "models" (just weight vectors for now).

    Implement:
        - register(name, weights): store a model's weights
        - predict(name, features): compute dot product of weights and features
        - list_models(): return list of registered model names
    """

    def __init__(self):
        self._models = {}

    def register(self, name: str, weights: list[float]):
        # YOUR CODE HERE
        pass

    def predict(self, name: str, features: list[float]) -> float:
        """
        Compute prediction as dot product of weights and features.
        Raise KeyError if model not found.
        Raise ValueError if features length doesn't match weights length.
        """
        # YOUR CODE HERE
        pass

    def list_models(self) -> list[str]:
        # YOUR CODE HERE
        pass


# Create global registry with a sample model
registry = ModelRegistry()
registry.register("linear_regression", [0.5, 1.2, -0.3])
registry.register("sentiment_model", [0.8, -0.4, 0.6, 0.2])


# ---------------------
# TASK 3: Implement the API endpoints
# ---------------------

@app.get("/health")
def health_check():
    """Return {"status": "healthy"}"""
    # YOUR CODE HERE
    pass


@app.get("/models")
def list_models():
    """Return {"models": [...list of model names...]}"""
    # YOUR CODE HERE
    pass


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Make a prediction using the specified model.

    - Look up the model in the registry
    - Compute the prediction
    - Return PredictionResponse with confidence = sigmoid(abs(prediction))
    - Handle errors: 404 if model not found, 400 if feature mismatch

    Sigmoid: 1 / (1 + exp(-x))
    """
    # YOUR CODE HERE
    pass


@app.post("/models/{model_name}")
def register_model(model_name: str, weights: list[float]):
    """
    Register a new model with the given weights.
    Return {"message": "Model {model_name} registered", "n_features": len(weights)}
    """
    # YOUR CODE HERE
    pass
