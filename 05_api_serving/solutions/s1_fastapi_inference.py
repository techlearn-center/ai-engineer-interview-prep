"""
SOLUTIONS - FastAPI Inference API
===================================
Try to solve the problems yourself first!
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import math

app = FastAPI(title="ML Inference API")


# ---------------------
# Pydantic models
# ---------------------

class PredictionRequest(BaseModel):
    """
    Key insight: Pydantic validates types automatically.
    Field() adds extra validation like min_length.
    """
    features: list[float] = Field(..., min_length=1)
    model_name: str = "linear_regression"


class PredictionResponse(BaseModel):
    prediction: float
    model_name: str
    confidence: float = Field(..., ge=0, le=1)


# ---------------------
# Model Registry
# ---------------------

class ModelRegistry:
    """
    Key insight: This is a simplified version of what MLflow, BentoML, etc. do.
    In production, you'd load actual model artifacts (pickle, ONNX, etc.)
    """

    def __init__(self):
        self._models = {}

    def register(self, name: str, weights: list[float]):
        self._models[name] = np.array(weights)

    def predict(self, name: str, features: list[float]) -> float:
        if name not in self._models:
            raise KeyError(f"Model '{name}' not found")

        weights = self._models[name]
        features_arr = np.array(features)

        if len(features_arr) != len(weights):
            raise ValueError(
                f"Expected {len(weights)} features, got {len(features_arr)}"
            )

        return float(np.dot(weights, features_arr))

    def list_models(self) -> list[str]:
        return list(self._models.keys())


# Create global registry with sample models
registry = ModelRegistry()
registry.register("linear_regression", [0.5, 1.2, -0.3])
registry.register("sentiment_model", [0.8, -0.4, 0.6, 0.2])


# ---------------------
# API endpoints
# ---------------------

@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/models")
def list_models():
    return {"models": registry.list_models()}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """
    Key insight: Convert domain exceptions to HTTP status codes.
    KeyError -> 404 (not found)
    ValueError -> 400 (bad request)
    """
    try:
        prediction = registry.predict(request.model_name, request.features)
    except KeyError:
        raise HTTPException(status_code=404, detail=f"Model '{request.model_name}' not found")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Compute confidence using sigmoid of absolute prediction
    confidence = 1 / (1 + math.exp(-abs(prediction)))

    return PredictionResponse(
        prediction=prediction,
        model_name=request.model_name,
        confidence=confidence,
    )


@app.post("/models/{model_name}")
def register_model(model_name: str, weights: list[float]):
    registry.register(model_name, weights)
    return {"message": f"Model {model_name} registered", "n_features": len(weights)}
