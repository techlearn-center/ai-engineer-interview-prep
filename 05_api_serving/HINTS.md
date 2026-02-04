# Hints - API / Model Serving

## P1: FastAPI Inference

### PredictionRequest (Pydantic model)
```python
class PredictionRequest(BaseModel):
    features: list[float] = Field(..., min_length=1)
    model_name: str = "linear_regression"
```
- `Field(...)` means required, `Field(..., min_length=1)` adds validation
- Default values work just like regular Python function defaults

### PredictionResponse
```python
class PredictionResponse(BaseModel):
    prediction: float
    model_name: str
    confidence: float = Field(..., ge=0, le=1)  # ge=greater or equal, le=less or equal
```

### ModelRegistry.predict
- Use `np.dot(weights, features)` for the prediction
- Raise `KeyError` if model name not in `self._models`
- Raise `ValueError` if `len(features) != len(weights)`

### Endpoints
- `@app.get("/health")` -> return `{"status": "healthy"}`
- `@app.get("/models")` -> return `{"models": registry.list_models()}`
- `@app.post("/predict")` -> try/except to catch KeyError (-> 404) and ValueError (-> 400)
- Use `HTTPException(status_code=404, detail="message")` for errors

### Confidence via sigmoid
```python
import math
confidence = 1 / (1 + math.exp(-abs(prediction)))
```

### Key interview talking points
- FastAPI auto-generates OpenAPI docs at `/docs`
- Pydantic handles validation and serialization
- In production: add rate limiting, authentication, model versioning
- Consider async endpoints for I/O-bound inference
