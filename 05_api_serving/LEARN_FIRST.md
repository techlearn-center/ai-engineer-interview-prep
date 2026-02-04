# Learn: APIs & FastAPI for AI Engineers

Read this BEFORE attempting the problems. Run each example as you go.

---

## 1. What is an API?

An API (Application Programming Interface) is how programs talk to each other.

**Real-world analogy:** A restaurant menu is an API.
- You (the client) look at the menu (the API docs)
- You place an order (send a request)
- The kitchen (server) prepares your food (processes the request)
- The waiter brings your food back (returns a response)

**Why AI engineers need this:**
When you train a model, it sits on your laptop. To let anyone USE that model,
you wrap it in an API. Users send data in, your model runs, predictions come back.

```
User sends data --> [Your API Server] --> Model runs --> Prediction returned
```

---

## 2. HTTP Basics (5 minutes)

APIs use HTTP - the same protocol your browser uses. There are 4 main operations:

| Method | Purpose              | Example                        |
|--------|----------------------|--------------------------------|
| GET    | Read/fetch data      | Get list of models             |
| POST   | Send data / create   | Send features, get prediction  |
| PUT    | Update existing data | Update a model's weights       |
| DELETE | Remove data          | Delete a model                 |

Every HTTP request gets a **status code** back:
- `200` = OK (success)
- `400` = Bad Request (you sent wrong data)
- `404` = Not Found (the thing you asked for doesn't exist)
- `500` = Server Error (something broke on the server side)

---

## 3. What is FastAPI?

FastAPI is a Python framework for building APIs. It's the #1 choice for ML/AI
because:
- It's fast and simple
- Auto-generates documentation
- Validates input data automatically
- Supports async for handling many requests

### Install it:
```bash
pip install fastapi uvicorn
```

---

## 4. Your First API (try this now!)

Create a file called `my_first_api.py`:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def home():
    return {"message": "Hello! My API is running."}

@app.get("/health")
def health():
    return {"status": "healthy"}
```

Run it:
```bash
cd 05_api_serving
uvicorn my_first_api:app --reload
```

Now open your browser: http://127.0.0.1:8000
You'll see: `{"message": "Hello! My API is running."}`

Open http://127.0.0.1:8000/docs for auto-generated documentation!

Press `Ctrl+C` to stop the server.

**What just happened:**
- `FastAPI()` creates your app
- `@app.get("/")` means "when someone visits `/`, run this function"
- The function returns a Python dict, FastAPI converts it to JSON automatically

---

## 5. Accepting Data with POST

GET is for fetching. POST is for sending data TO the server.
This is what you use for ML predictions.

Update `my_first_api.py`:

```python
from fastapi import FastAPI

app = FastAPI()

@app.get("/health")
def health():
    return {"status": "healthy"}

@app.post("/predict")
def predict(features: list[float]):
    """A dummy prediction endpoint."""
    # Fake model: just sum the features
    prediction = sum(features)
    return {
        "prediction": prediction,
        "n_features": len(features)
    }
```

Restart the server and go to http://127.0.0.1:8000/docs
- Click on POST `/predict`
- Click "Try it out"
- Enter: `[1.0, 2.0, 3.0]`
- Click "Execute"
- You'll see the prediction!

---

## 6. Pydantic Models (Input Validation)

In production, you need to validate what users send you. Pydantic does this.

```python
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI()


# This defines WHAT the user must send
class PredictionRequest(BaseModel):
    features: list[float]           # required: list of numbers
    model_name: str = "default"     # optional: has a default value


# This defines WHAT we send back
class PredictionResponse(BaseModel):
    prediction: float
    model_name: str


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    # Access the validated data
    prediction = sum(request.features)

    return PredictionResponse(
        prediction=prediction,
        model_name=request.model_name,
    )
```

**Why Pydantic matters:**
- If someone sends `features: "hello"` instead of a list, FastAPI
  automatically returns a 400 error with a clear message
- You don't have to write any validation code yourself
- It documents your API automatically

---

## 7. Handling Errors

When things go wrong, return proper HTTP error codes:

```python
from fastapi import FastAPI, HTTPException

app = FastAPI()

models = {"linear": [0.5, 1.2], "logistic": [0.8, -0.3]}

@app.post("/predict/{model_name}")
def predict(model_name: str, features: list[float]):

    # Model not found? Return 404
    if model_name not in models:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{model_name}' not found"
        )

    weights = models[model_name]

    # Wrong number of features? Return 400
    if len(features) != len(weights):
        raise HTTPException(
            status_code=400,
            detail=f"Expected {len(weights)} features, got {len(features)}"
        )

    prediction = sum(w * f for w, f in zip(weights, features))
    return {"prediction": prediction}
```

---

## 8. Path Parameters vs Query Parameters vs Request Body

Three ways to send data to an API:

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Item(BaseModel):
    name: str
    price: float

# PATH parameter - part of the URL
# Example: GET /items/42
@app.get("/items/{item_id}")
def get_item(item_id: int):
    return {"item_id": item_id}

# QUERY parameter - after ? in the URL
# Example: GET /search?q=python&limit=10
@app.get("/search")
def search(q: str, limit: int = 10):
    return {"query": q, "limit": limit}

# REQUEST BODY - sent as JSON in the request
# Example: POST /items with {"name": "Widget", "price": 9.99}
@app.post("/items")
def create_item(item: Item):
    return {"created": item.name, "price": item.price}
```

---

## 9. Putting It All Together - ML Model Server

This is what a real ML inference API looks like:

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import math

app = FastAPI(title="My ML API")


# --- Data models ---
class PredictionRequest(BaseModel):
    features: list[float] = Field(..., min_length=1)
    model_name: str = "default"

class PredictionResponse(BaseModel):
    prediction: float
    confidence: float
    model_name: str


# --- Simple model storage ---
MODELS = {
    "default": np.array([0.5, 1.2, -0.3]),
}


# --- Endpoints ---
@app.get("/health")
def health():
    return {"status": "healthy"}


@app.get("/models")
def list_models():
    return {"models": list(MODELS.keys())}


@app.post("/predict", response_model=PredictionResponse)
def predict(req: PredictionRequest):
    if req.model_name not in MODELS:
        raise HTTPException(404, f"Model '{req.model_name}' not found")

    weights = MODELS[req.model_name]
    if len(req.features) != len(weights):
        raise HTTPException(400, f"Need {len(weights)} features, got {len(req.features)}")

    prediction = float(np.dot(weights, req.features))
    confidence = 1 / (1 + math.exp(-abs(prediction)))

    return PredictionResponse(
        prediction=prediction,
        confidence=confidence,
        model_name=req.model_name,
    )
```

---

## 10. Testing Your API (without running the server)

FastAPI has a `TestClient` for testing without starting a server:

```python
from fastapi.testclient import TestClient

client = TestClient(app)

# Test the health endpoint
response = client.get("/health")
print(response.status_code)  # 200
print(response.json())       # {"status": "healthy"}

# Test a prediction
response = client.post("/predict", json={
    "features": [1.0, 2.0, 3.0],
    "model_name": "default"
})
print(response.json())  # {"prediction": ..., "confidence": ..., ...}
```

This is exactly what the test file does! Now you understand what's being tested.

---

## Now Try the Problems

You're ready. Open `p1_fastapi_inference.py` and fill in the functions.
Run the tests to check:

```bash
pytest 05_api_serving/tests/test_p1_fastapi_inference.py -v
```

---

## Interview Vocabulary Cheat Sheet

| Term | Meaning |
|------|---------|
| **Endpoint** | A URL path that does something (e.g., `/predict`) |
| **Request** | Data sent FROM the client TO the server |
| **Response** | Data sent FROM the server TO the client |
| **JSON** | The data format used (like a Python dict) |
| **REST** | A style of API design using HTTP methods |
| **Pydantic** | Library for data validation and serialization |
| **Status code** | Number indicating success (200) or error (400, 404, 500) |
| **Middleware** | Code that runs before/after every request (auth, logging) |
| **Serialization** | Converting Python objects to JSON |
| **Uvicorn** | The server that runs your FastAPI app |
