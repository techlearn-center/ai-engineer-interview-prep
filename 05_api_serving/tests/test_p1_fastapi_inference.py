import sys
import os
import pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient
from p1_fastapi_inference import app, registry, ModelRegistry


client = TestClient(app)


class TestModelRegistry:
    def test_register_and_list(self):
        reg = ModelRegistry()
        reg.register("test_model", [1.0, 2.0])
        assert "test_model" in reg.list_models()

    def test_predict(self):
        reg = ModelRegistry()
        reg.register("test", [1.0, 2.0, 3.0])
        result = reg.predict("test", [1.0, 1.0, 1.0])
        assert result == 6.0  # 1*1 + 2*1 + 3*1

    def test_predict_unknown_model(self):
        reg = ModelRegistry()
        with pytest.raises(KeyError):
            reg.predict("unknown", [1.0])

    def test_predict_wrong_features(self):
        reg = ModelRegistry()
        reg.register("test", [1.0, 2.0])
        with pytest.raises(ValueError):
            reg.predict("test", [1.0, 2.0, 3.0])  # wrong length


class TestHealthEndpoint:
    def test_health(self):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}


class TestModelsEndpoint:
    def test_list_models(self):
        response = client.get("/models")
        assert response.status_code == 200
        data = response.json()
        assert "models" in data
        assert "linear_regression" in data["models"]


class TestPredictEndpoint:
    def test_predict_success(self):
        response = client.post("/predict", json={
            "features": [1.0, 2.0, 3.0],
            "model_name": "linear_regression"
        })
        assert response.status_code == 200
        data = response.json()
        assert "prediction" in data
        assert "confidence" in data
        assert 0 <= data["confidence"] <= 1

    def test_predict_unknown_model(self):
        response = client.post("/predict", json={
            "features": [1.0],
            "model_name": "nonexistent"
        })
        assert response.status_code == 404

    def test_predict_wrong_features(self):
        response = client.post("/predict", json={
            "features": [1.0],  # linear_regression expects 3 features
            "model_name": "linear_regression"
        })
        assert response.status_code == 400


class TestRegisterEndpoint:
    def test_register_new_model(self):
        response = client.post("/models/my_new_model", json=[1.0, 2.0])
        assert response.status_code == 200
        data = response.json()
        assert "my_new_model" in data["message"]
        assert data["n_features"] == 2
