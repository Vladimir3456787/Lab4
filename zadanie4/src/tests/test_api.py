import pytest
from fastapi.testclient import TestClient
import numpy as np
from src.api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_version" in data

def test_predict_endpoint():
    test_data = {
        "user_id": 123,
        "transaction_amount": 299.99
    }
    
    response = client.post("/predict", json=test_data)
    assert response.status_code == 200
    
    data = response.json()
    assert "prediction" in data
    assert "prediction_proba" in data
    assert "latency_ms" in data
    assert data["prediction"] in [0, 1]
    assert 0 <= data["prediction_proba"] <= 1

def test_predict_invalid_user():
    test_data = {
        "user_id": 999999,  # Несуществующий пользователь
        "transaction_amount": 100.00
    }
    
    response = client.post("/predict", json=test_data)
    # Может вернуть 404 или использовать средние фичи
    assert response.status_code in [200, 404]

def test_predict_negative_amount():
    test_data = {
        "user_id": 123,
        "transaction_amount": -100.00
    }
    
    response = client.post("/predict", json=test_data)
    # Pydantic должен валидировать и вернуть 422
    assert response.status_code == 422

def test_batch_predict():
    test_data = [
        {"user_id": 1, "transaction_amount": 50.00},
        {"user_id": 2, "transaction_amount": 150.00},
        {"user_id": 999999, "transaction_amount": 200.00}  # Ошибка ожидается
    ]
    
    response = client.post("/predict_batch", json=test_data)
    assert response.status_code == 200
    
    data = response.json()
    assert len(data["results"]) == 3
    assert any("error" in r for r in data["results"])

def test_metrics_endpoint():
    response = client.get("/metrics")
    assert response.status_code == 200
    # Prometheus метрики в текстовом формате
    assert "inference_requests_total" in response.text

def test_model_info():
    response = client.get("/model/info")
    assert response.status_code == 200
    
    data = response.json()
    assert "model_name" in data
    assert "version" in data
    assert "stage" in data
    assert data["model_name"] == "EcommerceRecommendationModel"

if __name__ == "__main__":
    pytest.main([__file__, "-v"])