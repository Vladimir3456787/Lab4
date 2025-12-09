from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class InferenceRequest(BaseModel):
    """Модель запроса для инференса"""
    user_id: int = Field(..., description="ID пользователя", example=12345)
    transaction_amount: float = Field(
        ..., 
        description="Сумма текущей транзакции",
        example=299.99,
        ge=0.0
    )

class InferenceResponse(BaseModel):
    """Модель ответа инференса"""
    user_id: int
    prediction: int = Field(..., description="Бинарное предсказание (0/1)")
    prediction_proba: float = Field(
        ..., 
        description="Вероятность положительного класса",
        ge=0.0,
        le=1.0
    )
    model_version: str
    latency_ms: float
    timestamp: str

class HealthResponse(BaseModel):
    """Модель ответа health check"""
    status: str  # healthy/unhealthy
    timestamp: str
    model_loaded: bool = True
    feature_store_available: bool = True
    model_version: Optional[str] = None
    error: Optional[str] = None

class ModelMetrics(BaseModel):
    """Модель метрик производительности"""
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    throughput_rps: float
    error_rate: float
    timestamp: str