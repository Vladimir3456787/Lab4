import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, List, Any
from datetime import datetime
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest, REGISTRY
import prometheus_client

from .models import InferenceRequest, InferenceResponse, HealthResponse
from .feature_store import FeatureStore
from .monitoring import InferenceMetrics

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus метрики
REQUEST_COUNT = Counter(
    'inference_requests_total',
    'Total number of inference requests',
    ['model', 'status']
)

REQUEST_LATENCY = Histogram(
    'inference_request_latency_seconds',
    'Inference request latency in seconds',
    ['model']
)

ERROR_COUNT = Counter(
    'inference_errors_total',
    'Total number of inference errors',
    ['error_type']
)

# Глобальные переменные
feature_store = None
model = None
model_version = None
client = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Инициализация при старте и очистка при остановке"""
    global feature_store, model, model_version, client
    
    # Инициализация
    logger.info("Starting up inference service...")
    
    import joblib

    # Настройка MLflow
    mlflow_tracking_uri = "http://mlflow:5000"
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    client = MlflowClient()
    
    # Загрузка production модели
    model_name = "EcommerceRecommendationModel"
    model_stage = "Production"
    
    try:
        # Сначала пробуем загрузить из MLflow
        model = mlflow.pyfunc.load_model(
            f"models:/{model_name}/{model_stage}"
        )
        logger.info(f"✅ Production model loaded from MLflow: {model_name}")
        model_version = "from_mlflow"
        
    except Exception as e:
        logger.warning(f"MLflow model not found: {e}. Loading from local file...")
        
        # Если нет в MLflow, загружаем из файла задания 3
        model_path = Path("/app/artifacts/production_models/xgboost_model.joblib")
        if not model_path.exists():
            # Создаем простую заглушку-модель для демонстрации
            logger.warning("Creating dummy model for demonstration")
            from sklearn.ensemble import RandomForestClassifier
            import numpy as np
            
            # Создаем простую модель на случайных данных
            X_dummy = np.random.randn(100, 13)
            y_dummy = np.random.randint(0, 2, 100)
            model = RandomForestClassifier(n_estimators=10)
            model.fit(X_dummy, y_dummy)
            
            # Сохраняем модель
            joblib.dump(model, model_path)
            model_version = "dummy_model"
        else:
            # Загружаем существующую модель
            model = joblib.load(model_path)
            model_version = "local_file"
        
        logger.info(f"✅ Model loaded: {model_version}")

    # Инициализация Feature Store
    feature_store_path = Path("/app/local_data/user_features.csv")
    feature_store = FeatureStore(feature_store_path)
    logger.info("✅ Feature Store initialized")
    
    yield
    
    # Очистка при остановке
    logger.info("Shutting down inference service...")

# Создание FastAPI приложения
app = FastAPI(
    title="Ecommerce Recommendation Inference API",
    description="Production inference service for recommendation model",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware для логирования и метрик
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    
    try:
        response = await call_next(request)
        
        # Логирование метрик
        process_time = time.time() - start_time
        REQUEST_LATENCY.labels(model="production").observe(process_time)
        
        if response.status_code >= 400:
            REQUEST_COUNT.labels(model="production", status="error").inc()
        else:
            REQUEST_COUNT.labels(model="production", status="success").inc()
            
    except Exception as e:
        ERROR_COUNT.labels(error_type=type(e).__name__).inc()
        logger.error(f"Request failed: {str(e)}")
        raise
    
    return response

@app.get("/")
async def root():
    """Корневой endpoint"""
    return {
        "service": "Ecommerce Recommendation Inference API",
        "version": "1.0.0",
        "status": "operational",
        "model": "EcommerceRecommendationModel",
        "model_version": model_version
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        # Проверка модели
        test_features = np.zeros((1, 13))
        _ = model.predict(test_features)
        
        # Проверка Feature Store
        if not feature_store.is_available():
            raise Exception("Feature Store unavailable")
        
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            model_loaded=True,
            feature_store_available=True,
            model_version=model_version
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now().isoformat(),
            model_loaded=False,
            feature_store_available=False,
            error=str(e)
        )

@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    """
    Получение предсказания для пользователя
    
    - **user_id**: ID пользователя
    - **transaction_amount**: Сумма транзакции
    """
    start_time = time.time()
    
    try:
        # Получение фич из Feature Store
        logger.info(f"Getting features for user {request.user_id}")
        features = feature_store.get_user_features(request.user_id)
        
        if features is None:
            raise HTTPException(
                status_code=404,
                detail=f"User {request.user_id} not found in Feature Store"
            )
        
        # Добавление transaction_amount
        features['transaction_amount'] = request.transaction_amount
        
        # Преобразование в формат для модели
        feature_array = features.values.reshape(1, -1)
        
        # Инференс
        prediction_proba = model.predict(feature_array)
        prediction = int(prediction_proba[0] > 0.5)
        
        # Логирование
        latency = (time.time() - start_time) * 1000  # в мс
        logger.info(f"Prediction for user {request.user_id}: {prediction} (latency: {latency:.2f}ms)")
        
        return InferenceResponse(
            user_id=request.user_id,
            prediction=prediction,
            prediction_proba=float(prediction_proba[0]),
            model_version=model_version,
            latency_ms=latency,
            timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        ERROR_COUNT.labels(error_type="prediction_error").inc()
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )

@app.post("/predict_batch")
async def predict_batch(requests: List[InferenceRequest]):
    """Пакетное предсказание"""
    results = []
    
    for req in requests:
        try:
            response = await predict(req)
            results.append(response.dict())
        except Exception as e:
            results.append({
                "user_id": req.user_id,
                "error": str(e),
                "success": False
            })
    
    return {"results": results}

@app.get("/metrics")
async def get_metrics():
    """Endpoint для Prometheus метрик"""
    return generate_latest(REGISTRY)

@app.get("/model/info")
async def get_model_info():
    """Информация о загруженной модели"""
    try:
        model_info = client.get_model_version(
            name="EcommerceRecommendationModel",
            version=model_version
        )
        
        return {
            "model_name": model_info.name,
            "version": model_info.version,
            "stage": model_info.current_stage,
            "run_id": model_info.run_id,
            "created_at": model_info.creation_timestamp.isoformat(),
            "description": model_info.description
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)