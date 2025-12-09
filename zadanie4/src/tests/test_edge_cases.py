import json
import numpy as np
from src.api.feature_store import FeatureStore
from pathlib import Path

def test_feature_store_edge_cases():
    """Тестирование граничных случаев Feature Store"""
    
    # Создание временного Feature Store
    store = FeatureStore(Path("test_features.csv"))
    
    # Тест 1: Несуществующий пользователь
    features = store.get_user_features(999999)
    assert features is not None  # Должен вернуть средние значения
    
    # Тест 2: Отрицательный user_id
    features = store.get_user_features(-1)
    assert features is not None
    
    # Тест 3: user_id = 0
    features = store.get_user_features(0)
    assert features is not None
    
    # Тест 4: Очень большой user_id
    features = store.get_user_features(10**9)
    assert features is not None
    
    # Тест 5: Проверка доступности
    assert store.is_available() == True
    
    # Тест 6: Получение статистики
    stats = store.get_stats()
    assert "total_users" in stats
    assert "features" in stats
    
    print("✅ Все тесты Feature Store пройдены")

def test_model_edge_cases():
    """Тестирование граничных случаев для модели"""
    import mlflow
    
    # Загрузка production модели
    model = mlflow.pyfunc.load_model(
        "models:/EcommerceRecommendationModel/Production"
    )
    
    test_cases = [
        # (features, description)
        (np.zeros((1, 13)), "Все нули"),
        (np.ones((1, 13)), "Все единицы"),
        (np.full((1, 13), 1000), "Большие значения"),
        (np.full((1, 13), -1000), "Отрицательные значения"),
        (np.random.randn(1, 13) * 100, "Случайные значения"),
    ]
    
    for features, description in test_cases:
        try:
            prediction = model.predict(features)
            print(f"✅ {description}: prediction = {prediction[0]:.4f}")
        except Exception as e:
            print(f"❌ {description}: Ошибка - {e}")

if __name__ == "__main__":
    test_feature_store_edge_cases()
    test_model_edge_cases()