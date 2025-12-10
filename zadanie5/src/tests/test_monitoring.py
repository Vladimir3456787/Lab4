import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import json

from src.monitoring.data_monitor import DataMonitor
from src.monitoring.model_monitor import ModelMonitor
from src.monitoring.alerting import AlertManager

@pytest.fixture
def sample_data():
    """Фикстура с тестовыми данными"""
    np.random.seed(42)
    n_samples = 1000
    
    reference_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.exponential(1, n_samples),
        'feature3': np.random.randint(0, 10, n_samples),
        'target': np.random.choice([0, 1], n_samples, p=[0.9, 0.1])
    })
    
    # Текущие данные с дрейфом
    current_data = pd.DataFrame({
        'feature1': np.random.normal(0.5, 1.2, n_samples),  # Сдвиг среднего
        'feature2': np.random.exponential(1.5, n_samples),  # Сдвиг масштаба
        'feature3': np.random.randint(2, 12, n_samples),    # Сдвиг диапазона
        'target': np.random.choice([0, 1], n_samples, p=[0.85, 0.15])  # Сдвиг распределения
    })
    
    return reference_data, current_data

def test_data_monitor_initialization():
    """Тест инициализации DataMonitor"""
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        tmp_path = Path(tmp.name)
    
    monitor = DataMonitor(tmp_path)
    
    # Проверка загрузки эталонных данных
    assert monitor.reference_data is None  # Файл пустой
    
    # Проверка порогов
    assert monitor.drift_thresholds['psi'] == 0.2
    assert monitor.drift_thresholds['missing_values'] == 0.1
    
    tmp_path.unlink()

def test_data_drift_detection(sample_data):
    """Тест обнаружения дрейфа данных"""
    reference_data, current_data = sample_data
    
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        tmp_path = Path(tmp.name)
        reference_data.to_csv(tmp_path, index=False)
    
    monitor = DataMonitor(tmp_path)
    result = monitor.monitor_data_drift(current_data, "test")
    
    # Проверка структуры результата
    assert 'status' in result
    assert 'metrics' in result
    assert 'alerts' in result
    
    # Проверка метрик
    assert 'drift_score' in result['metrics']
    assert 'drifted_columns' in result['metrics']
    
    # Должен быть обнаружен дрейф
    assert result['metrics']['drift_score'] > 0
    
    tmp_path.unlink()

def test_model_monitor_performance():
    """Тест мониторинга производительности модели"""
    monitor = ModelMonitor()
    
    # Тестовые данные
    n_samples = 500
    X_test = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.exponential(1, n_samples)
    })
    
    y_true = pd.Series(np.random.choice([0, 1], n_samples, p=[0.95, 0.05]))
    y_pred = y_true.copy()
    
    # Добавляем ошибки
    error_indices = np.random.choice(n_samples, size=50, replace=False)
    y_pred.iloc[error_indices] = 1 - y_pred.iloc[error_indices]
    
    y_pred_proba = pd.Series(np.where(y_pred == 1, 0.9, 0.1))
    
    # Мониторинг производительности
    result = monitor.monitor_model_performance(
        X_test, y_true, y_pred, y_pred_proba, "test"
    )
    
    # Проверка структуры результата
    assert 'status' in result
    assert 'performance_metrics' in result
    assert 'drift_metrics' in result
    
    # Проверка метрик производительности
    metrics = result['performance_metrics']
    assert 0 <= metrics['accuracy'] <= 1
    assert 0 <= metrics['precision'] <= 1
    assert 0 <= metrics['recall'] <= 1
    assert 0 <= metrics['f1_score'] <= 1
    assert 0 <= metrics['roc_auc'] <= 1

def test_alert_manager():
    """Тест системы алертинга"""
    alert_manager = AlertManager()
    
    # Тестовый алерт
    test_alert = {
        "level": "WARNING",
        "type": "TEST_ALERT",
        "message": "Test alert message",
        "metric": "test_metric",
        "value": 0.75,
        "threshold": 0.5
    }
    
    # Проверка форматирования сообщения
    message = alert_manager._format_alert_message(test_alert, "test")
    assert "ML MONITORING ALERT" in message
    assert "Test alert message" in message
    assert "test_metric" in message
    
    # Проверка рекомендаций
    recommendations = alert_manager._get_recommendations("DATA_DRIFT", test_alert)
    assert "проверьте источник данных" in recommendations.lower()

def test_concept_drift_detection():
    """Тест обнаружения концептуального дрейфа"""
    monitor = ModelMonitor()
    
    # Создаем данные с концептуальным дрейфом
    n_samples = 1000
    X = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.normal(0, 1, n_samples)
    })
    
    # Симулируем концептуальный дрейф: 
    # После 500 образцов связь между feature1 и целевой переменной меняется
    y_true = np.zeros(n_samples)
    y_pred = np.zeros(n_samples)
    
    # Первые 500: feature1 положительно коррелирует с целевой переменной
    y_true[:500] = (X['feature1'][:500] > 0).astype(int)
    y_pred[:500] = (X['feature1'][:500] > 0).astype(int)
    
    # После 500: feature1 отрицательно коррелирует с целевой переменной
    # но модель продолжает предсказывать по старой логике
    y_true[500:] = (X['feature1'][500:] < 0).astype(int)
    y_pred[500:] = (X['feature1'][500:] > 0).astype(int)  # Модель ошибается
    
    # Добавляем шум
    y_true = pd.Series(y_true)
    y_pred = pd.Series(y_pred)
    
    # Обнаружение концептуального дрейфа
    concept_drift = monitor._detect_concept_drift(X, y_true, y_pred)
    
    # Должен обнаружить концептуальный дрейф
    assert concept_drift == True

if __name__ == "__main__":
    pytest.main([__file__, "-v"])