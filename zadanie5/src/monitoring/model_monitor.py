import mlflow
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelMonitor:
    """Упрощенный класс для мониторинга производительности модели"""
    
    def __init__(self, model_name: str = "EcommerceRecommendationModel", 
                 model_stage: str = "Production",
                 mlflow_tracking_uri: str = "http://mlflow:5000"):
        self.model_name = model_name
        self.model_stage = model_stage
        self.mlflow_tracking_uri = mlflow_tracking_uri
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # Пороговые значения для алертов
        self.performance_thresholds = {
            'accuracy_drop': 0.1,      # Падение accuracy на 10%
            'precision_drop': 0.15,    # Падение precision на 15%
            'recall_drop': 0.15,       # Падение recall на 15%
            'f1_drop': 0.1,            # Падение F1 на 10%
            'auc_drop': 0.05,          # Падение ROC-AUC на 5%
        }
    
    def monitor_model_performance(self, X_test: pd.DataFrame, y_true: pd.Series, 
                                y_pred: pd.Series, y_pred_proba: Optional[pd.Series] = None,
                                dataset_name: str = "production") -> Dict[str, Any]:
        """
        Упрощенный мониторинг производительности модели
        """
        logger.info(f"Упрощенный мониторинг модели: {len(y_true)} образцов")
        
        # Расчет метрик вручную
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        if y_pred_proba is not None:
            try:
                auc = roc_auc_score(y_true, y_pred_proba)
            except:
                auc = 0.5
        else:
            auc = 0.5
        
        # Логирование в MLflow
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        with mlflow.start_run(run_name=f"model_monitoring_{dataset_name}_{timestamp}"):
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("roc_auc", auc)
        
        # Формирование результата
        result = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "dataset_name": dataset_name,
            "model_name": self.model_name,
            "model_stage": self.model_stage,
            "performance_metrics": {
                "accuracy": accuracy,
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "roc_auc": auc,
                "samples_count": len(y_true)
            },
            "alerts": self._generate_model_alerts(accuracy, precision, recall, f1, auc),
            "report_path": f"/app/artifacts/reports/model_performance_{dataset_name}_{timestamp}.json"
        }
        
        logger.info(f"Мониторинг модели завершен. Accuracy: {accuracy:.3f}")
        
        return result
    
    def _generate_model_alerts(self, accuracy: float, precision: float, recall: float, 
                             f1: float, auc: float) -> List[Dict[str, Any]]:
        """Генерация алертов на основе результатов мониторинга модели"""
        alerts = []
        
        # Проверка абсолютных значений метрик
        if accuracy < 0.8:
            alerts.append({
                "level": "WARNING",
                "type": "LOW_ACCURACY",
                "message": f"Низкая accuracy: {accuracy:.3f}",
                "metric": "accuracy",
                "value": accuracy,
                "threshold": 0.8
            })
        
        if f1 < 0.7:
            alerts.append({
                "level": "WARNING",
                "type": "LOW_F1",
                "message": f"Низкий F1-score: {f1:.3f}",
                "metric": "f1_score",
                "value": f1,
                "threshold": 0.7
            })
        
        return alerts

def main():
    """Пример использования ModelMonitor"""
    monitor = ModelMonitor()
    
    # Генерация тестовых данных
    np.random.seed(42)
    n_samples = 1000
    
    X_test = pd.DataFrame({
        'feature1': np.random.normal(0, 1, n_samples),
        'feature2': np.random.exponential(1, n_samples),
        'feature3': np.random.randint(0, 10, n_samples)
    })
    
    # Истинные метки
    y_true = pd.Series(np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05]))
    
    # Предсказания
    y_pred = y_true.copy()
    error_indices = np.random.choice(n_samples, size=int(n_samples * 0.15), replace=False)
    y_pred.iloc[error_indices] = 1 - y_pred.iloc[error_indices]
    
    y_pred_proba = pd.Series(np.where(y_pred == 1, 0.9, 0.1))
    
    # Мониторинг производительности
    result = monitor.monitor_model_performance(
        X_test, y_true, y_pred, y_pred_proba, "test_batch"
    )
    
    print("\n" + "="*80)
    print(" РЕЗУЛЬТАТЫ МОНИТОРИНГА МОДЕЛИ")
    print("="*80)
    print(f"Модель: {result['model_name']} ({result['model_stage']})")
    print(f"Метрики производительности:")
    for key, value in result['performance_metrics'].items():
        print(f"  {key}: {value}")
    
    if result['alerts']:
        print(f"\nАЛЕРТЫ:")
        for alert in result['alerts']:
            print(f"  [{alert['level']}] {alert['message']}")
    
    print(f"\nОтчет сохранен: {result.get('report_path', 'N/A')}")
    print("="*80)

if __name__ == "__main__":
    main()
