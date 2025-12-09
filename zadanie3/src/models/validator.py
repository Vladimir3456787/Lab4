import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import time
import joblib
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ModelValidator:
    """Класс для валидации моделей перед деплоем"""
    
    def __init__(self, mlflow_tracking_uri="http://mlflow:5000"):
        self.client = MlflowClient(tracking_uri=mlflow_tracking_uri)
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
    def validate_model(self, model, X_test, y_test):
        """
        Валидация модели перед деплоем
        Возвращает словарь с результатами валидации
        """
        from sklearn.metrics import roc_auc_score
        
        validation_results = {
            "passed": True,
            "errors": [],
            "metrics": {}
        }
        
        try:
            # 1. Проверка качества модели (AUC > 0.85)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_pred_proba)
            validation_results["metrics"]["auc"] = auc
            
            if auc < 0.85:
                validation_results["passed"] = False
                validation_results["errors"].append(
                    f"Модель не проходит валидацию по качеству (AUC={auc:.4f} < 0.85)"
                )
            
            # 2. Проверка latency (< 50 мс)
            latencies = []
            for _ in range(100):
                start_time = time.time()
                model.predict(X_test[:10])
                end_time = time.time()
                latencies.append((end_time - start_time) * 1000)  # в миллисекундах
            
            avg_latency = np.mean(latencies)
            validation_results["metrics"]["avg_latency_ms"] = avg_latency
            validation_results["metrics"]["p95_latency_ms"] = np.percentile(latencies, 95)
            
            if avg_latency > 50:
                validation_results["passed"] = False
                validation_results["errors"].append(
                    f"Модель слишком медленная (avg latency={avg_latency:.2f} ms > 50 ms)"
                )
            
            # 3. Проверка стабильности предсказаний
            predictions = []
            for _ in range(10):
                pred = model.predict(X_test[:100])
                predictions.append(pred)
            
            # Проверяем, что предсказания стабильны
            first_pred = predictions[0]
            stable = all(np.array_equal(first_pred, pred) for pred in predictions[1:])
            
            if not stable:
                validation_results["passed"] = False
                validation_results["errors"].append("Модель дает нестабильные предсказания")
            
            # 4. Проверка на наличие NaN/Inf в предсказаниях
            if np.any(np.isnan(y_pred_proba)) or np.any(np.isinf(y_pred_proba)):
                validation_results["passed"] = False
                validation_results["errors"].append("Модель возвращает NaN или Inf значения")
            
            logger.info(f"Валидация модели: AUC={auc:.4f}, Latency={avg_latency:.2f}ms")
            
        except Exception as e:
            validation_results["passed"] = False
            validation_results["errors"].append(f"Ошибка при валидации: {str(e)}")
            logger.error(f"Ошибка валидации: {str(e)}")
        
        return validation_results
    
    def compare_models(self, model1, model2, X_test, y_test, model1_name="Model1", model2_name="Model2"):
        """Сравнение двух моделей"""
        from sklearn.metrics import roc_auc_score
        
        comparison_results = {
            "model1": model1_name,
            "model2": model2_name,
            "metrics": {}
        }
        
        try:
            # Предсказания
            y_pred_proba1 = model1.predict_proba(X_test)[:, 1]
            y_pred_proba2 = model2.predict_proba(X_test)[:, 1]
            
            # Метрики
            auc1 = roc_auc_score(y_test, y_pred_proba1)
            auc2 = roc_auc_score(y_test, y_pred_proba2)
            
            # Latency
            latencies1 = self._measure_latency(model1, X_test[:10])
            latencies2 = self._measure_latency(model2, X_test[:10])
            
            comparison_results["metrics"] = {
                f"{model1_name}_auc": auc1,
                f"{model2_name}_auc": auc2,
                f"{model1_name}_avg_latency_ms": np.mean(latencies1),
                f"{model2_name}_avg_latency_ms": np.mean(latencies2),
                "auc_difference": auc1 - auc2,
                "latency_difference": np.mean(latencies1) - np.mean(latencies2)
            }
            
            # Логирование сравнения в MLflow
            with mlflow.start_run(run_name=f"model_comparison_{model1_name}_vs_{model2_name}"):
                for metric_name, metric_value in comparison_results["metrics"].items():
                    mlflow.log_metric(metric_name, metric_value)
                
                mlflow.log_param("model1", model1_name)
                mlflow.log_param("model2", model2_name)
                
                # Определение победителя
                if auc1 > auc2 and np.mean(latencies1) < np.mean(latencies2):
                    winner = model1_name
                elif auc2 > auc1 and np.mean(latencies2) < np.mean(latencies1):
                    winner = model2_name
                elif auc1 > auc2:
                    winner = f"{model1_name} (лучше AUC)"
                else:
                    winner = f"{model2_name} (лучше AUC)"
                
                mlflow.log_param("winner", winner)
                comparison_results["winner"] = winner
            
            logger.info(f"Сравнение моделей: {model1_name} AUC={auc1:.4f}, {model2_name} AUC={auc2:.4f}")
            logger.info(f"Победитель: {winner}")
            
        except Exception as e:
            logger.error(f"Ошибка при сравнении моделей: {str(e)}")
            comparison_results["error"] = str(e)
        
        return comparison_results
    
    def _measure_latency(self, model, X_sample, n_iterations=100):
        """Измерение latency модели"""
        latencies = []
        for _ in range(n_iterations):
            start_time = time.time()
            model.predict(X_sample)
            end_time = time.time()
            latencies.append((end_time - start_time) * 1000)
        return latencies

def run_validation():
    """Запуск валидации моделей"""
    logger.info("Запуск валидации моделей...")
    
    # Загрузка данных
    data_path = Path("/app/artifacts/models/processed_data.joblib")
    if not data_path.exists():
        logger.error("Данные не найдены!")
        return None
    
    data = joblib.load(data_path)
    X_test, y_test = data['X_test'], data['y_test']
    
    # Загрузка scaler
    scaler = joblib.load("/app/artifacts/models/scaler.joblib")
    X_test_scaled = scaler.transform(X_test)
    
    # Создание валидатора
    validator = ModelValidator()
    
    # Загрузка обученных моделей
    models = {}
    model_files = {
        "LightGBM": "/app/artifacts/models/LightGBM_v1.joblib",
        "XGBoost": "/app/artifacts/models/XGBoost_v1.joblib",
        "RandomForest": "/app/artifacts/models/RandomForest_v1.joblib"
    }
    
    validation_results = {}
    
    for model_name, model_path in model_files.items():
        if Path(model_path).exists():
            try:
                model = joblib.load(model_path)
                validation_result = validator.validate_model(model, X_test_scaled[:1000], y_test[:1000])
                validation_results[model_name] = validation_result
                
                logger.info(f"Валидация {model_name}: {'ПРОЙДЕНА' if validation_result['passed'] else 'НЕ ПРОЙДЕНА'}")
                if not validation_result["passed"]:
                    for error in validation_result["errors"]:
                        logger.warning(f"  - {error}")
            except Exception as e:
                logger.error(f"Ошибка при загрузке модели {model_name}: {str(e)}")
    
    # Сравнение моделей
    if len(models) >= 2:
        model_names = list(models.keys())
        comparison = validator.compare_models(
            models[model_names[0]], models[model_names[1]],
            X_test_scaled[:1000], y_test[:1000],
            model_names[0], model_names[1]
        )
        validation_results["comparison"] = comparison
    
    # Сохранение результатов валидации
    validation_path = Path("/app/artifacts/reports/validation_report.json")
    import json
    with open(validation_path, 'w', encoding='utf-8') as f:
        json.dump(validation_results, f, indent=2, default=str)
    
    # Сводка
    passed_models = [name for name, result in validation_results.items() 
                    if isinstance(result, dict) and result.get("passed", False)]
    
    print("\n" + "="*80)
    print(" РЕЗУЛЬТАТЫ ВАЛИДАЦИИ МОДЕЛЕЙ")
    print("="*80)
    print(f"Всего проверено моделей: {len(validation_results) - ('comparison' in validation_results)}")
    print(f"Успешно прошли валидацию: {len(passed_models)}")
    
    for model_name, result in validation_results.items():
        if model_name != "comparison" and isinstance(result, dict):
            status = "✅ ПРОЙДЕНА" if result.get("passed", False) else "❌ НЕ ПРОЙДЕНА"
            auc = result.get("metrics", {}).get("auc", "N/A")
            latency = result.get("metrics", {}).get("avg_latency_ms", "N/A")
            print(f"\n{model_name}: {status}")
            print(f"  AUC: {auc:.4f if isinstance(auc, (int, float)) else auc}")
            print(f"  Latency: {latency:.2f if isinstance(latency, (int, float)) else latency} ms")
    
    if "comparison" in validation_results:
        comp = validation_results["comparison"]
        print(f"\nСРАВНЕНИЕ МОДЕЛЕЙ:")
        print(f"  Победитель: {comp.get('winner', 'N/A')}")
        print(f"  Разница в AUC: {comp.get('metrics', {}).get('auc_difference', 0):.4f}")
    
    print("\nОтчет валидации сохранен в:", validation_path)
    print("="*80)
    
    return validation_results

if __name__ == "__main__":
    run_validation()