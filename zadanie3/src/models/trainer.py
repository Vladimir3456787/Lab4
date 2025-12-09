import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
import mlflow.xgboost
import mlflow.catboost
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, confusion_matrix,
    classification_report
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import json
from datetime import datetime
from pathlib import Path
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def train_and_log_models():
    """Обучение и логирование нескольких моделей в MLflow"""
    logger.info("Начинаем обучение моделей...")
    
    # Загрузка данных
    data_path = Path("/app/artifacts/models/processed_data.joblib")
    if not data_path.exists():
        logger.error("Данные не найдены!")
        return None
    
    data = joblib.load(data_path)
    X_train, X_test, y_train, y_test, feature_names = \
        data['X_train'], data['X_test'], data['y_train'], data['y_test'], data['feature_names']
    
    # Масштабирование
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Сохранение scaler
    joblib.dump(scaler, "/app/artifacts/models/scaler.joblib")
    
    # Обработка дисбаланса классов
    try:
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        logger.info(f"После SMOTE: {X_train_balanced.shape}")
    except Exception as e:
        logger.warning(f"SMOTE не применен: {e}")
        X_train_balanced, y_train_balanced = X_train_scaled, y_train
    
    # Настройка MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("ecommerce_recommendations_v3")
    
    # Список моделей для обучения
    models_config = {
        "LightGBM": {
            "module": "lightgbm",
            "class": "LGBMClassifier",
            "params": {
                "n_estimators": 200,
                "max_depth": 7,
                "learning_rate": 0.1,
                "random_state": 42,
                "class_weight": "balanced",
                "verbose": -1,
                "n_jobs": -1
            }
        },
        "XGBoost": {
            "module": "xgboost",
            "class": "XGBClassifier",
            "params": {
                "n_estimators": 200,
                "max_depth": 6,
                "learning_rate": 0.1,
                "random_state": 42,
                "eval_metric": "logloss",
                "use_label_encoder": False,
                "scale_pos_weight": len(y_train_balanced[y_train_balanced==0]) / len(y_train_balanced[y_train_balanced==1])
            }
        },
        "RandomForest": {
            "module": "sklearn.ensemble",
            "class": "RandomForestClassifier",
            "params": {
                "n_estimators": 150,
                "max_depth": 15,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
                "random_state": 42,
                "class_weight": "balanced",
                "n_jobs": -1
            }
        }
    }
    
    results = []
    
    for model_name, config in models_config.items():
        logger.info(f"Обучение модели: {model_name}")
        
        try:
            with mlflow.start_run(run_name=f"{model_name}_training"):
                # Логирование параметров
                mlflow.log_param("model_name", model_name)
                mlflow.log_param("n_features", len(feature_names))
                mlflow.log_param("training_samples", len(X_train_balanced))
                
                for param_name, param_value in config["params"].items():
                    mlflow.log_param(param_name, param_value)
                
                # Импорт и создание модели
                if config["module"] == "lightgbm":
                    import lightgbm as lgb
                    model = lgb.LGBMClassifier(**config["params"])
                elif config["module"] == "xgboost":
                    import xgboost as xgb
                    model = xgb.XGBClassifier(**config["params"])
                elif config["module"] == "sklearn.ensemble":
                    from sklearn.ensemble import RandomForestClassifier
                    model = RandomForestClassifier(**config["params"])
                
                # Обучение с таймингом
                train_start = time.time()
                model.fit(X_train_balanced, y_train_balanced)
                train_time = time.time() - train_start
                
                # Предсказание и оценка
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
                
                metrics = {
                    "accuracy": accuracy_score(y_test, y_pred),
                    "roc_auc": roc_auc_score(y_test, y_pred_proba),
                    "f1_score": f1_score(y_test, y_pred, average='weighted'),
                    "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    "training_time_seconds": train_time
                }
                
                # Измерение latency
                inference_start = time.time()
                for _ in range(100):
                    model.predict(X_test_scaled[:10])
                inference_time = (time.time() - inference_start) / 100 * 1000  # мс на запрос
                metrics["inference_time_ms"] = inference_time
                
                # Логирование метрик
                for metric_name, metric_value in metrics.items():
                    mlflow.log_metric(metric_name, metric_value)
                
                # Логирование confusion matrix
                cm = confusion_matrix(y_test, y_pred)
                cm_dict = {
                    "true_negatives": int(cm[0, 0]),
                    "false_positives": int(cm[0, 1]),
                    "false_negatives": int(cm[1, 0]),
                    "true_positives": int(cm[1, 1])
                }
                
                for cm_name, cm_value in cm_dict.items():
                    mlflow.log_metric(cm_name, cm_value)
                
                # Логирование модели
                if model_name == "LightGBM":
                    mlflow.lightgbm.log_model(model, "model")
                elif model_name == "XGBoost":
                    mlflow.xgboost.log_model(model, "model")
                elif model_name == "RandomForest":
                    mlflow.sklearn.log_model(model, "model")
                
                # Сохранение модели локально
                model_path = f"/app/artifacts/models/{model_name}_v1.joblib"
                joblib.dump(model, model_path)
                mlflow.log_artifact(model_path)
                
                # Сбор результатов
                results.append({
                    "model": model_name,
                    "run_id": mlflow.active_run().info.run_id,
                    **metrics,
                    **cm_dict
                })
                
                logger.info(f"✅ {model_name}: AUC={metrics['roc_auc']:.4f}, Time={metrics['inference_time_ms']:.2f}ms")
                
        except Exception as e:
            logger.error(f"Ошибка при обучении {model_name}: {str(e)}")
            continue
    
    # Сохранение результатов
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("roc_auc", ascending=False)
    results_df.to_csv("/app/artifacts/reports/model_training_results.csv", index=False)
    results_df.to_json("/app/artifacts/reports/model_training_results.json", orient="records", indent=2)
    
    # Логирование финального отчета
    with mlflow.start_run(run_name="training_summary"):
        mlflow.log_artifact("/app/artifacts/reports/model_training_results.csv")
        mlflow.log_metric("best_model_auc", results_df.iloc[0]["roc_auc"])
        mlflow.log_param("best_model_name", results_df.iloc[0]["model"])
        mlflow.log_param("total_models_trained", len(results_df))
    
    # Вывод результатов
    print("\n" + "="*80)
    print(" РЕЗУЛЬТАТЫ ОБУЧЕНИЯ МОДЕЛЕЙ")
    print("="*80)
    print(f"{'Модель':15s} {'ROC-AUC':>10s} {'Accuracy':>10s} {'F1-Score':>10s} {'Time(ms)':>10s}")
    print("-"*80)
    
    for _, row in results_df.iterrows():
        print(f"{row['model']:15s} {row['roc_auc']:10.4f} {row['accuracy']:10.4f} {row['f1_score']:10.4f} {row['inference_time_ms']:10.2f}")
    
    print("-"*80)
    best_model = results_df.iloc[0]
    print(f"\n ЛУЧШАЯ МОДЕЛЬ: {best_model['model']}")
    print(f" ROC-AUC: {best_model['roc_auc']:.4f}")
    print(f" Accuracy: {best_model['accuracy']:.4f}")
    print(f" Inference Time: {best_model['inference_time_ms']:.2f} ms")
    print(f" Training Time: {best_model['training_time_seconds']:.1f} sec")
    print(f" MLflow Run ID: {best_model['run_id']}")
    print("="*80)
    
    return results_df

if __name__ == "__main__":
    train_and_log_models()