import mlflow
from mlflow.tracking import MlflowClient
import json
from datetime import datetime
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def manage_model_registry():
    """Управление реестром моделей в MLflow"""
    logger.info("Управление Model Registry...")
    
    # Настройка клиента
    mlflow.set_tracking_uri("http://mlflow:5000")
    client = MlflowClient()
    
    # Загрузка результатов обучения
    results_path = Path("/app/artifacts/reports/model_training_results.csv")
    if not results_path.exists():
        logger.error("Результаты обучения не найдены!")
        return None
    
    import pandas as pd
    results_df = pd.read_csv(results_path)
    
    if results_df.empty:
        logger.error("Нет результатов обучения!")
        return None
    
    # Находим лучшую модель
    best_model_row = results_df.iloc[0]
    best_model_name = best_model_row["model"]
    best_run_id = best_model_row["run_id"]
    
    # Имя модели в реестре
    registered_model_name = "EcommerceRecommendationModel"
    
    try:
        # 1. Создание или получение зарегистрированной модели
        try:
            registered_model = client.get_registered_model(registered_model_name)
            logger.info(f"Модель {registered_model_name} уже существует в реестре")
        except mlflow.exceptions.RestException:
            client.create_registered_model(registered_model_name)
            logger.info(f"Создана новая модель в реестре: {registered_model_name}")
        
        # 2. Регистрация новой версии модели
        model_uri = f"runs:/{best_run_id}/model"
        
        logger.info(f"Регистрация модели: {model_uri}")
        
        mv = client.create_model_version(
            name=registered_model_name,
            source=model_uri,
            run_id=best_run_id
        )
        
        version = mv.version
        logger.info(f"Создана версия {version} модели {registered_model_name}")
        
        # 3. Добавление описания и тегов
        client.update_model_version(
            name=registered_model_name,
            version=version,
            description=f"Лучшая модель: {best_model_name}. AUC: {best_model_row['roc_auc']:.4f}"
        )
        
        # 4. Переход версии в Production
        client.transition_model_version_stage(
            name=registered_model_name,
            version=version,
            stage="Production",
            archive_existing_versions=True
        )
        
        logger.info(f"Версия {version} переведена в стадию Production")
        
        # 5. Добавление метаданных
        metadata = {
            "model_type": best_model_name,
            "training_date": datetime.now().isoformat(),
            "metrics": {
                "roc_auc": float(best_model_row["roc_auc"]),
                "accuracy": float(best_model_row["accuracy"]),
                "f1_score": float(best_model_row["f1_score"]),
                "inference_time_ms": float(best_model_row["inference_time_ms"])
            },
            "training_parameters": {
                "n_features": int(best_model_row.get("n_features", 0)),
                "training_samples": int(best_model_row.get("training_samples", 0))
            }
        }
        
        # Сохранение метаданных
        metadata_path = Path("/app/artifacts/reports/model_metadata.json")
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        # Логирование метаданных
        with mlflow.start_run(run_name="model_registry_metadata"):
            mlflow.log_param("registered_model_name", registered_model_name)
            mlflow.log_param("model_version", version)
            mlflow.log_param("model_type", best_model_name)
            mlflow.log_metric("production_auc", best_model_row["roc_auc"])
            mlflow.log_artifact(str(metadata_path))
        
        # 6. Загрузка production модели для проверки
        try:
            production_model = mlflow.pyfunc.load_model(
                f"models:/{registered_model_name}/Production"
            )
            logger.info(f"✅ Production модель успешно загружена: {registered_model_name}/v{version}")
            
            # Сохранение информации о загрузке модели
            model_info = {
                "model_name": registered_model_name,
                "version": version,
                "stage": "Production",
                "loaded_successfully": True,
                "load_timestamp": datetime.now().isoformat()
            }
            
            info_path = Path("/app/artifacts/reports/production_model_info.json")
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump(model_info, f, indent=2)
                
        except Exception as e:
            logger.error(f"Ошибка при загрузке production модели: {str(e)}")
        
        # 7. Получение информации о всех версиях модели
        model_versions = client.search_model_versions(f"name='{registered_model_name}'")
        
        versions_info = []
        for mv in model_versions:
            versions_info.append({
                "version": mv.version,
                "stage": mv.current_stage,
                "run_id": mv.run_id,
                "status": mv.status,
                "description": mv.description
            })
        
        # Сохранение информации о версиях
        versions_path = Path("/app/artifacts/reports/model_versions.json")
        with open(versions_path, 'w', encoding='utf-8') as f:
            json.dump(versions_info, f, indent=2)
        
        # Вывод результатов
        print("\n" + "="*80)
        print(" РЕЗУЛЬТАТЫ РАБОТЫ С MODEL REGISTRY")
        print("="*80)
        print(f"Зарегистрированная модель: {registered_model_name}")
        print(f"Лучшая модель: {best_model_name}")
        print(f"Версия в Production: {version}")
        print(f"ROC-AUC: {best_model_row['roc_auc']:.4f}")
        print(f"Inference Time: {best_model_row['inference_time_ms']:.2f} ms")
        print(f"\nВсе версии модели:")
        for vi in versions_info:
            print(f"  v{vi['version']}: {vi['stage']} - {vi.get('description', '')}")
        print("\nОтчеты сохранены в /app/artifacts/reports/")
        print("="*80)
        
        return {
            "registered_model_name": registered_model_name,
            "production_version": version,
            "best_model": best_model_name,
            "metrics": metadata["metrics"],
            "versions": versions_info
        }
        
    except Exception as e:
        logger.error(f"Ошибка при работе с Model Registry: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def compare_experiments():
    """Сравнение моделей из разных экспериментов"""
    logger.info("Сравнение экспериментов...")
    
    client = MlflowClient()
    mlflow.set_tracking_uri("http://mlflow:5000")
    
    try:
        # Поиск экспериментов
        experiments = mlflow.search_experiments()
        experiment_results = {}
        
        for exp in experiments:
            if exp.name.startswith("ecommerce_"):
                runs = mlflow.search_runs(
                    experiment_ids=[exp.experiment_id],
                    order_by=["metrics.roc_auc DESC"],
                    max_results=1
                )
                
                if not runs.empty:
                    best_run = runs.iloc[0]
                    experiment_results[exp.name] = {
                        "best_run_id": best_run["run_id"],
                        "roc_auc": best_run["metrics.roc_auc"],
                        "model": best_run["params.model_name"],
                        "inference_time_ms": best_run["metrics.inference_time_ms"]
                    }
        
        # Сравнение результатов
        if len(experiment_results) >= 2:
            experiment_names = list(experiment_results.keys())
            
            with mlflow.start_run(run_name="experiment_comparison"):
                for exp_name, results in experiment_results.items():
                    mlflow.log_metric(f"{exp_name}_auc", results["roc_auc"])
                    mlflow.log_metric(f"{exp_name}_latency", results["inference_time_ms"])
                
                # Определение лучшего эксперимента
                best_exp = max(experiment_results.items(), 
                             key=lambda x: x[1]["roc_auc"])
                
                mlflow.log_param("best_experiment", best_exp[0])
                mlflow.log_metric("best_experiment_auc", best_exp[1]["roc_auc"])
                
                logger.info(f"Лучший эксперимент: {best_exp[0]} (AUC: {best_exp[1]['roc_auc']:.4f})")
        
        return experiment_results
        
    except Exception as e:
        logger.error(f"Ошибка при сравнении экспериментов: {str(e)}")
        return None

if __name__ == "__main__":
    # 1. Управление реестром моделей
    registry_result = manage_model_registry()
    
    # 2. Сравнение экспериментов
    comparison_result = compare_experiments()
    
    # 3. Сохранение финального отчета
    final_report = {
        "timestamp": datetime.now().isoformat(),
        "model_registry": registry_result,
        "experiment_comparison": comparison_result
    }
    
    report_path = Path("/app/artifacts/reports/final_registry_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, indent=2, default=str)
    
    logger.info(f"Финальный отчет сохранен: {report_path}")