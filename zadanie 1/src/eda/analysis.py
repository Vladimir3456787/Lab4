import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import json
from pathlib import Path
import sys
import os

# Простой логгер
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def perform_eda():
    logger.info("Начинаем EDA анализ...")
    
    # Загружаем данные
    data_path = Path("/app/data/processed/october_processed.parquet")
    if not data_path.exists():
        logger.error("Данные не найдены!")
        return None
    
    df = pd.read_parquet(data_path)
    logger.info(f"Данные загружены: {df.shape}")
    
    try:
        # Подключаемся к MLflow
        mlflow.set_tracking_uri("http://mlflow:5000")
        mlflow.set_experiment("ecommerce_eda")
        
        with mlflow.start_run(run_name="basic_eda"):
            # Базовые метрики
            mlflow.log_param("total_samples", len(df))
            mlflow.log_param("total_columns", len(df.columns))
            mlflow.log_param("unique_users", df['user_id'].nunique())
            mlflow.log_param("unique_products", df['product_id'].nunique())
            
            # Распределение событий
            event_counts = df['event_type'].value_counts()
            for event, count in event_counts.items():
                mlflow.log_metric(f"count_{event}", count)
            
            # График 1: Распределение событий
            plt.figure(figsize=(10, 6))
            event_counts.plot(kind='bar', color=['skyblue', 'lightgreen', 'salmon'])
            plt.title('Распределение типов событий', fontsize=14, fontweight='bold')
            plt.xlabel('Тип события')
            plt.ylabel('Количество')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            plot_path = Path("/app/artifacts/eda/event_distribution.png")
            plot_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(plot_path, dpi=300)
            plt.close()
            mlflow.log_artifact(str(plot_path))
            
            # График 2: Активность по часам
            df['hour'] = df['event_time'].dt.hour
            hourly_activity = df.groupby('hour').size()
            
            plt.figure(figsize=(12, 6))
            hourly_activity.plot(kind='line', marker='o', linewidth=2)
            plt.title('Активность пользователей по часам', fontsize=14, fontweight='bold')
            plt.xlabel('Час дня')
            plt.ylabel('Количество событий')
            plt.grid(True, alpha=0.3)
            plt.xticks(range(0, 24))
            plt.tight_layout()
            
            plot_path2 = Path("/app/artifacts/eda/hourly_activity.png")
            plt.savefig(plot_path2, dpi=300)
            plt.close()
            mlflow.log_artifact(str(plot_path2))
            
            # Создаем отчет
            report = {
                "dataset_info": {
                    "shape": list(df.shape),
                    "columns": df.columns.tolist(),
                    "date_range": {
                        "start": df['event_time'].min().isoformat(),
                        "end": df['event_time'].max().isoformat()
                    }
                },
                "event_distribution": event_counts.to_dict(),
                "user_analysis": {
                    "unique_users": int(df['user_id'].nunique()),
                    "events_per_user_avg": float(len(df) / df['user_id'].nunique())
                },
                "product_analysis": {
                    "unique_products": int(df['product_id'].nunique()),
                    "price_stats": {
                        "mean": float(df['price'].mean()),
                        "std": float(df['price'].std()),
                        "min": float(df['price'].min()),
                        "max": float(df['price'].max())
                    }
                }
            }
            
            report_path = Path("/app/artifacts/eda/report.json")
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            mlflow.log_artifact(str(report_path))
            
            logger.info("✅ EDA завершен!")
            
            # Выводим результаты
            print("\n" + "="*60)
            print("📊 РЕЗУЛЬТАТЫ EDA:")
            print("="*60)
            print(f"Всего строк: {df.shape[0]:,}")
            print(f"Распределение событий: {event_counts.to_dict()}")
            print(f"Уникальных пользователей: {df['user_id'].nunique()}")
            print(f"Уникальных товаров: {df['product_id'].nunique()}")
            print(f"Средняя цена: ${df['price'].mean():.2f}")
            print(f"Создано графиков: 2")
            print(f"Отчет сохранен: {report_path}")
            print("="*60)
            
            return report
            
    except Exception as e:
        logger.error(f"Ошибка при выполнении EDA: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🚀 Запуск EDA анализа...")
    report = perform_eda()
    if report:
        print("\n✅ EDA АНАЛИЗ ЗАВЕРШЕН УСПЕШНО!")
    else:
        print("\n❌ ОШИБКА ПРИ ВЫПОЛНЕНИИ EDA")
