#!/usr/bin/env python3
"""
Упрощенный планировщик задач мониторинга
"""

import schedule
import time
import logging
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np
import sys
import os

# Добавление src в path
sys.path.append('/app/src')

from monitoring.data_monitor import DataMonitor
from monitoring.model_monitor import ModelMonitor
from monitoring.alerting import AlertManager

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SimpleMonitoring:
    """Упрощенный планировщик задач мониторинга"""
    
    def __init__(self):
        self.data_monitor = DataMonitor(
            Path("/app/artifacts/reference_data/reference.csv")
        )
        self.model_monitor = ModelMonitor()
        self.alert_manager = AlertManager()
        
        # Статистика выполнения
        self.execution_stats = {
            'data_monitoring_runs': 0,
            'model_monitoring_runs': 0,
            'alerts_sent': 0,
            'last_run': None
        }
    
    def run_data_monitoring(self):
        """Запуск мониторинга данных"""
        try:
            logger.info("Starting data monitoring...")
            
            # Загрузка текущих данных
            current_data = self._load_current_data()
            
            if current_data is not None:
                # Запуск мониторинга
                result = self.data_monitor.monitor_data_drift(
                    current_data, 
                    "production"
                )
                
                # Отправка алертов при необходимости
                if result.get('alerts'):
                    for alert in result['alerts']:
                        self.alert_manager.send_alert(alert, "data_drift")
                        self.execution_stats['alerts_sent'] += 1
                
                self.execution_stats['data_monitoring_runs'] += 1
                self.execution_stats['last_run'] = datetime.now().isoformat()
                
                logger.info(f"Data monitoring completed. Drift score: {result.get('metrics', {}).get('drift_score', 0):.3f}")
            
        except Exception as e:
            logger.error(f"Error in data monitoring: {e}")
    
    def run_model_monitoring(self):
        """Запуск мониторинга модели"""
        try:
            logger.info("Starting model monitoring...")
            
            # Генерация тестовых данных
            X_test, y_true, y_pred, y_pred_proba = self._generate_test_data()
            
            # Запуск мониторинга
            result = self.model_monitor.monitor_model_performance(
                X_test, y_true, y_pred, y_pred_proba, "production"
            )
            
            # Отправка алертов при необходимости
            if result.get('alerts'):
                for alert in result['alerts']:
                    self.alert_manager.send_alert(alert, "model_performance")
                    self.execution_stats['alerts_sent'] += 1
            
            self.execution_stats['model_monitoring_runs'] += 1
            self.execution_stats['last_run'] = datetime.now().isoformat()
            
            logger.info(f"Model monitoring completed. Accuracy: {result.get('performance_metrics', {}).get('accuracy', 0):.3f}")
            
        except Exception as e:
            logger.error(f"Error in model monitoring: {e}")
    
    def _load_current_data(self) -> pd.DataFrame:
        """Загрузка текущих данных"""
        try:
            np.random.seed(int(datetime.now().timestamp()) % 1000)
            n_samples = 1000
            
            # Создаем синтетические данные
            data = pd.DataFrame({
                'price': np.random.exponential(100, n_samples),
                'hour': np.random.randint(0, 24, n_samples),
                'day_of_week': np.random.randint(0, 7, n_samples),
                'transaction_amount': np.random.exponential(80, n_samples)
            })
            
            return data
            
        except Exception as e:
            logger.error(f"Error loading current data: {e}")
            return None
    
    def _generate_test_data(self):
        """Генерация тестовых данных для мониторинга модели"""
        np.random.seed(int(datetime.now().timestamp()) % 1000)
        n_samples = 500
        
        # Признаки
        X_test = pd.DataFrame({
            'feature1': np.random.normal(0, 1, n_samples),
            'feature2': np.random.exponential(1, n_samples),
        })
        
        # Истинные метки
        y_true = pd.Series(np.random.choice([0, 1], n_samples, p=[0.92, 0.08]))
        
        # Предсказания
        y_pred = y_true.copy()
        error_rate = 0.12  # 12% ошибок
        error_indices = np.random.choice(
            n_samples, 
            size=int(n_samples * error_rate), 
            replace=False
        )
        y_pred.iloc[error_indices] = 1 - y_pred.iloc[error_indices]
        
        # Вероятности
        y_pred_proba = pd.Series(np.where(
            y_pred == 1, 
            np.random.beta(8, 2, n_samples),
            np.random.beta(2, 8, n_samples)
        ))
        
        return X_test, y_true, y_pred, y_pred_proba
    
    def run(self):
        """Основной цикл выполнения"""
        logger.info("Starting simple monitoring system...")
        logger.info("Press Ctrl+C to stop")
        
        try:
            # Запускаем сразу оба мониторинга
            self.run_data_monitoring()
            self.run_model_monitoring()
            
            # Затем планируем выполнение каждые 5 минут
            schedule.every(5).minutes.do(self.run_data_monitoring)
            schedule.every(10).minutes.do(self.run_model_monitoring)
            
            # Бесконечный цикл
            while True:
                schedule.run_pending()
                time.sleep(60)  # Проверка каждую минуту
                
        except KeyboardInterrupt:
            logger.info("Monitoring system stopped by user")
        except Exception as e:
            logger.error(f"Monitoring system error: {e}")

if __name__ == "__main__":
    monitor = SimpleMonitoring()
    monitor.run()
