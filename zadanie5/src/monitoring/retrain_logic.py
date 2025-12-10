import mlflow
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import json
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RetrainManager:
    """Менеджер автоматического переобучения модели"""
    
    def __init__(self, mlflow_tracking_uri: str = "http://mlflow:5000"):
        self.mlflow_tracking_uri = mlflow_tracking_uri
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # Конфигурация retrain
        self.retrain_config = {
            'data_drift_threshold': 0.3,      # PSI > 0.3
            'accuracy_drop_threshold': 0.15,  # Падение accuracy > 15%
            'weekly_retrain_day': 0,          # Понедельник (0 = Monday)
            'min_retrain_interval_days': 7,   # Минимум 7 дней между retrain
            'max_retrain_age_days': 30,       # Максимальный возраст модели
            'retrain_on_new_data': True,      # Retrain при появлении новых данных
            'min_new_data_samples': 1000      # Минимальное количество новых данных
        }
        
        # История retrain
        self.retrain_history_path = Path("/app/artifacts/reports/retrain_history.json")
        self.retrain_history = self._load_retrain_history()
    
    def _load_retrain_history(self) -> list:
        """Загрузка истории переобучений"""
        try:
            if self.retrain_history_path.exists():
                with open(self.retrain_history_path, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading retrain history: {e}")
        
        return []
    
    def _save_retrain_history(self):
        """Сохранение истории переобучений"""
        try:
            with open(self.retrain_history_path, 'w') as f:
                json.dump(self.retrain_history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving retrain history: {e}")
    
    def check_retrain_needed(self, 
                           data_drift_score: Optional[float] = None,
                           accuracy_drift: Optional[float] = None,
                           new_data_available: bool = False,
                           new_data_size: int = 0) -> Tuple[bool, Dict[str, Any]]:
        """
        Проверка необходимости переобучения модели
        
        Returns:
            Tuple[bool, Dict]: (Нужен ли retrain, Причины)
        """
        reasons = {}
        
        # 1. Проверка data drift
        if data_drift_score is not None:
            if data_drift_score > self.retrain_config['data_drift_threshold']:
                reasons['data_drift'] = {
                    'score': data_drift_score,
                    'threshold': self.retrain_config['data_drift_threshold'],
                    'message': f'Data drift превысил порог: {data_drift_score:.3f} > {self.retrain_config["data_drift_threshold"]}'
                }
        
        # 2. Проверка падения accuracy
        if accuracy_drift is not None:
            if accuracy_drift > self.retrain_config['accuracy_drop_threshold']:
                reasons['accuracy_drop'] = {
                    'drop': accuracy_drift,
                    'threshold': self.retrain_config['accuracy_drop_threshold'],
                    'message': f'Падение accuracy превысило порог: {accuracy_drift:.3f} > {self.retrain_config["accuracy_drop_threshold"]}'
                }
        
        # 3. Проверка еженедельного retrain
        if self._is_weekly_retrain_day():
            reasons['weekly_schedule'] = {
                'day': datetime.now().strftime('%A'),
                'message': 'Запланированный еженедельный retrain'
            }
        
        # 4. Проверка возраста модели
        model_age_days = self._get_production_model_age()
        if model_age_days > self.retrain_config['max_retrain_age_days']:
            reasons['model_age'] = {
                'age_days': model_age_days,
                'max_age': self.retrain_config['max_retrain_age_days'],
                'message': f'Модель слишком старая: {model_age_days} дней > {self.retrain_config["max_retrain_age_days"]}'
            }
        
        # 5. Проверка новых данных
        if (self.retrain_config['retrain_on_new_data'] and 
            new_data_available and 
            new_data_size >= self.retrain_config['min_new_data_samples']):
            reasons['new_data'] = {
                'samples': new_data_size,
                'min_samples': self.retrain_config['min_new_data_samples'],
                'message': f'Достаточно новых данных для retrain: {new_data_size} samples'
            }
        
        # 6. Проверка минимального интервала
        last_retrain = self._get_last_retrain_date()
        if last_retrain:
            days_since_last_retrain = (datetime.now() - last_retrain).days
            if days_since_last_retrain < self.retrain_config['min_retrain_interval_days']:
                # Слишком рано для retrain
                if reasons:
                    logger.info(f"Retrain needed but too soon. Last retrain: {days_since_last_retrain} days ago")
                    return False, {
                        'retrain_needed': True,
                        'delayed': True,
                        'days_since_last_retrain': days_since_last_retrain,
                        'min_interval': self.retrain_config['min_retrain_interval_days'],
                        'reasons': reasons
                    }
                return False, {}
        
        # Если есть причины для retrain
        if reasons:
            return True, {
                'retrain_needed': True,
                'reasons': reasons,
                'timestamp': datetime.now().isoformat()
            }
        
        return False, {}
    
    def _is_weekly_retrain_day(self) -> bool:
        """Проверка, сегодня ли день еженедельного retrain"""
        today = datetime.now().weekday()  # 0 = Monday, 6 = Sunday
        return today == self.retrain_config['weekly_retrain_day']
    
    def _get_production_model_age(self) -> int:
        """Получение возраста production модели в днях"""
        try:
            from mlflow.tracking import MlflowClient
            client = MlflowClient()
            
            # Поиск production модели
            model_versions = client.search_model_versions(
                "name='EcommerceRecommendationModel' and current_stage='Production'"
            )
            
            if model_versions:
                latest_version = model_versions[0]
                creation_time = latest_version.creation_timestamp
                
                if isinstance(creation_time, int):
                    creation_time = datetime.fromtimestamp(creation_time / 1000)
                
                age_days = (datetime.now() - creation_time).days
                return age_days
            
        except Exception as e:
            logger.error(f"Error getting model age: {e}")
        
        return 365  # Большое значение по умолчанию
    
    def _get_last_retrain_date(self) -> Optional[datetime]:
        """Получение даты последнего переобучения"""
        if self.retrain_history:
            last_entry = self.retrain_history[-1]
            last_date_str = last_entry.get('timestamp', '')
            
            try:
                return datetime.fromisoformat(last_date_str.replace('Z', '+00:00'))
            except:
                pass
        
        return None
    
    def execute_retrain(self, retrain_reasons: Dict[str, Any]) -> Dict[str, Any]:
        """
        Выполнение переобучения модели
        
        Args:
            retrain_reasons: Причины retrain
        
        Returns:
            Результаты retrain
        """
        logger.info("Starting model retraining...")
        
        retrain_result = {
            'status': 'started',
            'timestamp': datetime.now().isoformat(),
            'reasons': retrain_reasons.get('reasons', {}),
            'steps': []
        }
        
        try:
            # 1. Подготовка данных
            retrain_result['steps'].append({
                'step': 'data_preparation',
                'status': 'started',
                'timestamp': datetime.now().isoformat()
            })
            
            # Здесь должен быть код подготовки данных
            # data = self._prepare_retrain_data()
            
            retrain_result['steps'][-1].update({
                'status': 'completed',
                'completion_time': datetime.now().isoformat(),
                'details': {
                    'samples': 0,  # Заменить на реальное значение
                    'features': 0   # Заменить на реальное значение
                }
            })
            
            # 2. Обучение модели
            retrain_result['steps'].append({
                'step': 'model_training',
                'status': 'started',
                'timestamp': datetime.now().isoformat()
            })
            
            # Здесь должен быть код обучения модели
            # model = self._train_model(data)
            
            retrain_result['steps'][-1].update({
                'status': 'completed',
                'completion_time': datetime.now().isoformat(),
                'details': {
                    'model_type': 'XGBoost',  # Заменить на реальное значение
                    'training_time': '0:00:00'  # Заменить на реальное значение
                }
            })
            
            # 3. Оценка модели
            retrain_result['steps'].append({
                'step': 'model_evaluation',
                'status': 'started',
                'timestamp': datetime.now().isoformat()
            })
            
            # Здесь должен быть код оценки модели
            # metrics = self._evaluate_model(model, data)
            
            retrain_result['steps'][-1].update({
                'status': 'completed',
                'completion_time': datetime.now().isoformat(),
                'details': {
                    'accuracy': 0.95,  # Заменить на реальное значение
                    'roc_auc': 0.98    # Заменить на реальное значение
                }
            })
            
            # 4. Регистрация модели в MLflow
            retrain_result['steps'].append({
                'step': 'model_registration',
                'status': 'started',
                'timestamp': datetime.now().isoformat()
            })
            
            # Здесь должен быть код регистрации модели
            # self._register_model(model, metrics)
            
            retrain_result['steps'][-1].update({
                'status': 'completed',
                'completion_time': datetime.now().isoformat(),
                'details': {
                    'model_version': 2,  # Заменить на реальное значение
                    'stage': 'Staging'   # Заменить на реальное значение
                }
            })
            
            # 5. Валидация перед деплоем
            retrain_result['steps'].append({
                'step': 'validation',
                'status': 'started',
                'timestamp': datetime.now().isoformat()
            })
            
            # Здесь должна быть валидация модели
            # validation_passed = self._validate_model(model)
            
            retrain_result['steps'][-1].update({
                'status': 'completed',
                'completion_time': datetime.now().isoformat(),
                'details': {
                    'validation_passed': True,  # Заменить на реальное значение
                    'validation_metrics': {}    # Заменить на реальное значение
                }
            })
            
            # 6. Деплой (если валидация прошла)
            retrain_result['steps'].append({
                'step': 'deployment',
                'status': 'started',
                'timestamp': datetime.now().isoformat()
            })
            
            # Здесь должен быть код деплоя
            # if validation_passed:
            #     self._deploy_model(model)
            
            retrain_result['steps'][-1].update({
                'status': 'completed',
                'completion_time': datetime.now().isoformat(),
                'details': {
                    'deployed': True,          # Заменить на реальное значение
                    'deployment_time': datetime.now().isoformat()
                }
            })
            
            # Обновление статуса
            retrain_result['status'] = 'completed'
            retrain_result['completion_time'] = datetime.now().isoformat()
            
            # Логирование в MLflow
            with mlflow.start_run(run_name=f"retrain_execution_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
                mlflow.log_dict(retrain_result, "retrain_result.json")
                mlflow.log_param("retrain_reasons", str(retrain_reasons))
                
                # Логирование метрик retrain
                for step in retrain_result['steps']:
                    if 'details' in step:
                        for key, value in step['details'].items():
                            if isinstance(value, (int, float)):
                                mlflow.log_metric(f"retrain_{step['step']}_{key}", value)
            
            # Сохранение в историю
            self.retrain_history.append({
                'timestamp': datetime.now().isoformat(),
                'result': retrain_result,
                'reasons': retrain_reasons.get('reasons', {})
            })
            self._save_retrain_history()
            
            logger.info("Model retraining completed successfully")
            
        except Exception as e:
            logger.error(f"Error during retraining: {e}")
            retrain_result.update({
                'status': 'failed',
                'error': str(e),
                'completion_time': datetime.now().isoformat()
            })
        
        return retrain_result
    
    def run_retrain_pipeline(self, 
                           data_drift_score: Optional[float] = None,
                           accuracy_drift: Optional[float] = None) -> Dict[str, Any]:
        """
        Запуск полного пайплайна проверки и выполнения retrain
        
        Returns:
            Результаты проверки и retrain
        """
        # Проверка необходимости retrain
        retrain_needed, retrain_info = self.check_retrain_needed(
            data_drift_score=data_drift_score,
            accuracy_drift=accuracy_drift,
            new_data_available=True,  # Предположим, что новые данные есть
            new_data_size=150000      # Примерное количество новых данных
        )
        
        result = {
            'check_timestamp': datetime.now().isoformat(),
            'retrain_needed': retrain_needed,
            'retrain_info': retrain_info,
            'retrain_executed': False,
            'retrain_result': None
        }
        
        # Если retrain нужен и не отложен
        if retrain_needed and not retrain_info.get('delayed', False):
            # Выполнение retrain
            retrain_result = self.execute_retrain(retrain_info)
            result.update({
                'retrain_executed': True,
                'retrain_result': retrain_result
            })
        
        # Логирование результата
        with mlflow.start_run(run_name=f"retrain_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            mlflow.log_dict(result, "retrain_check_result.json")
            mlflow.log_param("retrain_needed", retrain_needed)
            
            if retrain_needed:
                mlflow.log_param("retrain_reasons", str(retrain_info.get('reasons', {})))
        
        return result

def main():
    """Пример использования RetrainManager"""
    manager = RetrainManager()
    
    print("\n" + "="*80)
    print(" ПРОВЕРКА НЕОБХОДИМОСТИ RETRAIN")
    print("="*80)
    
    # Тестовые данные (в реальной системе - из мониторинга)
    data_drift_score = 0.35  # Выше порога 0.3
    accuracy_drift = 0.12    # Ниже порога 0.15
    
    # Проверка необходимости retrain
    retrain_needed, retrain_info = manager.check_retrain_needed(
        data_drift_score=data_drift_score,
        accuracy_drift=accuracy_drift,
        new_data_available=True,
        new_data_size=2000
    )
    
    print(f"Retrain needed: {retrain_needed}")
    
    if retrain_needed:
        print(f"\nПричины:")
        for reason, details in retrain_info.get('reasons', {}).items():
            print(f"  - {details['message']}")
        
        if retrain_info.get('delayed', False):
            print(f"\n⚠️  Retrain отложен. Прошло {retrain_info['days_since_last_retrain']} дней с последнего retrain")
            print(f"   Минимальный интервал: {retrain_info['min_interval']} дней")
        
        # Запуск retrain пайплайна
        print(f"\nЗапуск retrain пайплайна...")
        result = manager.run_retrain_pipeline(data_drift_score, accuracy_drift)
        
        if result['retrain_executed']:
            print(f"✅ Retrain выполнен успешно")
        else:
            print(f"⏸️  Retrain не выполнен (отложен или не требуется)")
    else:
        print(f"✅ Retrain не требуется")
    
    print(f"\nИстория retrain: {len(manager.retrain_history)} записей")
    print("="*80)

if __name__ == "__main__":
    main()