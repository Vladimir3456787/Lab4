import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Dict, Any
import pickle
import json

logger = logging.getLogger(__name__)

class FeatureStore:
    """Простой Feature Store для демонстрации"""
    
    def __init__(self, data_path: Path):
        self.data_path = data_path
        self.features_df = None
        self._load_features()
    
    def _load_features(self):
        """Загрузка фич из CSV/Parquet файла"""
        try:
            if self.data_path.exists():
                self.features_df = pd.read_csv(self.data_path, index_col='user_id')
                logger.info(f"Feature Store loaded: {len(self.features_df)} users")
            else:
                # Создание фиктивных данных для демонстрации
                logger.warning("Feature Store file not found, creating mock data")
                self._create_mock_features()
        except Exception as e:
            logger.error(f"Failed to load Feature Store: {e}")
            self._create_mock_features()
    
    def _create_mock_features(self):
        """Создание фиктивных данных"""
        np.random.seed(42)
        n_users = 1000
        
        # Создание реалистичных фич
        data = {
            'user_id': range(1, n_users + 1),
            'avg_purchase_amount': np.random.exponential(100, n_users),
            'purchase_frequency': np.random.beta(2, 5, n_users) * 10,
            'days_since_last_purchase': np.random.randint(0, 30, n_users),
            'total_purchases': np.random.poisson(5, n_users),
            'category_preference_fashion': np.random.beta(2, 2, n_users),
            'category_preference_electronics': np.random.beta(1, 3, n_users),
            'category_preference_home': np.random.beta(3, 2, n_users),
            'session_duration_avg': np.random.normal(300, 60, n_users),
            'cart_abandonment_rate': np.random.beta(1, 4, n_users),
            'discount_sensitivity': np.random.beta(2, 3, n_users),
            'device_mobile_ratio': np.random.beta(3, 2, n_users),
            'time_since_registration_days': np.random.randint(0, 365, n_users)
        }
        
        self.features_df = pd.DataFrame(data).set_index('user_id')
        
        # Сохранение для будущего использования
        self.features_df.to_csv(self.data_path)
        logger.info(f"Mock Feature Store created: {self.data_path}")
    
    def get_user_features(self, user_id: int) -> Optional[pd.Series]:
        """Получение фич для конкретного пользователя"""
        try:
            if user_id in self.features_df.index:
                return self.features_df.loc[user_id]
            else:
                # Возвращаем средние значения для новых пользователей
                logger.warning(f"User {user_id} not found, returning average features")
                return self.features_df.mean()
        except Exception as e:
            logger.error(f"Error getting features for user {user_id}: {e}")
            return None
    
    def update_user_features(self, user_id: int, features: Dict[str, Any]):
        """Обновление фич пользователя"""
        try:
            # В реальном Feature Store здесь была бы логика обновления
            # Для демо просто логируем
            logger.info(f"Feature update for user {user_id}: {features}")
            return True
        except Exception as e:
            logger.error(f"Error updating features: {e}")
            return False
    
    def is_available(self) -> bool:
        """Проверка доступности Feature Store"""
        return self.features_df is not None and not self.features_df.empty
    
    def get_stats(self) -> Dict[str, Any]:
        """Статистика Feature Store"""
        if self.features_df is None:
            return {"error": "Feature Store not loaded"}
        
        return {
            "total_users": len(self.features_df),
            "features": list(self.features_df.columns),
            "memory_usage_mb": self.features_df.memory_usage(deep=True).sum() / 1024**2,
            "last_updated": pd.Timestamp.now().isoformat()
        }