import numpy as np
import pandas as pd
from collections import defaultdict
import joblib
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import get_logger

logger = get_logger(__name__)

class ColdStartHandler:
    """
    Обработчик холодного старта для новых пользователей и товаров
    """
    
    def __init__(self):
        self.user_stats = defaultdict(lambda: {'count': 0, 'last_seen': None})
        self.product_stats = defaultdict(lambda: {'count': 0, 'last_seen': None})
        self.category_stats = defaultdict(lambda: {'count': 0, 'avg_price': 0})
        self.global_avg_rating = 0.5
        self.min_events_for_warm = 3
        
    def initialize_from_data(self, X, y):
        """
        Инициализирует статистики из тренировочных данных
        """
        logger.info("Инициализация обработчика холодного старта...")
        
        # Здесь предполагается, что X содержит колонки user_id, product_id и т.д.
        # В реальном проекте нужно адаптировать под структуру данных
        
        # Пример: подсчет событий по пользователям
        if 'user_id' in X.columns:
            user_counts = X['user_id'].value_counts()
            for user_id, count in user_counts.items():
                self.user_stats[user_id]['count'] = count
        
        logger.info(f"Инициализировано {len(self.user_stats)} пользователей")
        logger.info(f"Порог для 'теплых' пользователей: {self.min_events_for_warm} событий")
        
        return self
    
    def is_cold_user(self, user_id):
        """
        Проверяет, является ли пользователь 'холодным'
        """
        count = self.user_stats.get(user_id, {}).get('count', 0)
        return count < self.min_events_for_warm
    
    def is_cold_product(self, product_id):
        """
        Проверяет, является ли товар 'холодным'
        """
        count = self.product_stats.get(product_id, {}).get('count', 0)
        return count < self.min_events_for_warm
    
    def get_fallback_recommendations(self, user_id=None, n_recommendations=5):
        """
        Возвращает fallback-рекомендации для холодных стартеров
        """
        # Самые популярные товары
        popular_products = [
            'popular_product_1',
            'popular_product_2', 
            'popular_product_3',
            'popular_product_4',
            'popular_product_5'
        ]
        
        # Если есть категория пользователя, можно рекомендовать по категории
        return popular_products[:n_recommendations]
    
    def update_user_activity(self, user_id):
        """
        Обновляет статистику пользователя
        """
        if user_id not in self.user_stats:
            self.user_stats[user_id] = {'count': 0, 'last_seen': None}
        
        self.user_stats[user_id]['count'] += 1
        self.user_stats[user_id]['last_seen'] = pd.Timestamp.now()
    
    def save(self, path):
        """Сохраняет обработчик на диск"""
        joblib.dump(self, path)
        logger.info(f"ColdStartHandler сохранен в {path}")
    
    @classmethod
    def load(cls, path):
        """Загружает обработчик с диска"""
        return joblib.load(path)