import pandas as pd
import numpy as np
from datetime import datetime
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_data():
    """Подготовка данных для обучения"""
    logger.info("Подготовка данных для задания 3...")
    
    # Загрузка данных
    data_path = Path("/app/local_data/2019-Oct.csv")
    if not data_path.exists():
        logger.error("Данные не найдены!")
        return None
    
    # Чтение с оптимизацией
    df = pd.read_csv(
        data_path,
        nrows=150000,
        parse_dates=['event_time'],
        dtype={
            'event_type': 'category',
            'product_id': 'str',
            'category_id': 'str',
            'brand': 'str',
            'user_id': 'str',
            'user_session': 'str'
        }
    )
    
    logger.info(f"Загружено данных: {df.shape}")
    
    # Создание целевой переменной (покупка = 1, остальное = 0)
    df['target'] = (df['event_type'] == 'purchase').astype(int)
    
    # Создание признаков
    df['hour'] = df['event_time'].dt.hour
    df['day_of_week'] = df['event_time'].dt.dayofweek
    df['day_of_month'] = df['event_time'].dt.day
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Частотное кодирование категориальных признаков
    for col in ['brand', 'category_id', 'product_id']:
        if col in df.columns:
            freq_encoding = df[col].value_counts(normalize=True)
            df[f'{col}_freq'] = df[col].map(freq_encoding)
    
    # Статистики пользователей
    user_stats = df.groupby('user_id').agg({
        'price': ['mean', 'std', 'count'],
        'event_type': lambda x: (x == 'purchase').sum()
    }).fillna(0)
    
    user_stats.columns = ['user_price_mean', 'user_price_std', 'user_event_count', 'user_purchase_count']
    user_stats['user_purchase_rate'] = user_stats['user_purchase_count'] / user_stats['user_event_count']
    user_stats['user_purchase_rate'] = user_stats['user_purchase_rate'].fillna(0)
    
    df = df.merge(user_stats, left_on='user_id', right_index=True, how='left')
    
    # Выбор признаков для обучения
    features = [
        'price', 'hour', 'day_of_week', 'day_of_month', 'is_weekend',
        'brand_freq', 'category_id_freq', 'product_id_freq',
        'user_price_mean', 'user_price_std', 'user_event_count', 
        'user_purchase_count', 'user_purchase_rate'
    ]
    
    # Отбираем только существующие признаки
    existing_features = [f for f in features if f in df.columns]
    
    X = df[existing_features].fillna(0)
    y = df['target']
    
    # Разделение на train/test
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Сохранение данных
    import joblib
    data_dir = Path("/app/artifacts/models")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    joblib.dump({
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_names': existing_features
    }, data_dir / "processed_data.joblib")
    
    logger.info(f"Данные подготовлены. Признаков: {len(existing_features)}")
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    logger.info(f"Баланс классов: {y_train.value_counts(normalize=True).to_dict()}")
    
    print("\n" + "="*60)
    print("✅ ДАННЫЕ ПОДГОТОВЛЕНЫ ДЛЯ ОБУЧЕНИЯ")
    print("="*60)
    print(f"Всего признаков: {len(existing_features)}")
    print(f"Размер обучающей выборки: {X_train.shape}")
    print(f"Размер тестовой выборки: {X_test.shape}")
    print(f"Данные сохранены в: {data_dir / 'processed_data.joblib'}")
    print("="*60)
    
    return X_train, X_test, y_train, y_test, existing_features

if __name__ == "__main__":
    prepare_data()