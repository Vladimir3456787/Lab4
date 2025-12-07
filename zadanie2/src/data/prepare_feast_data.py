import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def prepare_feast_data():
    """Подготовка данных для Feast Feature Store"""
    logger.info("Подготовка данных для Feast...")
    
    # Загружаем исходные данные
    data_path = Path("/app/local_data/2019-Oct.csv")
    if not data_path.exists():
        logger.error("Данные не найдены!")
        return
    
    # Читаем данные с оптимизацией
    df = pd.read_csv(
        data_path,
        nrows=100000,  # Для демонстрации
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
    
    # ========== 1. Пользовательские фичи ==========
    logger.info("Создание пользовательских фич...")
    
    user_stats = df.groupby('user_id').agg({
        'event_time': ['count', 'max'],
        'event_type': lambda x: (x == 'purchase').sum(),
        'price': 'mean'
    }).reset_index()
    
    user_stats.columns = ['user_id', 'total_events', 'last_activity', 'total_purchases', 'avg_price']
    
    # Дополнительные пользовательские фичи
    user_session_count = df.groupby('user_id')['user_session'].nunique().reset_index()
    user_session_count.columns = ['user_id', 'session_count']
    
    user_stats = user_stats.merge(user_session_count, on='user_id')
    
    # Подсчет по типам событий
    event_counts = df.groupby(['user_id', 'event_type']).size().unstack(fill_value=0)
    event_counts = event_counts.rename(columns={
        'view': 'total_views',
        'cart': 'total_carts',
        'purchase': 'total_purchases_calc'
    })
    
    user_stats = user_stats.merge(event_counts, left_on='user_id', right_index=True)
    
    # Вычисление частоты покупок
    user_stats['purchase_frequency'] = user_stats['total_purchases_calc'] / user_stats['total_events']
    
    # Дни с последней активности
    max_date = df['event_time'].max()
    user_stats['days_since_last_activity'] = (
        max_date - user_stats['last_activity']
    ).dt.days
    
    # Форматирование для Feast
    user_features = user_stats[[
        'user_id', 'last_activity', 'total_events', 'total_purchases_calc',
        'total_views', 'total_carts', 'avg_price', 'purchase_frequency',
        'session_count', 'days_since_last_activity'
    ]].copy()
    
    user_features = user_features.rename(columns={
        'last_activity': 'event_timestamp',
        'total_purchases_calc': 'user_total_purchases',
        'total_events': 'user_total_events',
        'total_views': 'user_total_views',
        'total_carts': 'user_total_carts',
        'avg_price': 'user_avg_price',
        'purchase_frequency': 'user_purchase_frequency',
        'session_count': 'user_session_count',
        'days_since_last_activity': 'days_since_last_activity'
    })
    
    user_features['created_timestamp'] = datetime.now()
    
    # ========== 2. Товарные фичи ==========
    logger.info("Создание товарных фич...")
    
    product_stats = df.groupby('product_id').agg({
        'event_time': ['count', 'max'],
        'event_type': lambda x: ((x == 'purchase').sum(), (x == 'view').sum()),
        'price': 'mean'
    }).reset_index()
    
    product_stats.columns = [
        'product_id', 'total_events', 'last_view',
        'purchase_view_tuple', 'avg_price'
    ]
    
    # Разделение tuple
    product_stats['product_total_purchases'] = product_stats['purchase_view_tuple'].apply(lambda x: x[0])
    product_stats['product_total_views'] = product_stats['purchase_view_tuple'].apply(lambda x: x[1])
    
    product_carts = df[df['event_type'] == 'cart'].groupby('product_id').size().reset_index()
    product_carts.columns = ['product_id', 'product_total_carts']
    
    product_stats = product_stats.merge(product_carts, on='product_id', how='left').fillna(0)
    
    # Вычисление конверсии
    product_stats['product_view_to_purchase_rate'] = (
        product_stats['product_total_purchases'] / product_stats['product_total_views'].replace(0, 1)
    )
    
    # Скор популярности
    product_stats['product_popularity_score'] = (
        0.5 * product_stats['product_total_views'] +
        0.3 * product_stats['product_total_carts'] +
        0.2 * product_stats['product_total_purchases']
    )
    
    product_stats['days_since_last_view'] = (max_date - product_stats['last_view']).dt.days
    
    product_features = product_stats[[
        'product_id', 'last_view', 'product_total_views',
        'product_total_purchases', 'product_total_carts',
        'product_view_to_purchase_rate', 'avg_price',
        'product_popularity_score', 'days_since_last_view'
    ]].copy()
    
    product_features = product_features.rename(columns={
        'last_view': 'event_timestamp',
        'avg_price': 'product_avg_price'
    })
    
    product_features['created_timestamp'] = datetime.now()
    
    # ========== 3. Взаимодействия пользователь-товар ==========
    logger.info("Создание фич взаимодействий...")
    
    interactions = df.groupby(['user_id', 'product_id']).agg({
        'event_time': ['count', 'max'],
        'event_type': lambda x: {
            'views': (x == 'view').sum(),
            'carts': (x == 'cart').sum(),
            'purchases': (x == 'purchase').sum()
        }
    }).reset_index()
    
    interactions.columns = ['user_id', 'product_id', 'interaction_count', 'last_interaction', 'event_counts']
    
    # Разделение счетчиков событий
    interactions['user_product_view_count'] = interactions['event_counts'].apply(lambda x: x['views'])
    interactions['user_product_cart_count'] = interactions['event_counts'].apply(lambda x: x['carts'])
    interactions['user_product_purchase_count'] = interactions['event_counts'].apply(lambda x: x['purchases'])
    
    # Время с последнего взаимодействия
    interactions['user_product_last_interaction_hours'] = (
        max_date - interactions['last_interaction']
    ).dt.total_seconds() / 3600
    
    # Скор предпочтения
    interactions['user_product_preference_score'] = (
        0.1 * interactions['user_product_view_count'] +
        0.3 * interactions['user_product_cart_count'] +
        0.6 * interactions['user_product_purchase_count']
    )
    
    interaction_features = interactions[[
        'user_id', 'product_id', 'last_interaction',
        'user_product_view_count', 'user_product_cart_count',
        'user_product_purchase_count', 'user_product_last_interaction_hours',
        'user_product_preference_score'
    ]].copy()
    
    interaction_features = interaction_features.rename(columns={
        'last_interaction': 'event_timestamp'
    })
    
    interaction_features['created_timestamp'] = datetime.now()
    
    # ========== 4. Сохранение данных ==========
    logger.info("Сохранение данных для Feast...")
    
    # Создаем директории
    feast_dir = Path("/app/data/feast")
    feast_dir.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем в Parquet
    user_features.to_parquet(feast_dir / "user_stats.parquet", index=False)
    product_features.to_parquet(feast_dir / "product_stats.parquet", index=False)
    interaction_features.to_parquet(feast_dir / "user_product_interactions.parquet", index=False)
    
    # Сохраняем метаданные
    metadata = {
        'preparation_date': datetime.now().isoformat(),
        'user_features_count': len(user_features),
        'product_features_count': len(product_features),
        'interaction_features_count': len(interaction_features),
        'feature_columns': {
            'user_stats': list(user_features.columns),
            'product_stats': list(product_features.columns),
            'interactions': list(interaction_features.columns)
        }
    }
    
    import json
    with open(feast_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"Данные сохранены в {feast_dir}")
    logger.info(f"  • Пользовательские фичи: {len(user_features)} записей")
    logger.info(f"  • Товарные фичи: {len(product_features)} записей")
    logger.info(f"  • Взаимодействия: {len(interaction_features)} записей")
    
    return True

if __name__ == "__main__":
    try:
        success = prepare_feast_data()
        if success:
            print("\n" + "="*60)
            print("✅ ДАННЫЕ ДЛЯ FEAST ПОДГОТОВЛЕНЫ УСПЕШНО!")
            print("="*60)
            print("Локальные файлы:")
            print("  • user_stats.parquet")
            print("  • product_stats.parquet")
            print("  • user_product_interactions.parquet")
            print("="*60)
        else:
            print("\n❌ ОШИБКА ПРИ ПОДГОТОВКЕ ДАННЫХ")
    except Exception as e:
        logger.error(f"Критическая ошибка: {str(e)}")
        import traceback
        traceback.print_exc()