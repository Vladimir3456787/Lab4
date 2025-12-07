import pandas as pd
import numpy as np
import sys
import io
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def process_raw_data(df):
    """Обработка сырых данных"""
    logger.info("Обработка данных...")
    
    # Удаляем дубликаты
    initial_shape = df.shape
    df = df.drop_duplicates()
    logger.info(f"Удалено дубликатов: {initial_shape[0] - df.shape[0]}")
    
    # Обработка пропусков
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        logger.info(f"Пропущенные значения: {missing_values[missing_values > 0].to_dict()}")
        
        # Заполняем пропуски
        if 'brand' in df.columns:
            df['brand'] = df['brand'].fillna('unknown')
        if 'price' in df.columns:
            df['price'] = df['price'].fillna(df['price'].median())
    
    # Добавляем временные признаки
    df['hour'] = df['event_time'].dt.hour
    df['day_of_week'] = df['event_time'].dt.dayofweek
    df['day_of_month'] = df['event_time'].dt.day
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Логируем статистику
    logger.info(f"Обработано данных: {df.shape}")
    logger.info(f"Уникальных пользователей: {df['user_id'].nunique()}")
    logger.info(f"Уникальных товаров: {df['product_id'].nunique()}")
    logger.info(f"Диапазон дат: {df['event_time'].min()} - {df['event_time'].max()}")
    
    return df

def download_october_data(sample_size=200000):
    """Загрузка данных из локального файла"""
    logger.info(" Загрузка данных из локального файла...")
    
    # ПРАВИЛЬНЫЙ ПУТЬ ВНУТРИ КОНТЕЙНЕРА
    local_file_path = Path("/app/local_data/2019-Oct.csv")
    
    if not local_file_path.exists():
        logger.error(f"Локальный файл не найден по пути: {local_file_path}")
        raise FileNotFoundError(f"Локальный файл не найден: {local_file_path}")
    
    try:
        # Читаем файл с диска
        logger.info(f"Чтение файла: {local_file_path}")
        # Используем оптимизированное чтение для больших файлов
        df = pd.read_csv(
            local_file_path,
            nrows=sample_size,
            parse_dates=['event_time'],
            dtype={
                'event_type': 'category',
                'product_id': 'str',
                'category_id': 'str',
                'brand': 'str',
                'user_id': 'str',
                'user_session': 'str'
            },
            usecols=['event_time', 'event_type', 'product_id', 'category_id',
                     'brand', 'price', 'user_id', 'user_session']
        )
        logger.info(f" Данные загружены. Размер: {df.shape}")
        
        # Обработка данных
        df = process_raw_data(df)
        
        # Сохраняем
        processed_dir = Path("/app/data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        processed_path = processed_dir / "october_processed.parquet"
        
        # Сохраняем с оптимизацией
        df.to_parquet(
            processed_path, 
            index=False,
            compression='snappy',
            engine='pyarrow'
        )
        
        # Детальная статистика
        print("\n" + "="*70)
        print(" ДЕТАЛЬНАЯ СТАТИСТИКА РЕАЛЬНЫХ ДАННЫХ (из локального файла)")
        print("="*70)
        print(f"Всего событий: {df.shape[0]:,}")
        print(f"Уникальных пользователей: {df['user_id'].nunique():,}")
        print(f"Уникальных товаров: {df['product_id'].nunique():,}")
        print(f"Уникальных категорий: {df['category_id'].nunique():,}")
        print(f"Уникальных брендов: {df['brand'].nunique():,}")
        
        print(f"\n Распределение событий:")
        event_stats = df['event_type'].value_counts()
        for event, count in event_stats.items():
            percentage = (count / len(df)) * 100
            print(f"  {event:10s}: {count:8,} ({percentage:5.1f}%)")
        
        print(f"\n Статистика цен:")
        print(f"  Медиана: ${df['price'].median():.2f}")
        print(f"  Среднее: ${df['price'].mean():.2f}")
        print(f"  Минимум: ${df['price'].min():.2f}")
        print(f"  Максимум: ${df['price'].max():.2f}")
        
        print(f"\n Сохранено в: {processed_path}")
        print("="*70)
        
        return df
        
    except Exception as e:
        logger.error(f"Ошибка при чтении локального файла: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    try:
        # Увеличиваем размер выборки для более качественного обучения
        df = download_october_data(sample_size=200000)
        print("\n ДАННЫЕ УСПЕШНО ЗАГРУЖЕНЫ ИЗ ЛОКАЛЬНОГО ФАЙЛА!")
    except Exception as e:
        logger.error(f"Критическая ошибка: {str(e)}")
        print("\n НЕ УДАЛОСЬ ЗАГРУЗИТЬ ДАННЫЕ")
        sys.exit(1)