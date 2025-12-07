"""
Пример использования Feature Store для обучения и инференса
"""

import pandas as pd
from feast import FeatureStore
from datetime import datetime
import json

def demonstrate_feature_store_workflow():
    """Демонстрация полного workflow Feature Store"""
    
    # Инициализация Feature Store
    fs = FeatureStore(repo_path=".")
    
    print("="*60)
    print("ДЕМОНСТРАЦИЯ FEAST FEATURE STORE")
    print("="*60)
    
    # 1. Получение фич для обучения (offline)
    print("\n1. ПОЛУЧЕНИЕ ФИЧ ДЛЯ ОБУЧЕНИЯ (OFFLINE):")
    
    # Создаем entity dataframe для обучения
    training_entities = pd.DataFrame({
        'user': ['user_1', 'user_2', 'user_3'],
        'product': ['product_a', 'product_b', 'product_c'],
        'event_timestamp': pd.to_datetime(['2019-10-01', '2019-10-02', '2019-10-03'])
    })
    
    # Получаем исторические фичи
    training_df = fs.get_historical_features(
        entity_df=training_entities,
        features=[
            "user_stats:user_total_events",
            "user_stats:user_total_purchases",
            "user_stats:user_avg_price",
            "product_stats:product_total_views",
            "product_stats:product_view_to_purchase_rate",
            "user_product_interactions:user_product_view_count",
            "user_product_interactions:user_product_preference_score",
        ]
    ).to_df()
    
    print(f"Размер тренировочных данных: {training_df.shape}")
    print("Колонки:", list(training_df.columns))
    print("\nПервые строки:")
    print(training_df.head())
    
    # 2. Получение фич для инференса (online)
    print("\n" + "-"*60)
    print("2. ПОЛУЧЕНИЕ ФИЧ ДЛЯ ИНФЕРЕНСА (ONLINE):")
    
    # Получаем online фичи для реального предсказания
    online_features = fs.get_online_features(
        entity_rows=[
            {"user": "user_1", "product": "product_a"},
            {"user": "user_2", "product": "product_b"},
        ],
        features=[
            "user_stats:user_total_events",
            "user_stats:user_total_purchases",
            "product_stats:product_total_views",
            "user_product_interactions:user_product_view_count",
        ]
    ).to_dict()
    
    print("Online фичи для инференса:")
    for key, values in online_features.items():
        print(f"  {key}: {values}")
    
    # 3. Проверка согласованности
    print("\n" + "-"*60)
    print("3. ПРОВЕРКА СОГЛАСОВАННОСТИ OFFLINE/ONLINE:")
    
    # Сравниваем значения для user_1
    user_1_offline = training_df[training_df['user'] == 'user_1'].iloc[0]
    
    print("\nСравнение для user_1:")
    print(f"{'Фича':40s} {'Offline':>15s} {'Online':>15s} {'Совпадение':>12s}")
    print("-"*85)
    
    for feat in ['user_total_events', 'user_total_purchases']:
        offline_val = user_1_offline.get(feat)
        online_val = online_features.get(feat, [None])[0] if feat in online_features else None
        
        if offline_val is not None and online_val is not None:
            match = "✅" if abs(offline_val - online_val) < 0.001 else "❌"
            print(f"{feat:40s} {offline_val:15.4f} {online_val:15.4f} {match:>12s}")
        else:
            print(f"{feat:40s} {'N/A':>15s} {'N/A':>15s} {'N/A':>12s}")
    
    # 4. Преимущества Feature Store
    print("\n" + "-"*60)
    print("4. ПРЕИМУЩЕСТВА FEAST FEATURE STORE:")
    
    benefits = [
        "✓ Единый источник истины для фич",
        "✓ Автоматическая согласованность offline/online",
        "✓ Централизованное управление фичами",
        "✓ Воспроизводимость экспериментов",
        "✓ Автоматическое обновление фич через TTL",
        "✓ Поддержка on-demand вычислений",
        "✓ Интеграция с MLflow и другими инструментами",
        "✓ Масштабируемость через Redis/PostgreSQL"
    ]
    
    for benefit in benefits:
        print(f"  {benefit}")
    
    print("\n" + "="*60)
    print("ВЫВОД: Feature Store решает проблему feature mismatching через:")
    print("1. Единую дефиницию фич")
    print("2. Автоматическую материализацию")
    print("3. Гарантированную согласованность")
    print("="*60)
    
    # Сохраняем демонстрационные данные
    demo_data = {
        'timestamp': datetime.now().isoformat(),
        'training_data_sample': training_df.head().to_dict('records'),
        'online_features_sample': online_features,
        'consistency_check': {
            'user_1': {
                'offline': user_1_offline[['user_total_events', 'user_total_purchases']].to_dict(),
                'online': {k: v[0] for k, v in online_features.items() if k in ['user_total_events', 'user_total_purchases']}
            }
        }
    }
    
    with open('feature_store_demo.json', 'w') as f:
        json.dump(demo_data, f, indent=2, default=str)
    
    return True

if __name__ == "__main__":
    demonstrate_feature_store_workflow()