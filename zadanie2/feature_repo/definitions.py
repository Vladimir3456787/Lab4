from datetime import timedelta
from feast import Entity, Feature, FeatureView, ValueType
from feast.infra.offline_stores.file_source import FileSource
from feast.data_source import RequestSource
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float64, Int64, String
import pandas as pd

# ========== Сущности ==========
user = Entity(
    name="user",
    value_type=ValueType.STRING,
    description="Пользователь e-commerce",
)

product = Entity(
    name="product",
    value_type=ValueType.STRING,
    description="Товар в магазине",
)

# ========== Источники данных ==========
# Источник для пользовательских фич
user_stats_source = FileSource(
    name="user_stats_source",
    path="/app/data/feast/user_stats.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Источник для товарных фич
product_stats_source = FileSource(
    name="product_stats_source",
    path="/app/data/feast/product_stats.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Источник для взаимодействий пользователь-товар
user_product_source = FileSource(
    name="user_product_source",
    path="/app/data/feast/user_product_interactions.parquet",
    timestamp_field="event_timestamp",
    created_timestamp_column="created_timestamp",
)

# Источник для запросов (online фичи)
request_source = RequestSource(
    name="prediction_request",
    schema=[
        ("user_id", String),
        ("product_id", String),
        ("hour", Int64),
        ("day_of_week", Int64),
        ("price", Float64),
    ]
)

# ========== Feature Views ==========
# Пользовательские фичи (offline)
user_stats_view = FeatureView(
    name="user_stats",
    entities=[user],
    ttl=timedelta(days=7),
    features=[
        Feature(name="user_total_events", dtype=Int64),
        Feature(name="user_total_purchases", dtype=Int64),
        Feature(name="user_total_views", dtype=Int64),
        Feature(name="user_total_carts", dtype=Int64),
        Feature(name="user_avg_price", dtype=Float64),
        Feature(name="user_purchase_frequency", dtype=Float64),
        Feature(name="user_session_count", dtype=Int64),
        Feature(name="days_since_last_activity", dtype=Int64),
    ],
    online=True,
    source=user_stats_source,
    tags={"team": "recommendations", "type": "user"},
)

# Товарные фичи (offline)
product_stats_view = FeatureView(
    name="product_stats",
    entities=[product],
    ttl=timedelta(days=7),
    features=[
        Feature(name="product_total_views", dtype=Int64),
        Feature(name="product_total_purchases", dtype=Int64),
        Feature(name="product_total_carts", dtype=Int64),
        Feature(name="product_view_to_purchase_rate", dtype=Float64),
        Feature(name="product_avg_price", dtype=Float64),
        Feature(name="product_popularity_score", dtype=Float64),
        Feature(name="days_since_last_view", dtype=Int64),
    ],
    online=True,
    source=product_stats_source,
    tags={"team": "recommendations", "type": "product"},
)

# Взаимодействия пользователь-товар
user_product_view = FeatureView(
    name="user_product_interactions",
    entities=[user, product],
    ttl=timedelta(days=30),
    features=[
        Feature(name="user_product_view_count", dtype=Int64),
        Feature(name="user_product_cart_count", dtype=Int64),
        Feature(name="user_product_purchase_count", dtype=Int64),
        Feature(name="user_product_last_interaction_hours", dtype=Int64),
        Feature(name="user_product_preference_score", dtype=Float64),
    ],
    online=True,
    source=user_product_source,
    tags={"team": "recommendations", "type": "interaction"},
)

# ========== On-Demand Features ==========
@on_demand_feature_view(
    sources=[
        request_source,
        user_stats_view,
        product_stats_view,
        user_product_view,
    ],
    schema=[
        Feature(name="hour_sin", dtype=Float64),
        Feature(name="hour_cos", dtype=Float64),
        Feature(name="is_weekend", dtype=Int64),
        Feature(name="price_normalized", dtype=Float64),
        Feature(name="user_product_affinity", dtype=Float64),
        Feature(name="composite_score", dtype=Float64),
    ]
)
def transformed_features(inputs: pd.DataFrame) -> pd.DataFrame:
    import numpy as np
    from datetime import datetime
    
    df = pd.DataFrame()
    
    # Циклические фичи для часа
    df["hour_sin"] = np.sin(2 * np.pi * inputs["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * inputs["hour"] / 24)
    
    # Признак выходного дня
    df["is_weekend"] = inputs["day_of_week"].isin([5, 6]).astype(int)
    
    # Нормализация цены (online версия)
    df["price_normalized"] = inputs["price"] / inputs["price"].max()
    
    # Аффинити пользователя к товару (онлайн расчет)
    user_product_views = inputs.get("user_product_view_count", 0)
    user_total_views = inputs.get("user_total_views", 1)
    product_total_views = inputs.get("product_total_views", 1)
    
    df["user_product_affinity"] = (
        user_product_views / 
        np.sqrt(user_total_views * product_total_views + 1)
    )
    
    # Композитный скор (пример бизнес-логики)
    df["composite_score"] = (
        0.3 * inputs.get("user_purchase_frequency", 0) +
        0.3 * inputs.get("product_view_to_purchase_rate", 0) +
        0.2 * df["user_product_affinity"] +
        0.2 * (1 - df["price_normalized"])
    )
    
    return df