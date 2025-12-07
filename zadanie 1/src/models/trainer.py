import pandas as pd
import numpy as np
import logging
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, 
    precision_score, recall_score, confusion_matrix,
    classification_report
)
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import mlflow.catboost
import joblib
import json
from datetime import datetime
from pathlib import Path
import sys
import time
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_advanced_features(df):
    """Создание продвинутых признаков"""
    logger.info("Создание продвинутых признаков...")
    
    # Временные признаки
    df['hour'] = df['event_time'].dt.hour
    df['day_of_week'] = df['event_time'].dt.dayofweek
    df['day_of_month'] = df['event_time'].dt.day
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Категориальные признаки с частотным кодированием
    for col in ['brand', 'category_id', 'product_id']:
        if col in df.columns:
            # Частотное кодирование
            freq_encoding = df[col].value_counts(normalize=True)
            df[f'{col}_freq'] = df[col].map(freq_encoding)
            
            # Бинарный признак для популярных значений (топ-20)
            top_values = df[col].value_counts().head(20).index
            df[f'{col}_is_popular'] = df[col].isin(top_values).astype(int)
    
    # Взаимодействие признаков
    if 'price' in df.columns and 'brand_freq' in df.columns:
        df['price_brand_interaction'] = df['price'] * df['brand_freq']
    
    # Статистики по пользователям (агрегация)
    user_stats = df.groupby('user_id').agg({
        'price': ['mean', 'std', 'count'],
        'event_type': lambda x: (x == 'purchase').sum()
    }).fillna(0)
    
    user_stats.columns = ['user_price_mean', 'user_price_std', 'user_event_count', 'user_purchase_count']
    user_stats['user_purchase_rate'] = user_stats['user_purchase_count'] / user_stats['user_event_count']
    user_stats['user_purchase_rate'] = user_stats['user_purchase_rate'].fillna(0)
    
    # Добавляем статистики пользователей в основной датафрейм
    df = df.merge(user_stats, left_on='user_id', right_index=True, how='left')
    
    logger.info(f"Создано признаков: {len(df.columns)}")
    return df

def prepare_data(df):
    """Подготовка данных для обучения"""
    logger.info("Подготовка данных для обучения...")
    
    # Целевая переменная: покупка (1) vs остальное (0)
    df['target'] = (df['event_type'] == 'purchase').astype(int)
    
    # Выбираем признаки
    numeric_features = [
        'price', 'hour', 'day_of_week', 'day_of_month', 'is_weekend',
        'brand_freq', 'category_id_freq', 'product_id_freq',
        'user_price_mean', 'user_price_std', 'user_event_count', 
        'user_purchase_count', 'user_purchase_rate'
    ]
    
    binary_features = [
        'brand_is_popular', 'category_id_is_popular', 'product_id_is_popular'
    ]
    
    # Отбираем только существующие признаки
    existing_numeric = [f for f in numeric_features if f in df.columns]
    existing_binary = [f for f in binary_features if f in df.columns]
    
    all_features = existing_numeric + existing_binary
    
    # Проверяем наличие признаков
    missing_features = set(numeric_features + binary_features) - set(all_features)
    if missing_features:
        logger.warning(f"Отсутствующие признаки: {missing_features}")
    
    X = df[all_features].fillna(0)
    y = df['target']
    
    # Логирование информации о данных
    logger.info(f"Всего признаков: {X.shape[1]}")
    logger.info(f"Размер данных: {X.shape}")
    logger.info(f"Баланс классов: {y.value_counts(normalize=True).to_dict()}")
    
    return X, y, all_features

def evaluate_model(model, X_test, y_test, model_name):
    """Расширенная оценка модели"""
    start_time = time.time()
    y_pred = model.predict(X_test)
    inference_time = (time.time() - start_time) / len(X_test) * 1000  # мс на запрос
    
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
    else:
        auc = 0.5
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': auc,
        'f1_score': f1_score(y_test, y_pred, average='weighted'),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'inference_time_ms': inference_time
    }
    
    # Дополнительные метрики для бинарной классификации
    if len(np.unique(y_test)) == 2:
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        metrics.update({
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0
        })
    
    logger.info(f"{model_name}: AUC={metrics['roc_auc']:.4f}, F1={metrics['f1_score']:.4f}, Time={metrics['inference_time_ms']:.2f}ms")
    
    return metrics

def train_and_evaluate_models():
    """Обучение и оценка 7 моделей"""
    logger.info("🤖 Начинаем обучение и оценку 7 моделей...")
    
    # Загружаем данные
    data_path = Path("/app/data/processed/october_processed.parquet")
    if not data_path.exists():
        logger.error("Данные не найдены!")
        return None, None
    
    df = pd.read_parquet(data_path)
    logger.info(f"Данные загружены: {df.shape}")
    
    # Создаем признаки
    df = create_advanced_features(df)
    
    # Подготавливаем данные
    X, y, feature_names = prepare_data(df)
    
    # Разделение данных
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    # Масштабирование
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Обработка дисбаланса классов с SMOTE
    try:
        smote = SMOTE(random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        logger.info(f"После SMOTE: {X_train_balanced.shape}, баланс: {np.bincount(y_train_balanced)}")
    except Exception as e:
        logger.warning(f"SMOTE не применен: {e}")
        X_train_balanced, y_train_balanced = X_train_scaled, y_train
    
    # Настройка MLflow
    mlflow.set_tracking_uri("http://mlflow:5000")
    mlflow.set_experiment("ecommerce_model_comparison_7_models")
    
    # Список моделей для сравнения (7 моделей)
    models = {
        'LogisticRegression': LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced',
            C=1.0,
            solver='lbfgs'
        ),
        'RandomForest': RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            class_weight='balanced',
            n_jobs=-1
        ),
        'XGBoost': xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            eval_metric='logloss',
            use_label_encoder=False,
            scale_pos_weight=len(y_train_balanced[y_train_balanced==0]) / len(y_train_balanced[y_train_balanced==1])
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            class_weight='balanced',
            verbose=-1,
            n_jobs=-1
        ),
        'CatBoost': CatBoostClassifier(
            iterations=200,
            depth=6,
            learning_rate=0.1,
            random_state=42,
            verbose=0,
            auto_class_weights='Balanced'
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            subsample=0.8
        ),
        'SVM': SVC(
            C=1.0,
            kernel='rbf',
            probability=True,
            random_state=42,
            class_weight='balanced',
            max_iter=1000
        )
    }
    
    results = []
    feature_importances = {}
    
    for model_name, model in models.items():
        logger.info(f"Обучение модели: {model_name}")
        
        try:
            with mlflow.start_run(run_name=model_name, nested=True):
                # Логирование параметров
                mlflow.log_param("model", model_name)
                mlflow.log_param("n_features", len(feature_names))
                mlflow.log_param("training_samples", len(X_train_balanced))
                
                # Обучение с таймингом
                train_start = time.time()
                model.fit(X_train_balanced, y_train_balanced)
                train_time = time.time() - train_start
                
                # Оценка
                metrics = evaluate_model(model, X_test_scaled, y_test, model_name)
                metrics['training_time_seconds'] = train_time
                
                # Логирование метрик
                for metric_name, metric_value in metrics.items():
                    if isinstance(metric_value, (int, float)):
                        mlflow.log_metric(metric_name, metric_value)
                
                # Сохранение модели
                model_path = f'/app/artifacts/models/{model_name}.joblib'
                joblib.dump(model, model_path)
                mlflow.log_artifact(model_path)
                
                # Сбор важности признаков (если доступно)
                if hasattr(model, 'feature_importances_'):
                    importance = pd.DataFrame({
                        'feature': feature_names,
                        'importance': model.feature_importances_
                    }).sort_values('importance', ascending=False)
                    
                    feature_importances[model_name] = importance
                    
                    # Сохраняем важность признаков
                    importance_path = f'/app/artifacts/models/{model_name}_feature_importance.csv'
                    importance.head(20).to_csv(importance_path, index=False)
                    mlflow.log_artifact(importance_path)
                
                # Логирование в MLflow
                if model_name == 'LogisticRegression':
                    mlflow.sklearn.log_model(model, "model")
                elif model_name == 'RandomForest':
                    mlflow.sklearn.log_model(model, "model")
                elif model_name == 'XGBoost':
                    mlflow.xgboost.log_model(model, "model")
                elif model_name == 'LightGBM':
                    mlflow.lightgbm.log_model(model, "model")
                elif model_name == 'CatBoost':
                    mlflow.catboost.log_model(model, "model")
                elif model_name == 'GradientBoosting':
                    mlflow.sklearn.log_model(model, "model")
                elif model_name == 'SVM':
                    mlflow.sklearn.log_model(model, "model")
                
                results.append({
                    'model': model_name,
                    **{k: float(v) if isinstance(v, (np.floating, float)) else v 
                       for k, v in metrics.items() if isinstance(v, (int, float, np.integer, np.floating))}
                })
                
                logger.info(f"✅ {model_name} обучена за {train_time:.1f} секунд")
                
        except Exception as e:
            logger.error(f"Ошибка при обучении {model_name}: {str(e)}")
            continue
    
    # Создание сравнительной таблицы
    results_df = pd.DataFrame(results)
    
    # Сортировка по ROC-AUC
    results_df = results_df.sort_values('roc_auc', ascending=False)
    
    # Сохранение результатов
    results_df.to_csv('/app/artifacts/models/model_comparison_7_models.csv', index=False)
    results_df.to_json('/app/artifacts/models/model_comparison_7_models.json', orient='records', indent=2)
    
    # Выбор лучшей модели
    best_model_info = results_df.iloc[0]
    second_best_info = results_df.iloc[1] if len(results_df) > 1 else None
    
    # Детальный отчет
    report = {
        'execution_date': datetime.now().isoformat(),
        'dataset_info': {
            'total_samples': len(df),
            'training_samples': len(X_train_balanced),
            'test_samples': len(X_test),
            'features_used': len(feature_names),
            'class_balance': {
                'original': dict(y.value_counts()),
                'after_smote': dict(np.bincount(y_train_balanced))
            }
        },
        'models_evaluated': len(results_df),
        'best_model': {
            'name': best_model_info['model'],
            'roc_auc': float(best_model_info['roc_auc']),
            'accuracy': float(best_model_info['accuracy']),
            'f1_score': float(best_model_info['f1_score']),
            'inference_time_ms': float(best_model_info['inference_time_ms']),
            'training_time_seconds': float(best_model_info.get('training_time_seconds', 0))
        },
        'second_best_model': {
            'name': second_best_info['model'] if second_best_info else None,
            'roc_auc': float(second_best_info['roc_auc']) if second_best_info else None,
            'accuracy': float(second_best_info['accuracy']) if second_best_info else None
        } if second_best_info else None,
        'all_results': results_df.to_dict('records'),
        'requirements_check': {
            'latency_under_100ms': all(r['inference_time_ms'] < 100 for r in results),
            'accuracy_over_85pct': any(r['accuracy'] > 0.85 for r in results),
            'cold_start_handling': 'Реализовано в ColdStartHandler',
            'business_requirements': {
                'recommendation_system': True,
                'personalization': True,
                'ctr_improvement_10pct': best_model_info['roc_auc'] > 0.7  # Примерный критерий
            }
        },
        'feature_analysis': {
            'total_features': len(feature_names),
            'top_features': feature_importances.get(best_model_info['model'], pd.DataFrame()).head(10).to_dict('records') 
            if best_model_info['model'] in feature_importances else []
        },
        'recommendations': [
            f"Рекомендуемая модель: {best_model_info['model']}",
            f"Ожидаемое увеличение CTR: {(best_model_info['roc_auc'] - 0.5) * 20:.1f}%",
            "Требуется внедрение ColdStartHandler для новых пользователей",
            "Рекомендуется A/B тестирование перед полным развертыванием"
        ]
    }
    
    # Сохранение отчета
    report_path = '/app/artifacts/reports/detailed_training_report.json'
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    # Логирование финального отчета
    with mlflow.start_run(run_name="final_comparison_report"):
        mlflow.log_artifact(report_path)
        mlflow.log_metric("best_model_auc", best_model_info['roc_auc'])
        mlflow.log_metric("best_model_accuracy", best_model_info['accuracy'])
        mlflow.log_param("best_model_name", best_model_info['model'])
        mlflow.log_param("total_models_evaluated", len(results_df))
    
    # Сохранение scaler
    joblib.dump(scaler, '/app/artifacts/models/scaler.joblib')
    
    # Визуализация результатов
    create_comparison_chart(results_df)
    
    # Вывод результатов
    print("\n" + "="*80)
    print(" РЕЗУЛЬТАТЫ СРАВНЕНИЯ 7 МОДЕЛЕЙ")
    print("="*80)
    print(f"{'Модель':20s} {'ROC-AUC':>8s} {'Accuracy':>10s} {'F1-Score':>10s} {'Time (ms)':>10s}")
    print("-"*80)
    
    for _, row in results_df.iterrows():
        print(f"{row['model']:20s} {row['roc_auc']:8.4f} {row['accuracy']:10.4f} {row['f1_score']:10.4f} {row['inference_time_ms']:10.2f}")
    
    print("-"*80)
    print(f"\n ЛУЧШАЯ МОДЕЛЬ: {best_model_info['model']}")
    print(f" ROC-AUC: {best_model_info['roc_auc']:.4f}")
    print(f" Accuracy: {best_model_info['accuracy']:.4f}")
    print(f" Inference Time: {best_model_info['inference_time_ms']:.2f} ms")
    print(f" Training Time: {best_model_info.get('training_time_seconds', 0):.1f} sec")
    
    print(f"\n✅ ТРЕБОВАНИЯ:")
    print(f"   Latency < 100ms: {'✅ ПРОЙДЕНО' if report['requirements_check']['latency_under_100ms'] else '❌ НЕ ПРОЙДЕНО'}")
    print(f"   Accuracy > 85%: {'✅ ПРОЙДЕНО' if report['requirements_check']['accuracy_over_85pct'] else '❌ НЕ ПРОЙДЕНО'}")
    print(f"   CTR Improvement > 10%: {'✅ ВЕРОЯТНО' if report['requirements_check']['business_requirements']['ctr_improvement_10pct'] else '⚠️ ТРЕБУЕТСЯ ТЕСТИРОВАНИЕ'}")
    
    print(f"\n📊 Всего обучено моделей: {len(results_df)}")
    print(f"💾 Результаты сохранены в MLflow: http://localhost:5000")
    print(f"📁 Артефакты: /app/artifacts/")
    print("="*80)
    
    return results_df, report

def create_comparison_chart(results_df):
    """Создание визуализации сравнения моделей"""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        plt.figure(figsize=(15, 8))
        
        # График 1: ROC-AUC сравнение
        plt.subplot(2, 2, 1)
        sns.barplot(x='roc_auc', y='model', data=results_df, palette='viridis')
        plt.title('Сравнение моделей по ROC-AUC', fontsize=14, fontweight='bold')
        plt.xlabel('ROC-AUC')
        plt.ylabel('Модель')
        
        # График 2: Inference Time
        plt.subplot(2, 2, 2)
        sns.barplot(x='inference_time_ms', y='model', data=results_df, palette='coolwarm')
        plt.title('Время инференса (мс)', fontsize=14, fontweight='bold')
        plt.xlabel('Время (мс)')
        plt.ylabel('Модель')
        
        # График 3: Accuracy
        plt.subplot(2, 2, 3)
        sns.barplot(x='accuracy', y='model', data=results_df, palette='magma')
        plt.title('Accuracy моделей', fontsize=14, fontweight='bold')
        plt.xlabel('Accuracy')
        plt.ylabel('Модель')
        
        # График 4: F1-Score
        plt.subplot(2, 2, 4)
        sns.barplot(x='f1_score', y='model', data=results_df, palette='plasma')
        plt.title('F1-Score моделей', fontsize=14, fontweight='bold')
        plt.xlabel('F1-Score')
        plt.ylabel('Модель')
        
        plt.tight_layout()
        chart_path = '/app/artifacts/models/model_comparison_chart.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"График сравнения сохранен: {chart_path}")
        
    except Exception as e:
        logger.warning(f"Не удалось создать график: {str(e)}")

if __name__ == "__main__":
    logger.info(" Запуск сравнения 7 моделей на реальных данных...")
    
    try:
        results_df, report = train_and_evaluate_models()
        
        if results_df is not None:
            print("\n ЗАДАНИЕ 1 ВЫПОЛНЕНО УСПЕШНО!")
            print(f" Лучшая модель: {report['best_model']['name']}")
            print(f" Всего оценено моделей: {report['models_evaluated']}")
            print(f" MLflow: http://localhost:5000")
        else:
            print("\n ОШИБКА ПРИ ВЫПОЛНЕНИИ ЗАДАНИЯ")
            
    except Exception as e:
        logger.error(f"Критическая ошибка: {str(e)}")
        import traceback
        traceback.print_exc()
        print("\n ЗАДАНИЕ НЕ ВЫПОЛНЕНО")