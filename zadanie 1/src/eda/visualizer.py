import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from pathlib import Path
import mlflow
import sys

sys.path.append(str(Path(__file__).parent.parent))

from utils.logger import get_logger

logger = get_logger(__name__)

def create_eda_plots(df):
    """
    Создает все необходимые визуализации для EDA
    """
    logger.info("Создание визуализаций...")
    
    # Настройка стилей
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # 1. Распределение типов событий
    fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    event_counts = df['event_type'].value_counts()
    ax1.bar(event_counts.index, event_counts.values)
    ax1.set_title('Распределение типов событий', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Тип события')
    ax1.set_ylabel('Количество')
    ax1.tick_params(axis='x', rotation=45)
    
    # Круговая диаграмма
    ax2.pie(event_counts.values, labels=event_counts.index, autopct='%1.1f%%')
    ax2.set_title('Процентное распределение событий', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('/app/artifacts/eda/event_distribution.png', dpi=300, bbox_inches='tight')
    mlflow.log_artifact('/app/artifacts/eda/event_distribution.png')
    plt.close()
    
    # 2. Временной ряд активности
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(15, 10))
    
    # По часам
    hourly_counts = df.groupby('hour').size()
    ax3.plot(hourly_counts.index, hourly_counts.values, marker='o', linewidth=2)
    ax3.set_title('Активность по часам', fontsize=14, fontweight='bold')
    ax3.set_xlabel('Час дня')
    ax3.set_ylabel('Количество событий')
    ax3.grid(True, alpha=0.3)
    
    # По дням недели
    if 'day_of_week' in df.columns:
        day_names = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']
        daily_counts = df.groupby('day_of_week').size()
        ax4.bar(range(len(day_names)), daily_counts.values)
        ax4.set_title('Активность по дням недели', fontsize=14, fontweight='bold')
        ax4.set_xlabel('День недели')
        ax4.set_ylabel('Количество событий')
        ax4.set_xticks(range(len(day_names)))
        ax4.set_xticklabels(day_names)
    
    plt.tight_layout()
    plt.savefig('/app/artifacts/eda/temporal_activity.png', dpi=300, bbox_inches='tight')
    mlflow.log_artifact('/app/artifacts/eda/temporal_activity.png')
    plt.close()
    
    # 3. Распределение цен
    fig3, (ax5, ax6) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Гистограмма цен (логарифмическая шкала)
    prices = df['price'].dropna()
    log_prices = np.log1p(prices[prices > 0])
    
    ax5.hist(log_prices, bins=50, edgecolor='black', alpha=0.7)
    ax5.set_title('Распределение цен (log scale)', fontsize=14, fontweight='bold')
    ax5.set_xlabel('log(цена + 1)')
    ax5.set_ylabel('Частота')
    
    # Box plot по типам событий
    if 'event_type' in df.columns:
        price_by_event = []
        events = []
        for event in df['event_type'].unique():
            event_prices = df[df['event_type'] == event]['price'].dropna()
            if len(event_prices) > 0:
                price_by_event.append(event_prices)
                events.append(event)
        
        ax6.boxplot(price_by_event, labels=events)
        ax6.set_title('Цены по типам событий', fontsize=14, fontweight='bold')
        ax6.set_xlabel('Тип события')
        ax6.set_ylabel('Цена')
        ax6.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('/app/artifacts/eda/price_distribution.png', dpi=300, bbox_inches='tight')
    mlflow.log_artifact('/app/artifacts/eda/price_distribution.png')
    plt.close()
    
    # 4. Анализ пользователей (топ N пользователей по активности)
    fig4, ax7 = plt.subplots(figsize=(15, 6))
    
    top_users = df['user_id'].value_counts().head(20)
    ax7.bar(range(len(top_users)), top_users.values)
    ax7.set_title('Топ-20 пользователей по активности', fontsize=14, fontweight='bold')
    ax7.set_xlabel('Пользователь (ID)')
    ax7.set_ylabel('Количество событий')
    ax7.set_xticks(range(len(top_users)))
    ax7.set_xticklabels([f'User {i}' for i in range(len(top_users))])
    
    plt.tight_layout()
    plt.savefig('/app/artifacts/eda/top_users.png', dpi=300, bbox_inches='tight')
    mlflow.log_artifact('/app/artifacts/eda/top_users.png')
    plt.close()
    
    # 5. Heatmap корреляции (если есть числовые признаки)
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        fig5, ax8 = plt.subplots(figsize=(10, 8))
        
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
                   ax=ax8, fmt='.2f', linewidths=0.5)
        ax8.set_title('Корреляционная матрица числовых признаков', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('/app/artifacts/eda/correlation_heatmap.png', dpi=300, bbox_inches='tight')
        mlflow.log_artifact('/app/artifacts/eda/correlation_heatmap.png')
        plt.close()
    
    # 6. Анализ холодного старта
    fig6, (ax9, ax10) = plt.subplots(1, 2, figsize=(15, 6))
    
    user_event_counts = df['user_id'].value_counts()
    
    # Распределение количества событий на пользователя
    ax9.hist(user_event_counts.values, bins=50, edgecolor='black', alpha=0.7, log=True)
    ax9.set_title('Распределение событий на пользователя', fontsize=14, fontweight='bold')
    ax9.set_xlabel('Количество событий')
    ax9.set_ylabel('Количество пользователей (log scale)')
    
    # Процент пользователей с N событиями
    single_event_users = (user_event_counts == 1).sum()
    few_event_users = (user_event_counts <= 3).sum()
    
    categories = ['1 событие', '≤3 событий', '>3 событий']
    counts = [
        single_event_users,
        few_event_users - single_event_users,
        len(user_event_counts) - few_event_users
    ]
    
    ax10.bar(categories, counts)
    ax10.set_title('Пользователи по активности', fontsize=14, fontweight='bold')
    ax10.set_ylabel('Количество пользователей')
    
    for i, count in enumerate(counts):
        ax10.text(i, count + 5, str(count), ha='center')
    
    plt.tight_layout()
    plt.savefig('/app/artifacts/eda/cold_start_analysis.png', dpi=300, bbox_inches='tight')
    mlflow.log_artifact('/app/artifacts/eda/cold_start_analysis.png')
    plt.close()
    
    logger.info(f" Создано 6 графиков в /app/artifacts/eda/")