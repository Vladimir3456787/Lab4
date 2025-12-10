#!/usr/bin/env python3
"""
Скрипт для тестирования системы алертинга
"""

import sys
sys.path.append('/app/src')

from monitoring.alerting import AlertManager

def test_alerting():
    """Тестирование всех типов алертов"""
    alert_manager = AlertManager()
    
    print("🚀 Тестирование системы алертинга...")
    
    # Тестовые алерты разных уровней
    test_alerts = [
        {
            "level": "CRITICAL",
            "type": "DATA_DRIFT",
            "message": "Обнаружен критический дрейф данных!",
            "metric": "psi",
            "value": 0.45,
            "threshold": 0.2,
            "context": {
                "drifted_columns": ["price", "user_activity"],
                "dataset": "production"
            }
        },
        {
            "level": "WARNING",
            "type": "MODEL_DEGRADATION",
            "message": "Падение точности модели на 12%",
            "metric": "accuracy",
            "value": 0.78,
            "threshold": 0.85,
            "context": {
                "model": "EcommerceRecommendationModel",
                "previous_accuracy": 0.90
            }
        },
        {
            "level": "INFO",
            "type": "RETRAIN_COMPLETED",
            "message": "Модель успешно переобучена",
            "metric": "retrain_duration",
            "value": 125,
            "threshold": 300,
            "context": {
                "new_accuracy": 0.92,
                "improvement": "+0.04"
            }
        }
    ]
    
    # Отправка тестовых алертов
    for alert in test_alerts:
        print(f"\n📨 Отправка алерта: {alert['type']} [{alert['level']}]")
        print(f"   Сообщение: {alert['message']}")
        alert_manager.send_alert(alert)
    
    # Получение сводки
    print("\n📊 Сводка по алертам:")
    summary = alert_manager.get_alert_summary(1)
    print(f"   Всего алертов: {summary['total_alerts']}")
    print(f"   Критических: {summary['critical_alerts']}")
    print(f"   Предупреждений: {summary['warning_alerts']}")
    print(f"   Информационных: {summary['info_alerts']}")
    
    print("\n✅ Тестирование завершено!")
    print("\n💡 Для настройки реальных уведомлений установите переменные окружения:")
    print("   - SLACK_WEBHOOK_URL для Slack")
    print("   - EMAIL_NOTIFICATIONS=true для email")

if __name__ == "__main__":
    test_alerting()
