import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import logging
import json
from typing import Dict, Any, List, Optional
import os
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AlertManager:
    """Менеджер алертов для ML мониторинга"""
    
    def __init__(self):
        # Конфигурация из переменных окружения
        self.slack_webhook_url = os.getenv('SLACK_WEBHOOK_URL', '')
        self.email_enabled = os.getenv('EMAIL_NOTIFICATIONS', 'false').lower() == 'true'
        
        # Email конфигурация
        self.smtp_server = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.getenv('SMTP_PORT', '587'))
        self.smtp_username = os.getenv('SMTP_USERNAME', '')
        self.smtp_password = os.getenv('SMTP_PASSWORD', '')
        self.email_from = os.getenv('EMAIL_FROM', '')
        self.email_to = os.getenv('EMAIL_TO', '').split(',')
        
        # История алертов
        self.alert_history = []
        self.max_history_size = 100
    
    def send_alert(self, alert_data: Dict[str, Any], alert_type: str = "monitoring"):
        """
        Отправка алерта через настроенные каналы
        """
        # Логирование алерта
        self._log_alert(alert_data)
        
        # Формирование сообщения
        message = self._format_alert_message(alert_data, alert_type)
        
        # Отправка в Slack
        if self.slack_webhook_url:
            self._send_slack_alert(message, alert_data.get('level', 'WARNING'))
        
        # Отправка email
        if self.email_enabled and self.email_from and self.email_to:
            self._send_email_alert(message, alert_data, alert_type)
        
        # Логирование в консоль
        logger.warning(f"ALERT: {message}")
    
    def _format_alert_message(self, alert_data: Dict[str, Any], alert_type: str) -> str:
        """Форматирование сообщения алерта"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        level = alert_data.get('level', 'UNKNOWN')
        alert_type_display = alert_data.get('type', alert_type)
        
        # Базовое сообщение
        message_parts = [
            "🚨 *ML MONITORING ALERT* 🚨",
            "",
            f"*Время:* {timestamp}",
            f"*Уровень:* {level}",
            f"*Тип:* {alert_type_display}",
            f"*Сообщение:* {alert_data.get('message', 'No message')}",
            ""
        ]
        
        # Добавление метрик
        if 'metric' in alert_data and 'value' in alert_data and 'threshold' in alert_data:
            message_parts.extend([
                f"*Метрика:* {alert_data['metric']}",
                f"*Текущее значение:* {alert_data['value']}",
                f"*Порог:* {alert_data['threshold']}",
                ""
            ])
        
        # Добавление контекста
        if 'context' in alert_data:
            message_parts.append("*Контекст:*")
            for key, value in alert_data['context'].items():
                message_parts.append(f"  {key}: {value}")
            message_parts.append("")
        
        # Добавление рекомендаций
        recommendations = self._get_recommendations(alert_type_display, alert_data)
        if recommendations:
            message_parts.extend(["*Рекомендации:*", recommendations])
        
        return "\n".join(message_parts)
    
    def _get_recommendations(self, alert_type: str, alert_data: Dict[str, Any]) -> str:
        """Получение рекомендаций на основе типа алерта"""
        recommendations_map = {
            "DATA_DRIFT": """
1. Проверьте источник данных на изменения в пайплайне
2. Проанализируйте дрейфующие признаки
3. Рассмотрите обновление эталонных данных
4. При необходимости запустите переобучение модели
            """,
            "ACCURACY_DROP": """
1. Проверьте качество входных данных
2. Проанализируйте ошибки классификации
3. Изучите распределение предсказаний
4. Рассмотрите возможность retrain модели
            """,
            "CONCEPT_DRIFT": """
1. Проведите анализ изменений в бизнес-процессах
2. Проверьте релевантность признаков
3. Обновите тренировочные данные
4. Запланируйте переобучение модели
            """,
            "MISSING_VALUES": """
1. Проверьте источники данных на сбои
2. Настройте обработку пропусков в пайплайне
3. Убедитесь в корректности ETL процессов
            """,
            "PREDICTION_DRIFT": """
1. Проверьте распределение входных данных
2. Проанализируйте изменения в пороге классификации
3. Изучите сдвиги в предсказаниях по сегментам
            """
        }
        
        return recommendations_map.get(alert_type, """
1. Проверьте логи мониторинга
2. Проанализируйте метрики в MLflow
3. Изучите дашборды дрейфа
        """)
    
    def _send_slack_alert(self, message: str, level: str = "WARNING"):
        """Отправка алерта в Slack"""
        try:
            # Форматирование для Slack
            color_map = {
                "CRITICAL": "#FF0000",  # Красный
                "WARNING": "#FFA500",   # Оранжевый
                "INFO": "#36A64F"       # Зеленый
            }
            
            color = color_map.get(level, "#808080")
            
            payload = {
                "attachments": [
                    {
                        "color": color,
                        "text": message,
                        "mrkdwn_in": ["text"],
                        "footer": "ML Monitoring System",
                        "ts": datetime.now().timestamp()
                    }
                ]
            }
            
            response = requests.post(
                self.slack_webhook_url,
                json=payload,
                timeout=10
            )
            
            if response.status_code == 200:
                logger.info(f"Alert sent to Slack: {level}")
            else:
                logger.error(f"Failed to send Slack alert: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Error sending Slack alert: {e}")
    
    def _send_email_alert(self, message: str, alert_data: Dict[str, Any], alert_type: str):
        """Отправка алерта по email"""
        try:
            # Создание email сообщения
            msg = MIMEMultipart('alternative')
            msg['Subject'] = f"[{alert_data.get('level', 'ALERT')}] ML Monitoring Alert: {alert_type}"
            msg['From'] = self.email_from
            msg['To'] = ', '.join(self.email_to)
            
            # Текстовая версия
            text_part = MIMEText(message, 'plain')
            msg.attach(text_part)
            
            # HTML версия
            html_message = self._format_html_email(message, alert_data)
            html_part = MIMEText(html_message, 'html')
            msg.attach(html_part)
            
            # Отправка
            with smtplib.SMTP(self.smtp_server, self.smtp_port) as server:
                server.starttls()
                server.login(self.smtp_username, self.smtp_password)
                server.send_message(msg)
            
            logger.info(f"Email alert sent to {len(self.email_to)} recipients")
            
        except Exception as e:
            logger.error(f"Error sending email alert: {e}")
    
    def _format_html_email(self, message: str, alert_data: Dict[str, Any]) -> str:
        """Форматирование HTML email"""
        level = alert_data.get('level', 'WARNING')
        level_color = {
            'CRITICAL': '#dc3545',
            'WARNING': '#ffc107',
            'INFO': '#28a745'
        }.get(level, '#6c757d')
        
        # Экранируем фигурные скобки для CSS
        alert_style = f"border-left: 4px solid {level_color}; padding: 15px; margin: 20px 0; background-color: #f8f9fa;"
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                .alert {{ {alert_style} }}
                .alert-title {{ color: {level_color}; font-size: 18px; font-weight: bold; margin-bottom: 10px; }}
                .metric {{ background-color: white; padding: 10px; margin: 5px 0; border-radius: 4px; }}
                .recommendations {{ background-color: #e8f4fd; padding: 15px; margin-top: 15px; border-radius: 4px; }}
            </style>
        </head>
        <body>
            <div class="alert">
                <div class="alert-title">🚨 ML Monitoring Alert</div>
                <div>{message.replace(chr(10), '<br>')}</div>
            </div>
            <div class="recommendations">
                <strong>Рекомендации:</strong><br>
                {self._get_recommendations(alert_data.get('type', ''), alert_data).replace(chr(10), '<br>')}
            </div>
        </body>
        </html>
        """
        
        return html
    
    def _log_alert(self, alert_data: Dict[str, Any]):
        """Логирование алерта в историю"""
        alert_data['timestamp'] = datetime.now().isoformat()
        self.alert_history.append(alert_data)
        
        # Ограничение размера истории
        if len(self.alert_history) > self.max_history_size:
            self.alert_history = self.alert_history[-self.max_history_size:]
        
        # Сохранение в файл
        try:
            log_path = "/app/artifacts/reports/alert_history.json"
            with open(log_path, 'w') as f:
                json.dump(self.alert_history, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error saving alert history: {e}")
    
    def get_alert_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Получение сводки по алертам за указанный период"""
        cutoff_time = datetime.now().timestamp() - (hours * 3600)
        
        recent_alerts = [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert['timestamp']).timestamp() > cutoff_time
        ]
        
        summary = {
            "total_alerts": len(recent_alerts),
            "critical_alerts": sum(1 for a in recent_alerts if a.get('level') == 'CRITICAL'),
            "warning_alerts": sum(1 for a in recent_alerts if a.get('level') == 'WARNING'),
            "info_alerts": sum(1 for a in recent_alerts if a.get('level') == 'INFO'),
            "by_type": {},
            "latest_alerts": recent_alerts[-10:] if recent_alerts else []
        }
        
        # Группировка по типу
        for alert in recent_alerts:
            alert_type = alert.get('type', 'UNKNOWN')
            summary["by_type"][alert_type] = summary["by_type"].get(alert_type, 0) + 1
        
        return summary

def test_alerting():
    """Тестирование системы алертинга"""
    alert_manager = AlertManager()
    
    # Тестовые алерты
    test_alerts = [
        {
            "level": "CRITICAL",
            "type": "DATA_DRIFT",
            "message": "Обнаружен значительный дрейф данных!",
            "metric": "psi",
            "value": 0.35,
            "threshold": 0.2,
            "context": {
                "dataset": "production",
                "drifted_columns": ["price", "user_activity"],
                "timestamp": datetime.now().isoformat()
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
    
    print("Testing alert system...")
    for alert in test_alerts:
        alert_manager.send_alert(alert)
    
    # Получение сводки
    summary = alert_manager.get_alert_summary(1)
    print(f"\nAlert summary (last hour):")
    print(f"  Total alerts: {summary['total_alerts']}")
    print(f"  Critical: {summary['critical_alerts']}")
    print(f"  Warning: {summary['warning_alerts']}")
    print(f"  By type: {summary['by_type']}")

if __name__ == "__main__":
    test_alerting()
