import random
import time
import logging
from typing import Dict, Any, Optional
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class CanaryRelease:
    """Реализация Canary релиза"""
    
    def __init__(self, current_model, new_model):
        self.current_model = current_model
        self.new_model = new_model
        self.traffic_percent = 0  # Начинаем с 0%
        self.metrics = {
            "current_model": [],
            "new_model": [],
            "comparisons": []
        }
        
    def route_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """
        Маршрутизация запроса между моделями
        
        Args:
            request: Входные данные для инференса
        
        Returns:
            Результат инференса и метаданные
        """
        # Определение, куда направить запрос
        use_new_model = random.random() < (self.traffic_percent / 100)
        
        start_time = time.time()
        
        try:
            if use_new_model:
                model = self.new_model
                model_type = "new"
                prediction = model.predict([request["features"]])
            else:
                model = self.current_model
                model_type = "current"
                prediction = model.predict([request["features"]])
            
            latency = (time.time() - start_time) * 1000
            
            # Получение предсказания от другой модели для сравнения
            if use_new_model:
                current_start = time.time()
                current_pred = self.current_model.predict([request["features"]])
                current_latency = (time.time() - current_start) * 1000
            else:
                new_start = time.time()
                new_pred = self.new_model.predict([request["features"]])
                new_latency = (time.time() - new_start) * 1000
            
            # Запись метрик
            self._record_metrics(
                model_type=model_type,
                latency=latency,
                prediction=prediction[0],
                request_id=request.get("request_id", "unknown")
            )
            
            return {
                "prediction": float(prediction[0]),
                "model_used": model_type,
                "latency_ms": latency,
                "traffic_percent": self.traffic_percent,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error in canary routing: {e}")
            raise
    
    def _record_metrics(self, model_type: str, latency: float, 
                       prediction: float, request_id: str):
        """Запись метрик для анализа"""
        metric_record = {
            "timestamp": datetime.now().isoformat(),
            "model_type": model_type,
            "latency_ms": latency,
            "prediction": prediction,
            "request_id": request_id
        }
        
        self.metrics[f"{model_type}_model"].append(metric_record)
    
    def analyze_metrics(self, window_size: int = 100) -> Dict[str, Any]:
        """
        Анализ метрик для принятия решения
        
        Returns:
            Словарь с результатами анализа
        """
        if len(self.metrics["new_model"]) < 10:
            return {"status": "insufficient_data", "recommendation": "continue"}
        
        # Анализ latency
        new_latencies = [m["latency_ms"] for m in self.metrics["new_model"][-window_size:]]
        current_latencies = [m["latency_ms"] for m in self.metrics["current_model"][-window_size:]]
        
        new_avg_latency = np.mean(new_latencies) if new_latencies else 0
        current_avg_latency = np.mean(current_latencies) if current_latencies else 0
        
        # Анализ стабильности предсказаний
        # (в реальной системе здесь были бы бизнес-метрики)
        
        # Принятие решения
        latency_ratio = new_avg_latency / current_avg_latency if current_avg_latency > 0 else 1
        
        recommendation = "continue"
        if latency_ratio > 1.2:  # Новая модель на 20% медленнее
            recommendation = "rollback"
        elif latency_ratio < 0.9:  # Новая модель на 10% быстрее
            recommendation = "increase_traffic"
        
        return {
            "status": "analysis_complete",
            "recommendation": recommendation,
            "metrics": {
                "new_model_avg_latency_ms": new_avg_latency,
                "current_model_avg_latency_ms": current_avg_latency,
                "latency_ratio": latency_ratio,
                "new_model_requests": len(self.metrics["new_model"]),
                "current_model_requests": len(self.metrics["current_model"])
            }
        }
    
    def update_traffic_split(self, analysis_result: Dict[str, Any]):
        """Обновление процентного соотношения трафика"""
        recommendation = analysis_result.get("recommendation")
        
        if recommendation == "increase_traffic":
            self.traffic_percent = min(100, self.traffic_percent + 10)
            logger.info(f"✅ Increasing traffic to {self.traffic_percent}%")
        elif recommendation == "rollback":
            self.traffic_percent = max(0, self.traffic_percent - 20)
            logger.warning(f"⚠️ Rolling back traffic to {self.traffic_percent}%")
        elif recommendation == "continue":
            logger.info(f"Maintaining traffic at {self.traffic_percent}%")
        
        return self.traffic_percent
    
    def run_canary_validation(self, test_requests: list, 
                             max_traffic_percent: int = 100) -> bool:
        """
        Запуск полного цикла валидации Canary релиза
        
        Returns:
            True если релиз успешен, False если требуется откат
        """
        logger.info(f"Starting canary validation (target: {max_traffic_percent}%)")
        
        step = 5  # Шаг увеличения трафика
        self.traffic_percent = step
        
        while self.traffic_percent <= max_traffic_percent:
            logger.info(f"Current traffic: {self.traffic_percent}%")
            
            # Прогрев (сбор метрик)
            warmup_requests = test_requests[:100]
            for req in warmup_requests:
                try:
                    self.route_request(req)
                except Exception as e:
                    logger.error(f"Warmup request failed: {e}")
            
            # Анализ метрик
            analysis = self.analyze_metrics()
            
            if analysis["recommendation"] == "rollback":
                logger.error("❌ Canary validation failed, rolling back")
                self.traffic_percent = 0
                return False
            
            # Увеличение трафика
            self.update_traffic_split(analysis)
            
            # Пауза между шагами
            time.sleep(2)
        
        logger.info("✅ Canary validation successful")
        return True