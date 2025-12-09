import threading
import queue
import time
import logging
from typing import Dict, Any, Optional
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)

class ShadowDeployment:
    """Реализация Shadow Mode деплоя"""
    
    def __init__(self, production_model, shadow_model, 
                 comparison_callback=None):
        self.production_model = production_model
        self.shadow_model = shadow_model
        self.comparison_callback = comparison_callback
        self.request_queue = queue.Queue()
        self.results = []
        self.is_running = False
        self.worker_thread = None
        
    def start(self):
        """Запуск shadow mode"""
        self.is_running = True
        self.worker_thread = threading.Thread(
            target=self._process_shadow_requests,
            daemon=True
        )
        self.worker_thread.start()
        logger.info("Shadow deployment started")
    
    def stop(self):
        """Остановка shadow mode"""
        self.is_running = False
        if self.worker_thread:
            self.worker_thread.join(timeout=5)
        logger.info("Shadow deployment stopped")
    
    def process_request(self, request: Dict[str, Any], 
                       production_result: Dict[str, Any]):
        """
        Обработка запроса в shadow mode
        
        Args:
            request: Оригинальный запрос
            production_result: Результат от production модели
        """
        # Добавляем в очередь для асинхронной обработки
        self.request_queue.put({
            "request": request,
            "production_result": production_result,
            "timestamp": datetime.now().isoformat()
        })
    
    def _process_shadow_requests(self):
        """Асинхронная обработка запросов shadow модели"""
        while self.is_running or not self.request_queue.empty():
            try:
                # Получение запроса из очереди
                item = self.request_queue.get(timeout=1)
                
                start_time = time.time()
                
                try:
                    # Инференс shadow модели
                    shadow_prediction = self.shadow_model.predict(
                        [item["request"]["features"]]
                    )
                    
                    shadow_latency = (time.time() - start_time) * 1000
                    
                    # Сравнение с production моделью
                    comparison = self._compare_predictions(
                        production_pred=item["production_result"]["prediction"],
                        shadow_pred=float(shadow_prediction[0]),
                        request=item["request"]
                    )
                    
                    # Запись результата
                    result = {
                        "timestamp": item["timestamp"],
                        "request_id": item["request"].get("request_id", "unknown"),
                        "production_prediction": item["production_result"]["prediction"],
                        "shadow_prediction": float(shadow_prediction[0]),
                        "production_latency_ms": item["production_result"].get("latency_ms", 0),
                        "shadow_latency_ms": shadow_latency,
                        "comparison": comparison,
                        "processing_time_ms": (time.time() - start_time) * 1000
                    }
                    
                    self.results.append(result)
                    
                    # Вызов callback если задан
                    if self.comparison_callback:
                        self.comparison_callback(result)
                    
                except Exception as e:
                    logger.error(f"Shadow inference failed: {e}")
                
                self.request_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in shadow processing: {e}")
    
    def _compare_predictions(self, production_pred: float, 
                           shadow_pred: float, request: Dict[str, Any]) -> Dict[str, Any]:
        """Сравнение предсказаний двух моделей"""
        
        # Вычисление разницы
        absolute_diff = abs(production_pred - shadow_pred)
        relative_diff = absolute_diff / max(production_pred, 0.001)
        
        # Определение типа расхождения
        if absolute_diff < 0.01:
            discrepancy_type = "minor"
        elif absolute_diff < 0.1:
            discrepancy_type = "moderate"
        else:
            discrepancy_type = "major"
        
        # Проверка на расхождения в бинарных предсказаниях
        prod_binary = 1 if production_pred > 0.5 else 0
        shadow_binary = 1 if shadow_pred > 0.5 else 0
        
        binary_mismatch = prod_binary != shadow_binary
        
        return {
            "absolute_difference": absolute_diff,
            "relative_difference": relative_diff,
            "discrepancy_type": discrepancy_type,
            "binary_mismatch": binary_mismatch,
            "production_binary": prod_binary,
            "shadow_binary": shadow_binary
        }
    
    def get_summary(self, window_hours: int = 24) -> Dict[str, Any]:
        """
        Получение сводки по shadow deployment
        
        Args:
            window_hours: Временное окно для анализа
        """
        if not self.results:
            return {"error": "No results available"}
        
        # Фильтрация по времени
        cutoff_time = time.time() - (window_hours * 3600)
        recent_results = [
            r for r in self.results 
            if time.mktime(datetime.fromisoformat(r["timestamp"]).timetuple()) > cutoff_time
        ]
        
        if not recent_results:
            return {"error": "No recent results"}
        
        # Статистика
        total_requests = len(recent_results)
        
        # Расхождения
        major_discrepancies = sum(
            1 for r in recent_results 
            if r["comparison"]["discrepancy_type"] == "major"
        )
        
        binary_mismatches = sum(
            1 for r in recent_results 
            if r["comparison"]["binary_mismatch"]
        )
        
        # Latency
        prod_latencies = [r["production_latency_ms"] for r in recent_results]
        shadow_latencies = [r["shadow_latency_ms"] for r in recent_results]
        
        return {
            "summary_timestamp": datetime.now().isoformat(),
            "analysis_window_hours": window_hours,
            "total_requests_processed": total_requests,
            "discrepancy_analysis": {
                "major_discrepancies": major_discrepancies,
                "major_discrepancy_rate": major_discrepancies / total_requests,
                "binary_mismatches": binary_mismatches,
                "binary_mismatch_rate": binary_mismatches / total_requests,
                "avg_absolute_difference": np.mean([
                    r["comparison"]["absolute_difference"] 
                    for r in recent_results
                ])
            },
            "performance_comparison": {
                "production_avg_latency_ms": np.mean(prod_latencies) if prod_latencies else 0,
                "shadow_avg_latency_ms": np.mean(shadow_latencies) if shadow_latencies else 0,
                "production_p95_latency_ms": np.percentile(prod_latencies, 95) if prod_latencies else 0,
                "shadow_p95_latency_ms": np.percentile(shadow_latencies, 95) if shadow_latencies else 0
            },
            "recommendation": self._generate_recommendation(
                major_discrepancies, 
                binary_mismatches, 
                total_requests
            )
        }
    
    def _generate_recommendation(self, major_discrepancies: int, 
                               binary_mismatches: int, total_requests: int) -> str:
        """Генерация рекомендации на основе анализа"""
        
        major_rate = major_discrepancies / total_requests if total_requests > 0 else 0
        mismatch_rate = binary_mismatches / total_requests if total_requests > 0 else 0
        
        if major_rate < 0.01 and mismatch_rate < 0.05:
            return "ready_for_production"
        elif major_rate < 0.05 and mismatch_rate < 0.1:
            return "requires_further_testing"
        else:
            return "not_ready_high_discrepancy"
    
    def clear_results(self):
        """Очистка результатов"""
        self.results.clear()
        logger.info("Shadow results cleared")