import time
import numpy as np
import threading
from typing import List, Dict, Any
from collections import deque
from datetime import datetime
import logging
from prometheus_client import Gauge, Counter

logger = logging.getLogger(__name__)

class InferenceMetrics:
    """Класс для мониторинга метрик инференса"""
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.latencies = deque(maxlen=window_size)
        self.errors = deque(maxlen=window_size)
        self.request_times = deque(maxlen=window_size)
        
        # Prometheus метрики
        self.latency_gauge = Gauge(
            'inference_latency_milliseconds',
            'Current inference latency in milliseconds'
        )
        self.throughput_gauge = Gauge(
            'inference_throughput_requests_per_second',
            'Current inference throughput'
        )
        self.error_rate_gauge = Gauge(
            'inference_error_rate_percent',
            'Current error rate percentage'
        )
        
        self.lock = threading.Lock()
    
    def record_latency(self, latency_ms: float):
        """Запись latency"""
        with self.lock:
            self.latencies.append(latency_ms)
            self.latency_gauge.set(latency_ms)
    
    def record_error(self, error_type: str):
        """Запись ошибки"""
        with self.lock:
            self.errors.append((datetime.now(), error_type))
    
    def record_request(self):
        """Запись времени запроса"""
        with self.lock:
            self.request_times.append(time.time())
    
    def get_metrics(self) -> Dict[str, Any]:
        """Получение текущих метрик"""
        with self.lock:
            if not self.latencies:
                return {"error": "No data available"}
            
            latencies_array = np.array(self.latencies)
            
            # Рассчет throughput
            throughput = 0
            if len(self.request_times) >= 2:
                time_window = self.request_times[-1] - self.request_times[0]
                if time_window > 0:
                    throughput = len(self.request_times) / time_window
            
            # Рассчет error rate
            error_rate = 0
            if self.errors:
                error_rate = len(self.errors) / self.window_size
            
            metrics = {
                "timestamp": datetime.now().isoformat(),
                "latency_ms": {
                    "p50": float(np.percentile(latencies_array, 50)),
                    "p95": float(np.percentile(latencies_array, 95)),
                    "p99": float(np.percentile(latencies_array, 99)),
                    "mean": float(np.mean(latencies_array)),
                    "std": float(np.std(latencies_array)),
                    "min": float(np.min(latencies_array)),
                    "max": float(np.max(latencies_array))
                },
                "throughput_rps": throughput,
                "error_rate": error_rate,
                "total_requests": len(self.latencies),
                "window_size": self.window_size
            }
            
            # Обновление Prometheus gauges
            self.throughput_gauge.set(throughput)
            self.error_rate_gauge.set(error_rate * 100)
            
            return metrics
    
    def reset(self):
        """Сброс метрик"""
        with self.lock:
            self.latencies.clear()
            self.errors.clear()
            self.request_times.clear()

def measure_inference_performance(model, test_data, n_iterations: int = 1000):
    """
    Измерение производительности модели
    
    Args:
        model: ML-модель
        test_data: Тестовые данные
        n_iterations: Количество итераций для теста
    
    Returns:
        Dict с метриками производительности
    """
    latencies = []
    errors = 0
    
    for i in range(n_iterations):
        start_time = time.time()
        
        try:
            # Инференс на случайном сэмпле
            sample_idx = np.random.randint(0, len(test_data))
            sample = test_data[sample_idx:sample_idx+1]
            
            prediction = model.predict(sample)
            
            latency = (time.time() - start_time) * 1000  # мс
            latencies.append(latency)
            
        except Exception as e:
            errors += 1
            logger.error(f"Inference error on iteration {i}: {e}")
    
    if not latencies:
        return {"error": "No successful inferences"}
    
    latencies_array = np.array(latencies)
    
    return {
        "p50_latency_ms": float(np.percentile(latencies_array, 50)),
        "p95_latency_ms": float(np.percentile(latencies_array, 95)),
        "p99_latency_ms": float(np.percentile(latencies_array, 99)),
        "mean_latency_ms": float(np.mean(latencies_array)),
        "throughput_rps": n_iterations / (sum(latencies_array) / 1000),
        "error_rate": errors / n_iterations,
        "total_iterations": n_iterations,
        "successful_iterations": n_iterations - errors
    }