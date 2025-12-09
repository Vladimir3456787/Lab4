import asyncio
import aiohttp
import time
import json
import numpy as np
import pandas as pd
from typing import List, Dict, Any
import logging
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LoadTester:
    """Класс для нагрузочного тестирования API"""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results = []
        
    async def make_request(self, session: aiohttp.ClientSession, 
                          request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Отправка одного запроса"""
        start_time = time.time()
        
        try:
            async with session.post(
                f"{self.base_url}/predict",
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                response_data = await response.json()
                latency = (time.time() - start_time) * 1000  # мс
                
                return {
                    "success": True,
                    "latency_ms": latency,
                    "status_code": response.status,
                    "response": response_data,
                    "error": None
                }
                
        except Exception as e:
            latency = (time.time() - start_time) * 1000
            return {
                "success": False,
                "latency_ms": latency,
                "status_code": None,
                "response": None,
                "error": str(e)
            }
    
    async def run_test(self, concurrent_users: int = 10, 
                      requests_per_user: int = 100,
                      think_time: float = 0.1) -> Dict[str, Any]:
        """
        Запуск нагрузочного теста
        
        Args:
            concurrent_users: Количество одновременных пользователей
            requests_per_user: Запросов на пользователя
            think_time: Время между запросами (секунды)
        """
        logger.info(f"Starting load test: {concurrent_users} users, "
                   f"{requests_per_user} requests each")
        
        self.results = []
        
        # Генерация тестовых данных
        test_requests = [
            {
                "user_id": np.random.randint(1, 1000),
                "transaction_amount": np.random.uniform(10, 1000)
            }
            for _ in range(concurrent_users * requests_per_user)
        ]
        
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            # Создание задач для конкурентных запросов
            tasks = []
            
            for i in range(concurrent_users):
                user_requests = test_requests[
                    i * requests_per_user:(i + 1) * requests_per_user
                ]
                
                task = self._simulate_user(session, user_requests, think_time)
                tasks.append(task)
            
            # Ожидание завершения всех задач
            await asyncio.gather(*tasks)
        
        total_time = time.time() - start_time
        total_requests = len(self.results)
        
        # Анализ результатов
        analysis = self._analyze_results(total_time, total_requests)
        
        # Сохранение результатов
        self._save_results(analysis)
        
        return analysis
    
    async def _simulate_user(self, session: aiohttp.ClientSession,
                           requests: List[Dict[str, Any]],
                           think_time: float):
        """Симуляция поведения пользователя"""
        for request in requests:
            result = await self.make_request(session, request)
            self.results.append(result)
            
            if think_time > 0:
                await asyncio.sleep(think_time)
    
    def _analyze_results(self, total_time: float, total_requests: int) -> Dict[str, Any]:
        """Анализ результатов теста"""
        if not self.results:
            return {"error": "No results to analyze"}
        
        # Разделение успешных и неуспешных запросов
        successful = [r for r in self.results if r["success"]]
        failed = [r for r in self.results if not r["success"]]
        
        # Latency анализ
        latencies = [r["latency_ms"] for r in successful]
        
        analysis = {
            "test_timestamp": datetime.now().isoformat(),
            "total_requests": total_requests,
            "total_time_seconds": total_time,
            "successful_requests": len(successful),
            "failed_requests": len(failed),
            "success_rate": len(successful) / total_requests if total_requests > 0 else 0,
            "throughput_rps": total_requests / total_time if total_time > 0 else 0,
            "latency_ms": {
                "mean": np.mean(latencies) if latencies else 0,
                "p50": np.percentile(latencies, 50) if latencies else 0,
                "p95": np.percentile(latencies, 95) if latencies else 0,
                "p99": np.percentile(latencies, 99) if latencies else 0,
                "min": np.min(latencies) if latencies else 0,
                "max": np.max(latencies) if latencies else 0,
                "std": np.std(latencies) if latencies else 0
            },
            "errors_by_type": self._categorize_errors(failed),
            "concurrent_users": len(set([r.get("user_id", 0) for r in self.results]))
        }
        
        return analysis
    
    def _categorize_errors(self, failed_requests: List[Dict[str, Any]]) -> Dict[str, int]:
        """Категоризация ошибок"""
        error_categories = {
            "timeout": 0,
            "connection_error": 0,
            "server_error": 0,
            "client_error": 0,
            "other": 0
        }
        
        for req in failed_requests:
            error_msg = req.get("error", "").lower()
            
            if "timeout" in error_msg:
                error_categories["timeout"] += 1
            elif "connection" in error_msg:
                error_categories["connection_error"] += 1
            elif "5" in str(req.get("status_code", "")):
                error_categories["server_error"] += 1
            elif "4" in str(req.get("status_code", "")):
                error_categories["client_error"] += 1
            else:
                error_categories["other"] += 1
        
        return error_categories
    
    def _save_results(self, analysis: Dict[str, Any]):
        """Сохранение результатов в файл"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"/app/artifacts/load_test_results/load_test_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        logger.info(f"Results saved to {filename}")
        
        # Создание визуализаций
        self._create_visualizations(analysis, timestamp)
    
    def _create_visualizations(self, analysis: Dict[str, Any], timestamp: str):
        """Создание графиков и визуализаций"""
        try:
            # Гистограмма latency
            latencies = [r["latency_ms"] for r in self.results if r["success"]]
            
            if latencies:
                plt.figure(figsize=(12, 8))
                
                # График 1: Распределение latency
                plt.subplot(2, 2, 1)
                plt.hist(latencies, bins=50, alpha=0.7, color='blue', edgecolor='black')
                plt.title('Latency Distribution')
                plt.xlabel('Latency (ms)')
                plt.ylabel('Frequency')
                plt.grid(True, alpha=0.3)
                
                # График 2: Временная шкала latency
                plt.subplot(2, 2, 2)
                plt.plot(range(len(latencies)), latencies, 'b.', alpha=0.5)
                plt.title('Latency Over Time')
                plt.xlabel('Request Number')
                plt.ylabel('Latency (ms)')
                plt.grid(True, alpha=0.3)
                
                # График 3: Success/Error rates
                plt.subplot(2, 2, 3)
                success_rate = analysis["success_rate"] * 100
                labels = ['Success', 'Errors']
                sizes = [success_rate, 100 - success_rate]
                colors = ['green', 'red']
                plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%')
                plt.title('Success Rate')
                
                # График 4: Throughput over time
                plt.subplot(2, 2, 4)
                window_size = 100
                throughputs = []
                for i in range(0, len(self.results), window_size):
                    window = self.results[i:i+window_size]
                    window_time = sum(r["latency_ms"] for r in window) / 1000
                    if window_time > 0:
                        throughputs.append(len(window) / window_time)
                    else:
                        throughputs.append(0)
                
                plt.plot(range(len(throughputs)), throughputs, 'g-', linewidth=2)
                plt.title('Throughput Over Time (moving window)')
                plt.xlabel('Window Number')
                plt.ylabel('Requests per Second')
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(f"/app/artifacts/load_test_results/load_test_plot_{timestamp}.png", 
                          dpi=150, bbox_inches='tight')
                plt.close()
                
                logger.info(f"Visualizations saved")
                
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
    
    async def run_edge_case_tests(self):
        """Тестирование граничных случаев"""
        edge_cases = [
            {"user_id": 999999, "transaction_amount": 0.01},  # Несуществующий пользователь
            {"user_id": 1, "transaction_amount": -100},  # Отрицательная сумма
            {"user_id": "invalid", "transaction_amount": 100},  # Неправильный тип
            {"user_id": 1, "transaction_amount": 1000000},  # Очень большая сумма
            {"user_id": 1},  # Отсутствует обязательное поле
        ]
        
        results = []
        
        async with aiohttp.ClientSession() as session:
            for case in edge_cases:
                result = await self.make_request(session, case)
                results.append({
                    "test_case": case,
                    "result": result
                })
        
        return results

async def main():
    """Основная функция запуска тестов"""
    tester = LoadTester("http://localhost:8000")
    
    # 1. Тестирование граничных случаев
    logger.info("Running edge case tests...")
    edge_results = await tester.run_edge_case_tests()
    
    for result in edge_results:
        logger.info(f"Test case: {result['test_case']}")
        logger.info(f"Result: {result['result']}")
    
    # 2. Нагрузочное тестирование
    logger.info("\nRunning load tests...")
    
    test_scenarios = [
        {"concurrent_users": 10, "requests_per_user": 50},
        {"concurrent_users": 50, "requests_per_user": 100},
        {"concurrent_users": 100, "requests_per_user": 200},
    ]
    
    all_results = []
    
    for scenario in test_scenarios:
        logger.info(f"\nTesting scenario: {scenario}")
        result = await tester.run_test(**scenario)
        all_results.append(result)
        
        # Краткий вывод
        logger.info(f"Success rate: {result['success_rate']:.2%}")
        logger.info(f"Throughput: {result['throughput_rps']:.2f} RPS")
        logger.info(f"P95 latency: {result['latency_ms']['p95']:.2f} ms")
    
    # 3. Сохранение сводного отчета
    summary = {
        "test_timestamp": datetime.now().isoformat(),
        "scenarios_tested": len(test_scenarios),
        "results": all_results,
        "edge_case_results": edge_results
    }
    
    summary_file = "/app/artifacts/load_test_results/test_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    logger.info(f"\nTest summary saved to {summary_file}")
    
    return all_results

if __name__ == "__main__":
    asyncio.run(main())