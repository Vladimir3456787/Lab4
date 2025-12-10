import pandas as pd
import numpy as np
import mlflow
from datetime import datetime
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# Evidently для мониторинга
from evidently.report import Report
from evidently.metrics import (
    DataDriftTable,
    DatasetSummaryMetric,
    ColumnSummaryMetric,
    ColumnMissingValuesMetric,
    ColumnQuantileMetric,
    ColumnDistributionMetric,
    ColumnValueRangeMetric
)
from evidently.test_suite import TestSuite
from evidently.tests import (
    TestNumberOfRows,
    TestNumberOfColumns,
    TestColumnValueMin,
    TestColumnValueMax,
    TestColumnValueMean,
    TestColumnValueStd,
    TestShareOfMissingValues,
    TestNumberOfMissingValues,
    TestColumnShareOfMissingValues,
    TestColumnNumberOfMissingValues
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataMonitor:
    """Класс для мониторинга данных и обнаружения дрейфа"""
    
    def __init__(self, reference_data_path: Path, mlflow_tracking_uri: str = "http://mlflow:5000"):
        self.reference_data_path = reference_data_path
        self.reference_data = None
        mlflow.set_tracking_uri(mlflow_tracking_uri)
        
        # Пороговые значения для алертов
        self.drift_thresholds = {
            'psi': 0.2,      # Population Stability Index
            'ks_statistic': 0.1,  # Kolmogorov-Smirnov test
            'wasserstein': 0.15,  # Wasserstein distance
            'missing_values': 0.1, # Максимальный процент пропусков
        }
        
        # Загрузка эталонных данных
        self._load_reference_data()
    
    def _load_reference_data(self):
        """Загрузка эталонных данных"""
        try:
            if self.reference_data_path.exists():
                self.reference_data = pd.read_csv(self.reference_data_path)
                logger.info(f"Загружены эталонные данные: {len(self.reference_data)} записей")
            else:
                logger.warning("Эталонные данные не найдены, будут созданы при первом запуске")
        except Exception as e:
            logger.error(f"Ошибка при загрузке эталонных данных: {e}")
    
    def save_reference_data(self, data: pd.DataFrame):
        """Сохранение новых эталонных данных"""
        try:
            data.to_csv(self.reference_data_path, index=False)
            self.reference_data = data
            logger.info(f"Эталонные данные сохранены: {len(data)} записей")
            return True
        except Exception as e:
            logger.error(f"Ошибка при сохранении эталонных данных: {e}")
            return False
    
    def monitor_data_drift(self, current_data: pd.DataFrame, dataset_name: str = "production") -> Dict[str, Any]:
        """
        Мониторинг дрейфа данных
        
        Args:
            current_data: Текущие данные для анализа
            dataset_name: Имя датасета для логирования
        
        Returns:
            Словарь с результатами мониторинга
        """
        if self.reference_data is None:
            logger.warning("Нет эталонных данных, создаем из текущих")
            self.save_reference_data(current_data)
            return {"status": "reference_created", "message": "Эталонные данные созданы"}
        
        # Проверка совместимости данных
        if not set(self.reference_data.columns).issubset(set(current_data.columns)):
            missing_cols = set(self.reference_data.columns) - set(current_data.columns)
            logger.error(f"Отсутствуют колонки: {missing_cols}")
            return {"status": "error", "message": f"Отсутствуют колонки: {missing_cols}"}
        
        # Выбор общих колонок
        common_cols = list(set(self.reference_data.columns) & set(current_data.columns))
        reference_data = self.reference_data[common_cols]
        current_data = current_data[common_cols]
        
        logger.info(f"Анализ дрейфа данных: {len(common_cols)} признаков")
        
        # 1. Полный отчет Evidently
        drift_report = Report(metrics=[
            DataDriftTable(),
            DatasetSummaryMetric(),
            ColumnMissingValuesMetric(),
            ColumnDistributionMetric(),
            ColumnValueRangeMetric()
        ])
        
        drift_report.run(
            reference_data=reference_data,
            current_data=current_data
        )
        
        # 2. Test Suite для автоматических проверок
        test_list = [
            TestNumberOfRows(),
            TestNumberOfColumns(),
            TestShareOfMissingValues(),
            TestNumberOfMissingValues(),
        ]
        
        # Проверяем дрейф для числовых колонок
        numeric_cols = reference_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols[:3]:  # Проверяем первые 3 числовых признака
            test_list.extend([
                TestColumnValueMin(column_name=col),
                TestColumnValueMax(column_name=col),
                TestColumnValueMean(column_name=col),
                TestColumnValueStd(column_name=col),
                TestColumnShareOfMissingValues(column_name=col),
                TestColumnNumberOfMissingValues(column_name=col)
            ])
        
        drift_test_suite = TestSuite(tests=test_list)
        
        drift_test_suite.run(
            reference_data=reference_data,
            current_data=current_data
        )
        
        # Сохранение отчетов
        report_dict = drift_report.as_dict()
        test_results = drift_test_suite.as_dict()
        
        # Извлечение ключевых метрик
        drift_score = report_dict["metrics"][0]["result"].get("drift_score", 0)
        drifted_columns = report_dict["metrics"][0]["result"].get("number_of_drifted_columns", 0)
        dataset_drift = report_dict["metrics"][0]["result"].get("dataset_drift", False)
        
        # Проверка на пропуски
        missing_values_report = {}
        for col in common_cols:
            ref_missing = reference_data[col].isna().mean()
            curr_missing = current_data[col].isna().mean()
            missing_values_report[col] = {
                "reference_missing_rate": ref_missing,
                "current_missing_rate": curr_missing,
                "missing_rate_change": curr_missing - ref_missing
            }
        
        # Проверка диапазонов значений
        value_ranges_report = {}
        numeric_cols = reference_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in reference_data.columns:
                ref_min = reference_data[col].min()
                ref_max = reference_data[col].max()
                ref_mean = reference_data[col].mean()
                ref_std = reference_data[col].std()
                
                curr_min = current_data[col].min()
                curr_max = current_data[col].max()
                curr_mean = current_data[col].mean()
                curr_std = current_data[col].std()
                
                value_ranges_report[col] = {
                    "reference": {"min": ref_min, "max": ref_max, "mean": ref_mean, "std": ref_std},
                    "current": {"min": curr_min, "max": curr_max, "mean": curr_mean, "std": curr_std},
                    "range_violation": curr_min < ref_min or curr_max > ref_max,
                    "mean_change_pct": abs((curr_mean - ref_mean) / ref_mean) if ref_mean != 0 else 0
                }
        
        # Логирование в MLflow
        with mlflow.start_run(run_name=f"data_drift_monitoring_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"):
            # Логирование метрик
            mlflow.log_metric("drift_score", drift_score)
            mlflow.log_metric("drifted_columns", drifted_columns)
            mlflow.log_metric("dataset_drift", int(dataset_drift))
            mlflow.log_metric("total_columns", len(common_cols))
            
            # Логирование метрик по пропускам
            max_missing_rate = max([v["current_missing_rate"] for v in missing_values_report.values()])
            mlflow.log_metric("max_missing_rate", max_missing_rate)
            
            # Логирование метрик по диапазонам
            range_violations = sum([1 for v in value_ranges_report.values() if v["range_violation"]])
            mlflow.log_metric("range_violations", range_violations)
            
            # Логирование артефактов
            report_path = Path(f"/app/artifacts/reports/data_drift_report_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(report_path, 'w') as f:
                json.dump(report_dict, f, indent=2, default=str)
            mlflow.log_artifact(str(report_path))
            
            # Сохранение HTML дашборда
            html_path = Path(f"/app/artifacts/drift_dashboards/drift_report_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
            drift_report.save_html(str(html_path))
            mlflow.log_artifact(str(html_path))
            
            # Логирование тестов
            tests_path = Path(f"/app/artifacts/reports/drift_tests_{dataset_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(tests_path, 'w') as f:
                json.dump(test_results, f, indent=2, default=str)
            mlflow.log_artifact(str(tests_path))
        
        # Формирование результата
        result = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "dataset_name": dataset_name,
            "metrics": {
                "drift_score": drift_score,
                "drifted_columns": drifted_columns,
                "dataset_drift": dataset_drift,
                "total_columns_analyzed": len(common_cols),
                "max_missing_rate": max_missing_rate,
                "range_violations": range_violations,
                "test_suite_passed": test_results["summary"]["success_tests"],
                "test_suite_failed": test_results["summary"]["failed_tests"],
                "test_suite_total": test_results["summary"]["total_tests"]
            },
            "alerts": self._generate_data_alerts(
                drift_score, drifted_columns, max_missing_rate, range_violations
            ),
            "report_path": str(report_path),
            "dashboard_path": str(html_path)
        }
        
        logger.info(f"Мониторинг данных завершен. Дрейф: {drift_score:.3f}, Алёртов: {len(result['alerts'])}")
        
        return result
    
    def _generate_data_alerts(self, drift_score: float, drifted_columns: int, 
                            max_missing_rate: float, range_violations: int) -> list:
        """Генерация алертов на основе результатов мониторинга"""
        alerts = []
        
        # Проверка дрейфа
        if drift_score > self.drift_thresholds['psi']:
            alerts.append({
                "level": "CRITICAL",
                "type": "DATA_DRIFT",
                "message": f"Обнаружен значительный дрейф данных! PSI: {drift_score:.3f} > {self.drift_thresholds['psi']}",
                "metric": "drift_score",
                "value": drift_score,
                "threshold": self.drift_thresholds['psi']
            })
        
        if drifted_columns > 0:
            alerts.append({
                "level": "WARNING",
                "type": "COLUMN_DRIFT",
                "message": f"Дрейф обнаружен в {drifted_columns} колонках",
                "metric": "drifted_columns",
                "value": drifted_columns,
                "threshold": 0
            })
        
        # Проверка пропусков
        if max_missing_rate > self.drift_thresholds['missing_values']:
            alerts.append({
                "level": "WARNING",
                "type": "MISSING_VALUES",
                "message": f"Высокий процент пропусков: {max_missing_rate:.3f} > {self.drift_thresholds['missing_values']}",
                "metric": "max_missing_rate",
                "value": max_missing_rate,
                "threshold": self.drift_thresholds['missing_values']
            })
        
        # Проверка диапазонов
        if range_violations > 0:
            alerts.append({
                "level": "WARNING",
                "type": "VALUE_RANGE",
                "message": f"Нарушение диапазонов значений в {range_violations} колонках",
                "metric": "range_violations",
                "value": range_violations,
                "threshold": 0
            })
        
        return alerts
    
    def calculate_detailed_drift_metrics(self, reference_data: pd.DataFrame, 
                                       current_data: pd.DataFrame) -> Dict[str, Any]:
        """Расчет детальных метрик дрейфа"""
        from scipy import stats
        import numpy as np
        
        results = {}
        numeric_cols = reference_data.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if col in reference_data.columns and col in current_data.columns:
                ref_data = reference_data[col].dropna()
                curr_data = current_data[col].dropna()
                
                if len(ref_data) > 0 and len(curr_data) > 0:
                    # PSI (Population Stability Index)
                    try:
                        psi = self._calculate_psi(ref_data, curr_data, bins=10)
                    except:
                        psi = None
                    
                    # KS-test
                    try:
                        ks_statistic, ks_pvalue = stats.ks_2samp(ref_data, curr_data)
                    except:
                        ks_statistic, ks_pvalue = None, None
                    
                    # Wasserstein distance
                    try:
                        from scipy.stats import wasserstein_distance
                        wasserstein = wasserstein_distance(ref_data, curr_data)
                    except:
                        wasserstein = None
                    
                    results[col] = {
                        "psi": psi,
                        "ks_statistic": ks_statistic,
                        "ks_pvalue": ks_pvalue,
                        "wasserstein_distance": wasserstein,
                        "reference_stats": {
                            "mean": ref_data.mean(),
                            "std": ref_data.std(),
                            "min": ref_data.min(),
                            "max": ref_data.max(),
                            "count": len(ref_data)
                        },
                        "current_stats": {
                            "mean": curr_data.mean(),
                            "std": curr_data.std(),
                            "min": curr_data.min(),
                            "max": curr_data.max(),
                            "count": len(curr_data)
                        }
                    }
        
        return results
    
    def _calculate_psi(self, reference: pd.Series, current: pd.Series, bins: int = 10) -> float:
        """Расчет Population Stability Index"""
        # Создание бинов на основе эталонных данных
        min_val = min(reference.min(), current.min())
        max_val = max(reference.max(), current.max())
        
        if min_val == max_val:
            return 0.0
        
        bin_edges = np.linspace(min_val, max_val, bins + 1)
        
        # Расчет процентов в каждом бине
        ref_percents = np.histogram(reference, bins=bin_edges)[0] / len(reference)
        curr_percents = np.histogram(current, bins=bin_edges)[0] / len(current)
        
        # PSI расчет (с обработкой нулей)
        psi = 0
        for i in range(len(ref_percents)):
            if ref_percents[i] == 0:
                ref_percents[i] = 0.0001
            if curr_percents[i] == 0:
                curr_percents[i] = 0.0001
            
            psi += (curr_percents[i] - ref_percents[i]) * np.log(curr_percents[i] / ref_percents[i])
        
        return psi

def main():
    """Пример использования DataMonitor"""
    monitor = DataMonitor(Path("/app/artifacts/reference_data/reference.csv"))
    
    # Загрузка текущих данных (в реальной системе - из Feature Store или API)
    current_data = pd.DataFrame({
        'feature1': np.random.normal(0, 1, 1000),
        'feature2': np.random.exponential(1, 1000),
        'feature3': np.random.randint(0, 10, 1000)
    })
    
    # Мониторинг дрейфа
    result = monitor.monitor_data_drift(current_data, "production")
    
    print("\n" + "="*80)
    print(" РЕЗУЛЬТАТЫ МОНИТОРИНГА ДАННЫХ")
    print("="*80)
    print(f"Статус: {result['status']}")
    print(f"Метрики:")
    for key, value in result['metrics'].items():
        print(f"  {key}: {value}")
    
    if result['alerts']:
        print(f"\nАЛЕРТЫ:")
        for alert in result['alerts']:
            print(f"  [{alert['level']}] {alert['message']}")
    
    print(f"\nОтчет сохранен: {result.get('report_path', 'N/A')}")
    print(f"Дашборд: {result.get('dashboard_path', 'N/A')}")
    print("="*80)

if __name__ == "__main__":
    main()
