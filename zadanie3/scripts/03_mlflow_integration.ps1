Write-Host " Запуск задания 3: Интеграция с MLflow" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green

# Создание директорий
New-Item -ItemType Directory -Force -Path artifacts/models
New-Item -ItemType Directory -Force -Path artifacts/reports

Write-Host "1. Запуск MLflow Tracking Server..." -ForegroundColor Yellow
docker-compose up -d mlflow

Write-Host "2. Ожидание запуска MLflow..." -ForegroundColor Yellow
Start-Sleep -Seconds 20

Write-Host "3. Запуск основного задания..." -ForegroundColor Yellow
docker-compose up --build task3

Write-Host "4. Проверка результатов..." -ForegroundColor Yellow
if (Test-Path "artifacts/reports/final_registry_report.json") {
    Write-Host "✅ Отчет создан успешно!" -ForegroundColor Green
    Write-Host ""
    
    # Показать краткую информацию
    $report = Get-Content artifacts/reports/final_registry_report.json | ConvertFrom-Json
    Write-Host "ЗАРЕГИСТРИРОВАННАЯ МОДЕЛЬ:" -ForegroundColor Cyan
    Write-Host "  Имя: $($report.model_registry.registered_model_name)"
    Write-Host "  Версия: $($report.model_registry.production_version)"
    Write-Host "  Лучшая модель: $($report.model_registry.best_model)"
    Write-Host "  ROC-AUC: $([math]::Round($report.model_registry.metrics.roc_auc, 4))"
    Write-Host "  Latency: $([math]::Round($report.model_registry.metrics.inference_time_ms, 2)) ms"
    
} else {
    Write-Host "❌ Ошибка при создании отчета" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host " Задание 3 выполнено!" -ForegroundColor Green
Write-Host ""
Write-Host "Доступ к сервисам:" -ForegroundColor Cyan
Write-Host "  • MLflow UI: http://localhost:5000" -ForegroundColor Cyan
Write-Host "  • Experiments: ecommerce_recommendations_v3" -ForegroundColor Cyan
Write-Host "  • Model Registry: EcommerceRecommendationModel" -ForegroundColor Cyan
Write-Host ""
Write-Host "Отчеты сохранены:" -ForegroundColor White
Write-Host "  • artifacts/reports/final_registry_report.json" -ForegroundColor White
Write-Host "  • artifacts/reports/model_training_results.csv" -ForegroundColor White
Write-Host "  • artifacts/reports/validation_report.json" -ForegroundColor White
Write-Host "==========================================" -ForegroundColor Green