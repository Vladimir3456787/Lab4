Write-Host " Запуск задания 4: Production инференс" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green

# Создание директорий
New-Item -ItemType Directory -Force -Path artifacts/load_test_results
New-Item -ItemType Directory -Force -Path artifacts/production_models
New-Item -ItemType Directory -Force -Path local_data

# Создание фиктивного Feature Store если нет
$featureStorePath = "local_data/user_features.csv"
if (-Not (Test-Path $featureStorePath)) {
    Write-Host "Создание фиктивного Feature Store..." -ForegroundColor Yellow
    
    # Генерация тестовых данных
    $pythonCode = @"
import pandas as pd
import numpy as np

np.random.seed(42)
n_users = 1000

data = {
    'user_id': range(1, n_users + 1),
    'avg_purchase_amount': np.random.exponential(100, n_users),
    'purchase_frequency': np.random.beta(2, 5, n_users) * 10,
    'days_since_last_purchase': np.random.randint(0, 30, n_users),
    'total_purchases': np.random.poisson(5, n_users),
    'category_preference_fashion': np.random.beta(2, 2, n_users),
    'category_preference_electronics': np.random.beta(1, 3, n_users),
    'category_preference_home': np.random.beta(3, 2, n_users),
    'session_duration_avg': np.random.normal(300, 60, n_users),
    'cart_abandonment_rate': np.random.beta(1, 4, n_users),
    'discount_sensitivity': np.random.beta(2, 3, n_users),
    'device_mobile_ratio': np.random.beta(3, 2, n_users),
    'time_since_registration_days': np.random.randint(0, 365, n_users)
}

df = pd.DataFrame(data)
df.to_csv('$featureStorePath', index=False)
print(f"Feature Store создан: {len(df)} пользователей")
"@

    $pythonCode | python
}

Write-Host "1. Запуск MLflow Tracking Server..." -ForegroundColor Yellow
docker-compose up -d mlflow

Write-Host "2. Ожидание запуска MLflow..." -ForegroundColor Yellow
Start-Sleep -Seconds 20

Write-Host "3. Запуск инференс-сервиса..." -ForegroundColor Yellow
docker-compose up -d inference-api

Write-Host "4. Ожидание запуска API..." -ForegroundColor Yellow
Start-Sleep -Seconds 10

Write-Host "5. Проверка здоровья сервиса..." -ForegroundColor Yellow
$healthResponse = Invoke-RestMethod -Uri "http://localhost:8000/health" -Method Get -ErrorAction SilentlyContinue

if ($healthResponse -and $healthResponse.status -eq "healthy") {
    Write-Host "✅ Сервис запущен успешно!" -ForegroundColor Green
    Write-Host "   Модель: $($healthResponse.model_version)" -ForegroundColor Cyan
    Write-Host "   Статус: $($healthResponse.status)" -ForegroundColor Cyan
} else {
    Write-Host "❌ Ошибка при запуске сервиса" -ForegroundColor Red
    Write-Host "Проверьте логи: docker-compose logs inference-api" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "6. Тестирование API..." -ForegroundColor Yellow

# Тестовые запросы
$testRequests = @(
    @{user_id = 123; transaction_amount = 299.99},
    @{user_id = 456; transaction_amount = 50.00},
    @{user_id = 789; transaction_amount = 1000.00}
)

foreach ($req in $testRequests) {
    try {
        $response = Invoke-RestMethod -Uri "http://localhost:8000/predict" `
                                     -Method Post `
                                     -Body ($req | ConvertTo-Json) `
                                     -ContentType "application/json"
        
        Write-Host "  Запрос: user_id=$($req.user_id), amount=$($req.transaction_amount)" -ForegroundColor White
        Write-Host "  Ответ: prediction=$($response.prediction), latency=$($response.latency_ms)ms" -ForegroundColor Green
    }
    catch {
        Write-Host "  Ошибка: $_" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "7. Запуск нагрузочного тестирования..." -ForegroundColor Yellow
python scripts/load_test.py

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host " Задание 4 выполнено!" -ForegroundColor Green
Write-Host ""
Write-Host "Доступ к сервисам:" -ForegroundColor Cyan
Write-Host "  • MLflow UI: http://localhost:5000" -ForegroundColor Cyan
Write-Host "  • Inference API: http://localhost:8000" -ForegroundColor Cyan
Write-Host "  • API Docs: http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "  • Метрики: http://localhost:8000/metrics" -ForegroundColor Cyan
Write-Host ""
Write-Host "Проверьте результаты нагрузочного тестирования:" -ForegroundColor White
Write-Host "  artifacts/load_test_results/" -ForegroundColor White
Write-Host "==========================================" -ForegroundColor Green