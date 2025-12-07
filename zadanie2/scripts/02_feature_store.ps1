Write-Host " Запуск задания 2: Feature Store" -ForegroundColor Green
Write-Host "======================================" -ForegroundColor Green

# Создаем директории
New-Item -ItemType Directory -Force -Path artifacts/feature_store
New-Item -ItemType Directory -Force -Path data/feast

Write-Host "1. Запуск инфраструктуры (Redis, PostgreSQL, Feast)..." -ForegroundColor Yellow
docker-compose up -d redis postgres feast-server

Write-Host "2. Ожидание запуска сервисов..." -ForegroundColor Yellow
Start-Sleep -Seconds 30

Write-Host "3. Запуск задания Feature Store..." -ForegroundColor Yellow
docker-compose up --build task2-feature-store

Write-Host "4. Проверка результатов..." -ForegroundColor Yellow
if (Test-Path "artifacts/feature_store/consistency_report.txt") {
    Write-Host "✅ Отчет создан успешно!" -ForegroundColor Green
    Write-Host ""
    Get-Content artifacts/feature_store/consistency_report.txt | Select-Object -Last 50
} else {
    Write-Host "❌ Ошибка при создании отчета" -ForegroundColor Red
    exit 1
}

Write-Host "======================================" -ForegroundColor Green
Write-Host " Задание 2 выполнено!" -ForegroundColor Green
Write-Host ""
Write-Host "Доступ к сервисам:" -ForegroundColor Cyan
Write-Host "  • Feast Server: http://localhost:6566" -ForegroundColor Cyan
Write-Host "  • Redis: localhost:6379" -ForegroundColor Cyan
Write-Host "  • PostgreSQL: localhost:5432 (feast/feast)" -ForegroundColor Cyan
Write-Host ""
Write-Host "Результаты сохранены в папке artifacts/feature_store/" -ForegroundColor Cyan
Write-Host "======================================" -ForegroundColor Green