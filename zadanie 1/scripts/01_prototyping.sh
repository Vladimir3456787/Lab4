#!/bin/bash

echo " Запуск задания 1: Прототипирование"
echo "======================================"

# Создаем директории
mkdir -p artifacts/eda
mkdir -p artifacts/models
mkdir -p artifacts/reports
mkdir -p data/raw
mkdir -p data/processed

echo "1. Запуск MLflow..."
docker-compose up -d mlflow

echo "2. Ожидание запуска MLflow..."
sleep 10

echo "3. Запуск задания прототипирования..."
docker-compose up --build task1-prototyping

echo "4. Проверка результатов..."
if [ -f "artifacts/reports/summary_report.txt" ]; then
    echo "✅ Отчет создан успешно!"
    cat artifacts/reports/summary_report.txt
else
    echo "❌ Ошибка при создании отчета"
    exit 1
fi

echo "5. Запуск тестового API..."
docker-compose up -d api-test

echo "======================================"
echo " Задание 1 выполнено!"
echo ""
echo "Доступ к сервисам:"
echo "  • MLflow UI: http://localhost:5000"
echo "  • API Docs: http://localhost:8000/docs"
echo ""
echo "Результаты сохранены в папке artifacts/"
echo "======================================"