#!/bin/bash

echo " Запуск задания 2: Feature Store"
echo "======================================"

# Создаем директории
mkdir -p artifacts/feature_store
mkdir -p data/feast

echo "1. Запуск инфраструктуры (Redis, PostgreSQL, Feast)..."
docker-compose up -d redis postgres feast-server

echo "2. Ожидание запуска сервисов..."
sleep 30

echo "3. Запуск задания Feature Store..."
docker-compose up --build task2-feature-store

echo "4. Проверка результатов..."
if [ -f "artifacts/feature_store/consistency_report.txt" ]; then
    echo "✅ Отчет создан успешно!"
    echo ""
    cat artifacts/feature_store/consistency_report.txt | tail -50
else
    echo "❌ Ошибка при создании отчета"
    exit 1
fi

echo "======================================"
echo " Задание 2 выполнено!"
echo ""
echo "Доступ к сервисам:"
echo "  • Feast Server: http://localhost:6566"
echo "  • Redis: localhost:6379"
echo "  • PostgreSQL: localhost:5432 (feast/feast)"
echo ""
echo "Результаты сохранены в папке artifacts/feature_store/"
echo "======================================"