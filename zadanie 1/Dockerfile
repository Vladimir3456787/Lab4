FROM python:3.9-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    curl \
    wget \
    git \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Копирование и установка зависимостей
COPY requirements.txt .
RUN pip install --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# Дополнительная установка setuptools для гарантии
RUN pip install --upgrade setuptools==68.2.2

# Копирование исходного кода
COPY src/ src/
COPY scripts/ scripts/

# Создание директорий для артефактов
RUN mkdir -p /app/artifacts/eda \
    /app/artifacts/models \
    /app/artifacts/reports \
    /app/data/processed \
    /app/local_data

WORKDIR /app/src

CMD ["python", "models/trainer.py"]