FROM python:3.9-slim

WORKDIR /app

# Установка системных зависимостей (только необходимых)
RUN apt-get update && apt-get install -y \
    libpq-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Копирование и установка зависимостей
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Копирование кода
COPY src/ src/
COPY feature_repo/ feature_repo/

# Создание директорий
RUN mkdir -p /app/artifacts/feature_store \
    /app/data/feast

WORKDIR /app

CMD ["bash"]