FROM python:3.9-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY config/ config/
COPY scripts/ scripts/

RUN mkdir -p /app/artifacts/reports \
    /app/artifacts/reference_data \
    /app/artifacts/current_data \
    /app/artifacts/drift_dashboards

CMD ["python", "/app/scripts/schedule_monitoring.py"]