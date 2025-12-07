FROM python:3.9-slim

RUN pip install mlflow==2.3.2

EXPOSE 5000

VOLUME /mlflow

CMD ["mlflow", "server", "--backend-store-uri", "sqlite:///mlflow.db", "--default-artifact-root", "/mlflow/artifacts", "--host", "0.0.0.0", "--port", "5000"]