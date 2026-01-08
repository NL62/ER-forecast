# MLflow Tracking Server
FROM python:3.11-slim

# Set working directory
WORKDIR /mlflow

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Install MLflow and dependencies
RUN pip install --no-cache-dir \
    mlflow==3.5.1 \
    psycopg2-binary==2.9.9 \
    boto3==1.34.10 \
    minio==7.2.0 \
    scikit-learn==1.3.2 \
    lightgbm==4.1.0 \
    pandas==2.1.4 \
    numpy==1.26.4

# Create artifact directory
RUN mkdir -p /mlflow/artifacts

# Expose MLflow port
EXPOSE 5050

# Start MLflow server
CMD ["sh", "-c", "mlflow server --backend-store-uri $BACKEND_STORE_URI --default-artifact-root $MLFLOW_ARTIFACT_ROOT --host 0.0.0.0 --port 5050"]
