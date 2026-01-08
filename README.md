# ER Patient Forecast MLOps System

> **Status**: Semi working Prototype/Sketch - Functional end-to-end MLOps pipeline demonstrating automated training, prediction, and monitoring. Not production ready yet. Architecture have to be reworked if we decide to go with AiQu.

An on-premises MLOps system for predicting Emergency Room (ER) patient arrivals for the next 7 days. The system features automated model training, daily batch predictions with confidence intervals, and simple monitoring.

## Overview

This system automates the end-to-end machine learning lifecycle for ER patient forecasting:
- **Automated Training**: Weekly training of 7 LightGBM models (one per forecast horizon: 1-7 days ahead)
- **Daily Predictions**: Batch predictions with point estimates and 95% confidence intervals  
- **Experiment Tracking**: MLflow for model versioning and experiment management
- **System Monitoring**: Grafana dashboards

## Prerequisites

- **Docker** (v20.10+) and **Docker Compose** (v2.0+)
- **Python 3.11+** (for local development)
- **uv** package manager ([installation guide](https://github.com/astral-sh/uv))

## Quick Start

### 1. Clone and Setup

```bash
git clone <repository-url>
cd ER-forecast

# Copy environment template
cp env.template .env
```

### 2. Install Python Dependencies (for local development)

```bash
# Using uv
uv sync

# Or using pip
pip install -e .
```

### 3. Start All Services

```bash
# Start services:
docker compose -f docker-compose.services.yml up -d

# Run training job:
docker compose -f docker-compose.jobs.yml run --build training

# Run prediction job:
docker compose -f docker-compose.jobs.yml run --build prediction

# Stop services:
docker compose -f docker-compose.services.yml down

# Stop and remove volumes:
docker compose -f docker-compose.services.yml down -v
```

This will start:
- **MLflow** (port 5050) - Experiment tracking and model registry
- **MinIO** (port 9000/9001) - S3-compatible artifact storage
- **PostgreSQL** (port 5432) - Database for MLflow metadata
- **Prometheus** (port 9090) - Metrics collection
- **Grafana** (port 3000) - Monitoring dashboards
- **pgAdmin** (port 5050) - Database management (for development)

### 4. Access the UIs

| Service | URL | Default Credentials |
|---------|-----|-------------------|
| MLflow | http://localhost:5050 | No auth |
| Grafana | http://localhost:3000 | admin / admin |
| MinIO Console | http://localhost:9001 | minioadmin / minioadmin123 |

### 5. Prepare Your Data

Place your raw ER patient visit CSV file in `data/raw/`:

```bash
# Expected format: Two columns - Surrogate_Key and Contact_Start
cp your_patient_data.csv data/raw/emergency_visits.csv
```

### 6. Run Training

```bash
# Option 1: Run via Docker (recommended)
docker compose -f docker-compose.jobs.yml run --build training

# Option 2: Run locally
uv run python flows/training_flow.py
```

### 7. Run Predictions

```bash
# Option 1: Run via Docker (recommended)
docker compose -f docker-compose.jobs.yml run --build prediction

# Option 2: Run locally
uv run python flows/prediction_flow.py
```

## Project Structure

```
ER-forecast/
├── data/                   # Data directories
│   ├── raw/                # Raw patient visit data
│   ├── processed/          # Processed features
│   └── predictions/        # Generated predictions
├── flows/                  # Training and prediction scripts
│   ├── training_flow.py    # Model training pipeline
│   └── prediction_flow.py  # Batch prediction pipeline
├── models/                 # Trained model artifacts (local)
├── scripts/                # Utility scripts
├── services/               # Docker service configurations
│   ├── mlflow/             # MLflow server
│   ├── postgres/           # PostgreSQL
│   ├── prometheus/         # Prometheus
│   └── grafana/            # Grafana dashboards
├── src/                    # Source code
│   ├── data/               # Data pipeline modules
│   ├── models/             # Model training and prediction
│   ├── monitoring/         # Metrics collection
│   └── utils/              # Shared utilities
├── tests/                  # Test suite
│   ├── unit/               # Unit tests
│   └── integration/        # Integration tests
├── docker-compose.services.yml  # Backend services
├── docker-compose.jobs.yml      # Training/prediction jobs
├── Makefile                # Common commands
├── pyproject.toml          # Python dependencies
└── README.md               
```

## Workflow Overview

### Training Flow
1. Load raw patient visit data (from SQL Server or CSV)
2. Fetch weather data from Open-Meteo API
3. Engineer features (date features, lags, rolling stats, Fourier features)
4. For each horizon (1-7 days):
   - Run Optuna hyperparameter search (300 trials)
   - Train LightGBM model with best parameters
   - Train quantile regression models for confidence intervals
   - Evaluate on test set
   - Log to MLflow
   - Promote to production if MAE < threshold
5. Archive old model versions (keep last 52)

### Prediction Flow
1. Load latest patient and weather data
2. Engineer features
3. Load production models from MLflow (all 7 horizons)
4. Generate predictions for next 7 days
5. Calculate 95% confidence intervals
6. Save predictions to CSV and/or database
7. Log metadata to MLflow

## Monitoring

### Grafana Dashboards

Access Grafana at http://localhost:3000 to view:
- **System Overview**: CPU, memory, disk usage per container
- **Model Performance**: MAE by horizon over time, training frequency

### MLflow Tracking

Access MLflow at http://localhost:5050 to:
- Compare experiments and hyperparameters
- View model metrics (MAE, RMSE, coverage, interval width)
- Download model artifacts
- Manage model versions and stages

## Model Details

### Architecture
- **Model**: LightGBM
- **Horizons**: 7 separate models (1-7 days ahead)
- **Features**: ~50 engineered features including:
  - Date/time features with cyclic encoding
  - Lag features (1, 2, 3, 7, 14, 21, 28 days)
  - Rolling statistics (3, 14, 30-day windows)
  - Fourier features for seasonal patterns
  - Weather data (temperature, precipitation, snowfall)

### Hyperparameter Tuning
- **Method**: Optuna Bayesian optimization
- **Trials**: 300 per horizon (configurable)
- **Metric**: Mean Absolute Error (MAE)

### Confidence Intervals
- **Method**: Quantile regression (2.5th and 97.5th percentiles)
- **Target Coverage**: 95%

## TODO

- [ ] Replace weather data with forecasted weather for the predictions
- [ ] Create CI/CD pipeline
- [ ] Add model explainability dashboards
- [x] ~~Implement SQL database integration for data loading~~ (Done - stored procedure support)

