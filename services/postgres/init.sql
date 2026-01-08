-- ============================================================================
-- PostgreSQL Initialization Script
-- ER Patient Forecast MLOps System
-- ============================================================================
-- This script creates all necessary databases and users for the MLOps system
-- It runs automatically when the PostgreSQL container is first initialized
-- ============================================================================

-- Create MLflow database and user
CREATE DATABASE mlflow_db;
CREATE USER mlflow_user WITH ENCRYPTED PASSWORD 'mlflow_secure_pwd_2024';
GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlflow_user;

-- Connect to mlflow_db and grant schema privileges
\c mlflow_db;
GRANT ALL ON SCHEMA public TO mlflow_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO mlflow_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO mlflow_user;

-- Create Prefect database and user
\c postgres;
CREATE DATABASE prefect_db;
CREATE USER prefect_user WITH ENCRYPTED PASSWORD 'prefect_secure_pwd_2024';
GRANT ALL PRIVILEGES ON DATABASE prefect_db TO prefect_user;

-- Connect to prefect_db and grant schema privileges
\c prefect_db;
GRANT ALL ON SCHEMA public TO prefect_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO prefect_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO prefect_user;

-- Create Application database and user (for predictions and monitoring)
\c postgres;
CREATE DATABASE er_forecast_db;
CREATE USER app_user WITH ENCRYPTED PASSWORD 'app_secure_pwd_2024';
GRANT ALL PRIVILEGES ON DATABASE er_forecast_db TO app_user;

-- Connect to er_forecast_db and set up application schema
\c er_forecast_db;
GRANT ALL ON SCHEMA public TO app_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO app_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON SEQUENCES TO app_user;

-- Create predictions table
CREATE TABLE IF NOT EXISTS predictions (
    id SERIAL PRIMARY KEY,
    prediction_timestamp TIMESTAMP NOT NULL,
    prediction_date DATE NOT NULL,
    horizon INT NOT NULL CHECK (horizon >= 1 AND horizon <= 7),
    model_version VARCHAR(100) NOT NULL,
    point_prediction FLOAT NOT NULL,
    lower_bound FLOAT NOT NULL,
    upper_bound FLOAT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for faster queries
CREATE INDEX idx_predictions_date ON predictions(prediction_date);
CREATE INDEX idx_predictions_timestamp ON predictions(prediction_timestamp);
CREATE INDEX idx_predictions_horizon ON predictions(horizon);
CREATE INDEX idx_predictions_model_version ON predictions(model_version);

-- Create model performance tracking table
CREATE TABLE IF NOT EXISTS model_performance (
    id SERIAL PRIMARY KEY,
    evaluation_date DATE NOT NULL,
    model_version VARCHAR(100) NOT NULL,
    horizon INT NOT NULL CHECK (horizon >= 1 AND horizon <= 7),
    mae FLOAT NOT NULL,
    rmse FLOAT NOT NULL,
    coverage FLOAT,
    interval_width FLOAT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for model performance
CREATE INDEX idx_model_perf_date ON model_performance(evaluation_date);
CREATE INDEX idx_model_perf_horizon ON model_performance(horizon);
CREATE INDEX idx_model_perf_model_version ON model_performance(model_version);

-- Grant table-specific privileges to app_user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO app_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO app_user;

-- Create a read-only user for dashboards (optional, for security)
CREATE USER dashboard_user WITH ENCRYPTED PASSWORD 'dashboard_secure_pwd_2024';
GRANT CONNECT ON DATABASE er_forecast_db TO dashboard_user;
GRANT USAGE ON SCHEMA public TO dashboard_user;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO dashboard_user;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT SELECT ON TABLES TO dashboard_user;

-- ============================================================================
-- Summary of created resources:
-- 
-- Databases:
--   - mlflow_db: MLflow experiment tracking and model registry
--   - prefect_db: Prefect workflow orchestration metadata
--   - er_forecast_db: Application data (predictions, model performance)
--
-- Users:
--   - mlflow_user: Full access to mlflow_db
--   - prefect_user: Full access to prefect_db
--   - app_user: Full access to er_forecast_db
--   - dashboard_user: Read-only access to er_forecast_db
--
-- Tables in er_forecast_db:
--   - predictions: Stores all prediction outputs with confidence intervals
--   - model_performance: Tracks model evaluation metrics over time
-- ============================================================================
