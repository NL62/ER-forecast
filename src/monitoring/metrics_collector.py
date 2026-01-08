"""
Prometheus metrics collector for MLOps system.

This module exposes custom metrics for:
- Model training performance (MAE by horizon)
- Prediction generation counts
- Training and prediction durations
- System health indicators

Metrics are exposed on port 8000 for Prometheus to scrape.
"""

import logging
import time
from typing import Dict, Optional
from prometheus_client import Counter, Gauge, Histogram, start_http_server
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# ==============================================================================
# METRIC DEFINITIONS
# ==============================================================================

# Model Performance Metrics
model_mae_by_horizon = Gauge(
    'model_mae_by_horizon',
    'Mean Absolute Error for each forecast horizon',
    ['horizon', 'dataset']  # Labels: horizon (1-7), dataset (train/val/test)
)

model_rmse_by_horizon = Gauge(
    'model_rmse_by_horizon',
    'Root Mean Squared Error for each forecast horizon',
    ['horizon', 'dataset']
)

model_coverage_by_horizon = Gauge(
    'model_coverage_by_horizon',
    'Confidence interval coverage for each forecast horizon',
    ['horizon', 'dataset']
)

model_interval_width_by_horizon = Gauge(
    'model_interval_width_by_horizon',
    'Average prediction interval width for each forecast horizon',
    ['horizon', 'dataset']
)

# Training Metrics
training_count_total = Counter(
    'training_count_total',
    'Total number of training runs completed',
    ['horizon', 'status']  # status: success/failure
)

training_duration_seconds = Histogram(
    'training_duration_seconds',
    'Duration of model training in seconds',
    ['horizon'],
    buckets=[60, 300, 600, 1800, 3600, 7200, 14400]  # 1min to 4hrs
)

optuna_trials_count = Gauge(
    'optuna_trials_count',
    'Number of Optuna trials run during training',
    ['horizon']
)

# Prediction Metrics
prediction_count_total = Counter(
    'prediction_count_total',
    'Total number of prediction batches generated',
    ['status']  # status: success/failure
)

prediction_duration_seconds = Histogram(
    'prediction_duration_seconds',
    'Duration of prediction generation in seconds',
    buckets=[1, 5, 10, 30, 60, 120, 300, 600, 900]  # 1s to 15min
)

predictions_generated_total = Counter(
    'predictions_generated_total',
    'Total number of individual predictions generated',
    ['horizon']
)

# Data Pipeline Metrics
data_processing_duration_seconds = Histogram(
    'data_processing_duration_seconds',
    'Duration of data processing steps in seconds',
    ['step'],  # step: load/preprocess/feature_engineering/weather_fetch
    buckets=[0.1, 0.5, 1, 5, 10, 30, 60, 120]
)

data_rows_processed = Counter(
    'data_rows_processed',
    'Number of data rows processed',
    ['step']
)

# Weather API Metrics
weather_api_requests_total = Counter(
    'weather_api_requests_total',
    'Total number of weather API requests',
    ['status']  # status: success/failure/timeout
)

weather_api_response_time_seconds = Histogram(
    'weather_api_response_time_seconds',
    'Weather API response time in seconds',
    buckets=[0.1, 0.5, 1, 2, 5, 10, 30]
)

# Model Registry Metrics
mlflow_model_registrations_total = Counter(
    'mlflow_model_registrations_total',
    'Total number of models registered in MLflow',
    ['horizon', 'stage']  # stage: Staging/Production/Archived
)

mlflow_model_promotions_total = Counter(
    'mlflow_model_promotions_total',
    'Total number of models promoted to production',
    ['horizon']
)

# System Health Metrics
last_training_timestamp = Gauge(
    'last_training_timestamp',
    'Unix timestamp of last successful training run',
    ['horizon']
)

last_prediction_timestamp = Gauge(
    'last_prediction_timestamp',
    'Unix timestamp of last successful prediction run'
)

flow_run_status = Gauge(
    'flow_run_status',
    'Status of last flow run (1=success, 0=failure)',
    ['flow_name']  # flow_name: training_flow/prediction_flow
)


# ==============================================================================
# METRIC RECORDING FUNCTIONS
# ==============================================================================

def record_training_metrics(
    horizon: int,
    mae: float,
    rmse: float,
    coverage: float,
    interval_width: float,
    duration: float,
    n_trials: int,
    dataset: str = 'test',
    success: bool = True
) -> None:
    """
    Record metrics from a model training run.
    
    Args:
        horizon: Forecast horizon (1-7)
        mae: Mean Absolute Error
        rmse: Root Mean Squared Error
        coverage: Confidence interval coverage (0-1)
        interval_width: Average prediction interval width
        duration: Training duration in seconds
        n_trials: Number of Optuna trials
        dataset: Dataset used for metrics (train/val/test)
        success: Whether training succeeded
    """
    try:
        # Performance metrics
        model_mae_by_horizon.labels(horizon=horizon, dataset=dataset).set(mae)
        model_rmse_by_horizon.labels(horizon=horizon, dataset=dataset).set(rmse)
        model_coverage_by_horizon.labels(horizon=horizon, dataset=dataset).set(coverage)
        model_interval_width_by_horizon.labels(horizon=horizon, dataset=dataset).set(interval_width)
        
        # Training metrics
        status = 'success' if success else 'failure'
        training_count_total.labels(horizon=horizon, status=status).inc()
        training_duration_seconds.labels(horizon=horizon).observe(duration)
        optuna_trials_count.labels(horizon=horizon).set(n_trials)
        
        # Update last training timestamp
        last_training_timestamp.labels(horizon=horizon).set(time.time())
        
        logger.info(f"Recorded training metrics for horizon {horizon}: MAE={mae:.2f}, RMSE={rmse:.2f}")
    except Exception as e:
        logger.error(f"Error recording training metrics: {e}")


def record_prediction_metrics(
    count: int,
    duration: float,
    success: bool = True,
    horizon_counts: Optional[Dict[int, int]] = None
) -> None:
    """
    Record metrics from a prediction generation run.
    
    Args:
        count: Total number of predictions generated
        duration: Prediction duration in seconds
        success: Whether prediction succeeded
        horizon_counts: Optional dict mapping horizon -> count
    """
    try:
        # Prediction metrics
        status = 'success' if success else 'failure'
        prediction_count_total.labels(status=status).inc()
        prediction_duration_seconds.observe(duration)
        
        # Track predictions by horizon if provided
        if horizon_counts:
            for horizon, h_count in horizon_counts.items():
                predictions_generated_total.labels(horizon=horizon).inc(h_count)
        
        # Update last prediction timestamp
        last_prediction_timestamp.set(time.time())
        
        logger.info(f"Recorded prediction metrics: {count} predictions in {duration:.2f}s")
    except Exception as e:
        logger.error(f"Error recording prediction metrics: {e}")


def record_data_processing_metrics(
    step: str,
    rows: int,
    duration: float
) -> None:
    """
    Record metrics from data processing steps.
    
    Args:
        step: Processing step name (load/preprocess/feature_engineering/weather_fetch)
        rows: Number of rows processed
        duration: Processing duration in seconds
    """
    try:
        data_processing_duration_seconds.labels(step=step).observe(duration)
        data_rows_processed.labels(step=step).inc(rows)
        logger.debug(f"Recorded data processing metrics: {step} - {rows} rows in {duration:.2f}s")
    except Exception as e:
        logger.error(f"Error recording data processing metrics: {e}")


def record_weather_api_metrics(
    response_time: float,
    success: bool = True
) -> None:
    """
    Record metrics from weather API calls.
    
    Args:
        response_time: API response time in seconds
        success: Whether API call succeeded
    """
    try:
        status = 'success' if success else 'failure'
        weather_api_requests_total.labels(status=status).inc()
        weather_api_response_time_seconds.observe(response_time)
        logger.debug(f"Recorded weather API metrics: {response_time:.2f}s ({status})")
    except Exception as e:
        logger.error(f"Error recording weather API metrics: {e}")


def record_model_registration(
    horizon: int,
    stage: str = 'Staging'
) -> None:
    """
    Record model registration in MLflow.
    
    Args:
        horizon: Forecast horizon (1-7)
        stage: Model stage (Staging/Production/Archived)
    """
    try:
        mlflow_model_registrations_total.labels(horizon=horizon, stage=stage).inc()
        logger.info(f"Recorded model registration: horizon {horizon}, stage {stage}")
    except Exception as e:
        logger.error(f"Error recording model registration: {e}")


def record_model_promotion(horizon: int) -> None:
    """
    Record model promotion to production.
    
    Args:
        horizon: Forecast horizon (1-7)
    """
    try:
        mlflow_model_promotions_total.labels(horizon=horizon).inc()
        logger.info(f"Recorded model promotion: horizon {horizon}")
    except Exception as e:
        logger.error(f"Error recording model promotion: {e}")


def record_flow_run_status(flow_name: str, success: bool) -> None:
    """
    Record the status of a Prefect flow run.
    
    Args:
        flow_name: Name of the flow (training_flow/prediction_flow)
        success: Whether flow succeeded
    """
    try:
        status_value = 1.0 if success else 0.0
        flow_run_status.labels(flow_name=flow_name).set(status_value)
        logger.info(f"Recorded flow run status: {flow_name} - {'success' if success else 'failure'}")
    except Exception as e:
        logger.error(f"Error recording flow run status: {e}")


# ==============================================================================
# CONTEXT MANAGERS FOR AUTOMATIC DURATION TRACKING
# ==============================================================================

@contextmanager
def track_training_duration(horizon: int):
    """Context manager to automatically track training duration."""
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        training_duration_seconds.labels(horizon=horizon).observe(duration)


@contextmanager
def track_prediction_duration():
    """Context manager to automatically track prediction duration."""
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        prediction_duration_seconds.observe(duration)


@contextmanager
def track_data_processing_duration(step: str):
    """Context manager to automatically track data processing duration."""
    start_time = time.time()
    try:
        yield
    finally:
        duration = time.time() - start_time
        data_processing_duration_seconds.labels(step=step).observe(duration)


# ==============================================================================
# METRICS SERVER
# ==============================================================================

def start_metrics_server(port: int = 8000) -> None:
    """
    Start the Prometheus metrics HTTP server.
    
    Exposes metrics on http://0.0.0.0:{port}/metrics for Prometheus to scrape.
    
    Args:
        port: Port to expose metrics on (default: 8000)
    """
    try:
        start_http_server(port)
        logger.info(f"Metrics server started on port {port}")
        logger.info(f"Metrics available at http://0.0.0.0:{port}/metrics")
    except Exception as e:
        logger.error(f"Failed to start metrics server: {e}")
        raise


# ==============================================================================
# INITIALIZATION
# ==============================================================================

def initialize_metrics() -> None:
    """
    Initialize all metrics with default values.
    
    This ensures all metrics exist before any data is recorded,
    preventing missing metrics in Grafana dashboards.
    """
    logger.info("Initializing metrics with default values...")
    
    # Initialize model metrics for all horizons
    for horizon in range(1, 8):
        for dataset in ['train', 'val', 'test']:
            model_mae_by_horizon.labels(horizon=horizon, dataset=dataset).set(0)
            model_rmse_by_horizon.labels(horizon=horizon, dataset=dataset).set(0)
            model_coverage_by_horizon.labels(horizon=horizon, dataset=dataset).set(0)
            model_interval_width_by_horizon.labels(horizon=horizon, dataset=dataset).set(0)
        
        optuna_trials_count.labels(horizon=horizon).set(0)
        last_training_timestamp.labels(horizon=horizon).set(0)
    
    # Initialize flow status
    for flow_name in ['training_flow', 'prediction_flow']:
        flow_run_status.labels(flow_name=flow_name).set(0)
    
    last_prediction_timestamp.set(0)
    
    logger.info("Metrics initialization complete")


