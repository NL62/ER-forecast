"""
Monitoring and observability components.

This module provides:
- Prometheus metrics collection
- Model performance tracking over time
- System health monitoring
"""

from src.monitoring.metrics_collector import (
    record_training_metrics,
    record_prediction_metrics,
    record_data_processing_metrics,
    record_weather_api_metrics,
    record_model_registration,
    record_model_promotion,
    record_flow_run_status,
    track_training_duration,
    track_prediction_duration,
    track_data_processing_duration,
    start_metrics_server,
    initialize_metrics
)

from src.monitoring.model_performance_tracker import (
    calculate_rolling_mae,
    log_performance_to_database,
    track_model_performance
)

__all__ = [
    # Metrics collection
    'record_training_metrics',
    'record_prediction_metrics',
    'record_data_processing_metrics',
    'record_weather_api_metrics',
    'record_model_registration',
    'record_model_promotion',
    'record_flow_run_status',
    'track_training_duration',
    'track_prediction_duration',
    'track_data_processing_duration',
    'start_metrics_server',
    'initialize_metrics',
    # Performance tracking
    'calculate_rolling_mae',
    'log_performance_to_database',
    'track_model_performance',
]


