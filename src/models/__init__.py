"""
Model training and prediction modules for ER patient forecasting.

This package contains modules for:
- LightGBM model wrapper with quantile regression
- Model training with Optuna hyperparameter tuning
- Model evaluation metrics
- Model promotion and versioning logic
- Batch prediction generation
"""

from src.models.lightgbm_model import LightGBMForecaster
from src.models.evaluation import (
    calculate_mae,
    calculate_rmse,
    calculate_coverage,
    calculate_interval_width,
    evaluate_model,
)

__all__ = [
    # Models
    "LightGBMForecaster",
    # Evaluation
    "calculate_mae",
    "calculate_rmse",
    "calculate_coverage",
    "calculate_interval_width",
    "evaluate_model",
]
