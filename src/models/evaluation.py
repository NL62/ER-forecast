"""
Model evaluation metrics (MAE, RMSE, coverage, etc).
"""

import logging
from typing import Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Configure module logger
logger = logging.getLogger(__name__)


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Absolute Error."""
    mae = mean_absolute_error(y_true, y_pred)
    logger.debug(f"MAE: {mae:.4f}")
    return float(mae)


def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    logger.debug(f"RMSE: {rmse:.4f}")
    return float(rmse)


def calculate_coverage(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray
) -> float:
    """
    Calculate coverage of confidence intervals.
    
    Coverage is the percentage of actual values that fall within
    the predicted confidence intervals. For 95% CI, coverage should be ~0.95.
    
    Args:
        y_true: Actual values
        y_lower: Lower bounds of confidence intervals
        y_upper: Upper bounds of confidence intervals
    
    Returns:
        Coverage as a fraction (0-1), where 1.0 = 100% coverage
    """
    # Check if actual values fall within intervals
    within_interval = (y_true >= y_lower) & (y_true <= y_upper)
    coverage = within_interval.mean()
    
    logger.debug(f"Coverage: {coverage:.4f} ({coverage*100:.2f}%)")
    
    return float(coverage)


def calculate_interval_width(
    y_lower: np.ndarray,
    y_upper: np.ndarray
) -> float:
    """
    Calculate average width of confidence intervals.
    
    Interval width is the distance between upper and lower bounds.
    Narrower intervals are more informative (but must maintain coverage).
    
    Args:
        y_lower: Lower bounds of confidence intervals
        y_upper: Upper bounds of confidence intervals
    
    Returns:
        Average interval width
    """
    widths = y_upper - y_lower
    avg_width = widths.mean()
    
    logger.debug(f"Average interval width: {avg_width:.4f}")
    logger.debug(f"Width range: [{widths.min():.2f}, {widths.max():.2f}]")
    
    return float(avg_width)


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    return_predictions: bool = False
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation.
    
    Evaluates model on test set and returns all relevant metrics:
    - MAE (Mean Absolute Error)
    - RMSE (Root Mean Squared Error)
    - Coverage (if confidence intervals available)
    - Interval Width (if confidence intervals available)
    
    Args:
        model: Trained model with predict() method
        X_test: Test features
        y_test: Test target values
        return_predictions: If True, include predictions in output
    
    Returns:
        Dictionary containing:
            - mae: Mean Absolute Error
            - rmse: Root Mean Squared Error
            - coverage: Coverage rate (if intervals available)
            - interval_width: Average interval width (if intervals available)
            - predictions: DataFrame with predictions (if return_predictions=True)
            - n_samples: Number of test samples
    """
    logger.info(f"Evaluating model on {len(X_test)} test samples")
    
    # Generate predictions with intervals
    predictions_df = model.predict(X_test, return_intervals=True)
    
    y_pred = predictions_df['point_prediction'].values
    y_true = y_test.values
    
    # Calculate point prediction metrics
    mae = calculate_mae(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    
    metrics = {
        'mae': mae,
        'rmse': rmse,
        'n_samples': len(X_test),
    }
    
    # Calculate interval metrics if available
    if 'lower_bound' in predictions_df.columns and 'upper_bound' in predictions_df.columns:
        y_lower = predictions_df['lower_bound'].values
        y_upper = predictions_df['upper_bound'].values
        
        coverage = calculate_coverage(y_true, y_lower, y_upper)
        interval_width = calculate_interval_width(y_lower, y_upper)
        
        metrics['coverage'] = coverage
        metrics['interval_width'] = interval_width
        
        logger.info(f"Evaluation: MAE={mae:.2f}, RMSE={rmse:.2f}, Coverage={coverage*100:.1f}%, Width={interval_width:.2f}")
    else:
        logger.info(f"Evaluation: MAE={mae:.2f}, RMSE={rmse:.2f} (no confidence intervals)")
        logger.warning("Confidence intervals not available. Coverage and width not calculated.")
    
    # Optionally include predictions
    if return_predictions:
        predictions_df['actual'] = y_true
        predictions_df['error'] = y_true - y_pred
        predictions_df['abs_error'] = np.abs(predictions_df['error'])
        metrics['predictions'] = predictions_df
    
    return metrics


def calculate_mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1.0) -> float:
    """
    Calculate Mean Absolute Percentage Error.
    
    MAPE is useful for understanding relative error magnitude.
    Note: Can be problematic when y_true contains values close to zero.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        epsilon: Small constant to avoid division by zero
    
    Returns:
        MAPE as a percentage (0-100+)
    """
    # Add epsilon to avoid division by zero
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    logger.debug(f"MAPE: {mape:.4f}%")
    
    return float(mape)


def calculate_metrics_by_horizon(
    predictions_dict: Dict[int, pd.DataFrame],
    actuals_dict: Dict[int, pd.Series]
) -> pd.DataFrame:
    """
    Calculate metrics for multiple forecast horizons.
    
    Useful for comparing model performance across different forecast horizons.
    
    Args:
        predictions_dict: Dict mapping horizon -> predictions DataFrame
        actuals_dict: Dict mapping horizon -> actual values Series
    
    Returns:
        DataFrame with metrics for each horizon
    """
    logger.info("Calculating metrics for multiple horizons")
    
    results = []
    
    for horizon in sorted(predictions_dict.keys()):
        preds_df = predictions_dict[horizon]
        actuals = actuals_dict[horizon]
        
        y_pred = preds_df['point_prediction'].values
        y_true = actuals.values
        
        metrics = {
            'horizon': horizon,
            'mae': calculate_mae(y_true, y_pred),
            'rmse': calculate_rmse(y_true, y_pred),
            'n_samples': len(y_true),
        }
        
        # Add interval metrics if available
        if 'lower_bound' in preds_df.columns and 'upper_bound' in preds_df.columns:
            y_lower = preds_df['lower_bound'].values
            y_upper = preds_df['upper_bound'].values
            
            metrics['coverage'] = calculate_coverage(y_true, y_lower, y_upper)
            metrics['interval_width'] = calculate_interval_width(y_lower, y_upper)
        
        results.append(metrics)
    
    results_df = pd.DataFrame(results)
    
    logger.info(f"Calculated metrics for {len(results)} horizons")
    logger.debug(f"MAE range: [{results_df['mae'].min():.2f}, {results_df['mae'].max():.2f}]")
    
    return results_df
