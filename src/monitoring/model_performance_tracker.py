"""
Model performance tracker for monitoring prediction accuracy over time.

This module provides functionality to:
- Calculate rolling MAE for predictions vs actuals
- Track model drift and degradation
- Log performance metrics to database
- Alert on performance degradation

The performance tracker compares predictions made by the model against
actual patient visit counts to monitor model accuracy in production.
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


def calculate_rolling_mae(
    predictions_df: pd.DataFrame,
    actuals_df: pd.DataFrame,
    window_days: int = 30
) -> pd.DataFrame:
    """
    Calculate rolling Mean Absolute Error over time.
    
    This function compares predictions made on different dates with the actual
    observed patient counts to measure model accuracy over time.
    
    Args:
        predictions_df: DataFrame with columns:
            - prediction_timestamp: when prediction was made
            - prediction_date: date being predicted
            - horizon: forecast horizon (1-7)
            - point_prediction: predicted value
        actuals_df: DataFrame with columns:
            - date: date of observation
            - patient_count: actual patient count
        window_days: Number of days for rolling window calculation
    
    Returns:
        DataFrame with columns:
            - date: date
            - horizon: forecast horizon
            - rolling_mae: rolling MAE over window
            - n_predictions: number of predictions in window
    """
    logger.info(f"Calculating rolling MAE with {window_days}-day window")
    
    try:
        # Ensure date columns are datetime
        predictions_df = predictions_df.copy()
        actuals_df = actuals_df.copy()
        
        predictions_df['prediction_date'] = pd.to_datetime(predictions_df['prediction_date'])
        predictions_df['prediction_timestamp'] = pd.to_datetime(predictions_df['prediction_timestamp'])
        actuals_df['date'] = pd.to_datetime(actuals_df['date'])
        
        # Merge predictions with actuals
        merged = predictions_df.merge(
            actuals_df,
            left_on='prediction_date',
            right_on='date',
            how='inner'
        )
        
        if len(merged) == 0:
            logger.warning("No matching predictions and actuals found")
            return pd.DataFrame(columns=['date', 'horizon', 'rolling_mae', 'n_predictions'])
        
        # Calculate absolute errors
        merged['absolute_error'] = np.abs(
            merged['point_prediction'] - merged['patient_count']
        )
        
        # Calculate rolling MAE for each horizon
        results = []
        
        for horizon in sorted(merged['horizon'].unique()):
            horizon_data = merged[merged['horizon'] == horizon].copy()
            horizon_data = horizon_data.sort_values('prediction_date')
            
            # Calculate rolling mean of absolute errors
            horizon_data['rolling_mae'] = (
                horizon_data['absolute_error']
                .rolling(window=window_days, min_periods=1)
                .mean()
            )
            
            # Count predictions in window
            horizon_data['n_predictions'] = (
                horizon_data['absolute_error']
                .rolling(window=window_days, min_periods=1)
                .count()
            )
            
            # Select relevant columns
            result = horizon_data[['prediction_date', 'horizon', 'rolling_mae', 'n_predictions']].copy()
            result.rename(columns={'prediction_date': 'date'}, inplace=True)
            
            results.append(result)
        
        # Combine all horizons
        rolling_mae_df = pd.concat(results, ignore_index=True)
        
        logger.info(f"Calculated rolling MAE for {len(rolling_mae_df)} data points")
        return rolling_mae_df
        
    except Exception as e:
        logger.error(f"Error calculating rolling MAE: {e}")
        raise


def calculate_point_mae(
    predictions_df: pd.DataFrame,
    actuals_df: pd.DataFrame
) -> Dict[int, float]:
    """
    Calculate point MAE for each horizon (not rolling).
    
    Args:
        predictions_df: Predictions DataFrame
        actuals_df: Actuals DataFrame
    
    Returns:
        Dictionary mapping horizon -> MAE
    """
    try:
        # Merge predictions with actuals
        predictions_df = predictions_df.copy()
        actuals_df = actuals_df.copy()
        
        predictions_df['prediction_date'] = pd.to_datetime(predictions_df['prediction_date'])
        actuals_df['date'] = pd.to_datetime(actuals_df['date'])
        
        merged = predictions_df.merge(
            actuals_df,
            left_on='prediction_date',
            right_on='date',
            how='inner'
        )
        
        # Calculate MAE for each horizon
        mae_by_horizon = {}
        for horizon in sorted(merged['horizon'].unique()):
            horizon_data = merged[merged['horizon'] == horizon]
            mae = np.mean(np.abs(
                horizon_data['point_prediction'] - horizon_data['patient_count']
            ))
            mae_by_horizon[horizon] = float(mae)
        
        return mae_by_horizon
        
    except Exception as e:
        logger.error(f"Error calculating point MAE: {e}")
        return {}


def log_performance_to_database(
    performance_metrics: Dict,
    connection_string: Optional[str] = None
) -> None:
    """
    Log model performance metrics to database.
    
    This is a stub for future database integration. Currently logs to file.
    
    Args:
        performance_metrics: Dictionary containing:
            - timestamp: datetime
            - horizon: int
            - mae: float
            - rolling_mae: float
            - n_predictions: int
        connection_string: Database connection string (for future use)
    """
    logger.info("Logging performance metrics to database (stub)")
    
    try:
        # For now, log to file
        log_dir = Path('logs')
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / 'model_performance.log'
        
        # Format metrics
        timestamp = performance_metrics.get('timestamp', datetime.now())
        horizon = performance_metrics.get('horizon', 'unknown')
        mae = performance_metrics.get('mae', 'N/A')
        rolling_mae = performance_metrics.get('rolling_mae', 'N/A')
        n_predictions = performance_metrics.get('n_predictions', 'N/A')
        
        log_entry = (
            f"{timestamp.isoformat()} | Horizon: {horizon} | "
            f"MAE: {mae} | Rolling MAE (30d): {rolling_mae} | "
            f"N Predictions: {n_predictions}\n"
        )
        
        with open(log_file, 'a') as f:
            f.write(log_entry)
        
        logger.info(f"Performance metrics logged: horizon={horizon}, mae={mae}")
        
        # TODO: Future enhancement - write to PostgreSQL database
        # if connection_string:
        #     engine = create_engine(connection_string)
        #     df = pd.DataFrame([performance_metrics])
        #     df.to_sql('model_performance', engine, if_exists='append', index=False)
        
    except Exception as e:
        logger.error(f"Error logging performance to database: {e}")


def detect_performance_degradation(
    rolling_mae_df: pd.DataFrame,
    baseline_mae: float,
    threshold_percentage: float = 20.0
) -> List[Dict]:
    """
    Detect if model performance has degraded beyond threshold.
    
    Args:
        rolling_mae_df: DataFrame with rolling MAE values
        baseline_mae: Baseline MAE from training/validation
        threshold_percentage: Degradation threshold as percentage
    
    Returns:
        List of dictionaries with degradation alerts
    """
    alerts = []
    
    try:
        # Calculate threshold
        threshold_mae = baseline_mae * (1 + threshold_percentage / 100)
        
        # Check each horizon
        for horizon in rolling_mae_df['horizon'].unique():
            horizon_data = rolling_mae_df[rolling_mae_df['horizon'] == horizon]
            
            # Get most recent rolling MAE
            latest_mae = horizon_data['rolling_mae'].iloc[-1]
            
            if latest_mae > threshold_mae:
                alert = {
                    'horizon': int(horizon),
                    'current_mae': float(latest_mae),
                    'baseline_mae': float(baseline_mae),
                    'threshold_mae': float(threshold_mae),
                    'degradation_percentage': float((latest_mae - baseline_mae) / baseline_mae * 100),
                    'timestamp': datetime.now()
                }
                alerts.append(alert)
                logger.warning(
                    f"Performance degradation detected for horizon {horizon}: "
                    f"MAE {latest_mae:.2f} exceeds threshold {threshold_mae:.2f}"
                )
        
        if not alerts:
            logger.info("No performance degradation detected")
        
        return alerts
        
    except Exception as e:
        logger.error(f"Error detecting performance degradation: {e}")
        return []


def track_model_performance(
    predictions_path: str,
    actuals_path: str,
    window_days: int = 30,
    baseline_maes: Optional[Dict[int, float]] = None,
    connection_string: Optional[str] = None
) -> Dict:
    """
    Prefect task to track model performance over time.
    
    This task:
    1. Loads predictions and actuals
    2. Calculates rolling MAE
    3. Detects performance degradation
    4. Logs metrics to database
    
    Args:
        predictions_path: Path to predictions CSV file
        actuals_path: Path to actuals CSV file
        window_days: Rolling window size in days
        baseline_maes: Dictionary of baseline MAEs per horizon
        connection_string: Database connection string
    
    Returns:
        Dictionary with performance summary
    """
    logger.info("Starting model performance tracking")
    
    try:
        # Load data
        predictions_df = pd.read_csv(predictions_path)
        actuals_df = pd.read_csv(actuals_path)
        
        logger.info(f"Loaded {len(predictions_df)} predictions and {len(actuals_df)} actuals")
        
        # Calculate rolling MAE
        rolling_mae_df = calculate_rolling_mae(
            predictions_df=predictions_df,
            actuals_df=actuals_df,
            window_days=window_days
        )
        
        # Calculate current point MAE
        current_maes = calculate_point_mae(predictions_df, actuals_df)
        
        # Log metrics for each horizon
        for horizon, mae in current_maes.items():
            horizon_rolling = rolling_mae_df[rolling_mae_df['horizon'] == horizon]
            
            if len(horizon_rolling) > 0:
                latest_rolling_mae = horizon_rolling['rolling_mae'].iloc[-1]
                n_predictions = horizon_rolling['n_predictions'].iloc[-1]
            else:
                latest_rolling_mae = mae
                n_predictions = 0
            
            metrics = {
                'timestamp': datetime.now(),
                'horizon': horizon,
                'mae': mae,
                'rolling_mae': latest_rolling_mae,
                'n_predictions': int(n_predictions)
            }
            
            log_performance_to_database(metrics, connection_string)
        
        # Detect degradation if baseline provided
        alerts = []
        if baseline_maes:
            for horizon, baseline_mae in baseline_maes.items():
                horizon_rolling = rolling_mae_df[rolling_mae_df['horizon'] == horizon]
                if len(horizon_rolling) > 0:
                    horizon_alerts = detect_performance_degradation(
                        horizon_rolling,
                        baseline_mae,
                        threshold_percentage=20.0
                    )
                    alerts.extend(horizon_alerts)
        
        # Create summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'n_predictions_evaluated': len(predictions_df),
            'current_maes': current_maes,
            'alerts': alerts,
            'status': 'degraded' if alerts else 'healthy'
        }
        
        logger.info(f"Performance tracking complete: {summary['status']}")
        return summary
        
    except Exception as e:
        logger.error(f"Error in model performance tracking: {e}")
        raise


