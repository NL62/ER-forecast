"""
Batch prediction module for ER patient forecasting.

This module provides functions for:
- Loading production models from MLflow
- Preparing features for prediction
- Generating daily batch predictions with confidence intervals
"""

import logging
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from src.models.lightgbm_model import LightGBMForecaster
from src.utils.mlflow_utils import get_latest_production_model

# Configure module logger
logger = logging.getLogger(__name__)


def load_production_models(
    model_name_pattern: str = "er_forecast_horizon_{horizon}"
) -> Dict[int, LightGBMForecaster]:
    """
    Load all production models from MLflow Model Registry.
    
    Loads the 7 production models (one per forecast horizon) from MLflow.
    Each model should be in "Production" stage.
    
    Args:
        model_name_pattern: Pattern for model names (should include {horizon} placeholder)
    
    Returns:
        Dictionary mapping horizon (1-7) -> LightGBMForecaster model
    
    Raises:
        Exception: If any production model is not found
    """
    logger.info("Loading production models from MLflow Model Registry")
    
    models = {}
    use_local_fallback = False
    
    for horizon in range(1, 8):  # Horizons 1-7
        model_name = model_name_pattern.format(horizon=horizon)
        
        try:
            logger.debug(f"Loading production model: {model_name}")
            model = get_latest_production_model(model_name)
            
            # MLflow returns pyfunc model, extract LightGBMForecaster
            models[horizon] = model
            
            logger.info(f"Loaded model for horizon {horizon}")
            
        except Exception as e:
            logger.warning(f"Failed to load model from MLflow for horizon {horizon}: {e}")
            
            # Fall back to loading from local file system (for local testing)
            local_model_path = Path(f"models/forecaster_horizon_{horizon}.pkl")
            
            if local_model_path.exists():
                logger.info(f"Falling back to local model file: {local_model_path}")
                from src.models.lightgbm_model import LightGBMForecaster
                
                model = LightGBMForecaster.load_model(str(local_model_path))
                models[horizon] = model
                use_local_fallback = True
                logger.info(f"Loaded model for horizon {horizon} from local file")
            else:
                logger.error(f"No local model file found at {local_model_path}")
                raise Exception(
                    f"Cannot load production model for horizon {horizon}. "
                    f"Model not in MLflow registry and no local file at {local_model_path}"
                ) from e
    
    if use_local_fallback:
        logger.warning("Used local file fallback for model loading (running in local/test mode)")
    else:
        logger.info(f"Successfully loaded all {len(models)} production models from MLflow")
    
    return models


def prepare_prediction_features(
    historical_df: pd.DataFrame,
    base_date: Optional[date] = None,
    weather_forecast: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Prepare features for making predictions.
    
    Creates feature rows for the next 7 days based on historical data.
    Features include all engineered features (date, lags, rolling stats, etc.)
    
    Each row corresponds to a prediction horizon (1-7 days ahead), and if
    weather forecasts are provided, each row gets the forecast weather for
    its specific target date.
    
    Args:
        historical_df: DataFrame with historical data and engineered features
        base_date: Date to predict from (default: last date in historical_df)
        weather_forecast: Optional DataFrame with weather forecasts for next 7 days.
            Should have columns: Date, Temp_Max, Temp_Min, Temp_Mean, Precipitation, Snowfall
    
    Returns:
        DataFrame with 7 rows (one per prediction day/horizon) with all features.
        Row 0 = horizon 1 (tomorrow), Row 6 = horizon 7.
    
    Note:
        The historical data must already have all features engineered.
        This function prepares the "current state" features for prediction,
        with horizon-specific weather forecasts when available.
    """
    logger.info("Preparing features for prediction")
    
    if base_date is None:
        base_date = historical_df['Date'].max().date()
    
    logger.info(f"Base date for predictions: {base_date}")
    
    # Get the last row as the starting point
    # This contains the most recent lags and rolling statistics
    last_row = historical_df.iloc[-1:].copy()
    
    # Create prediction dates (next 7 days)
    prediction_dates = pd.date_range(
        start=base_date + timedelta(days=1),
        periods=7,
        freq='D'
    )
    
    # Create DataFrame for prediction features
    # We'll use the last known feature values and update date-related features
    prediction_features = []
    
    for pred_date in prediction_dates:
        row = last_row.copy()
        row['Date'] = pred_date
        prediction_features.append(row)
    
    pred_df = pd.concat(prediction_features, ignore_index=True)
    
    # Weather feature columns
    weather_cols = ['Temp_Max', 'Temp_Min', 'Temp_Mean', 'Precipitation', 'Snowfall']
    
    # Pre-process weather forecast dates once (not inside loop)
    if weather_forecast is not None:
        weather_forecast = weather_forecast.copy()  # Don't modify original
        weather_forecast['Date'] = pd.to_datetime(weather_forecast['Date'])
    
    # Update features for each prediction date/horizon
    for idx, row in pred_df.iterrows():
        pred_date = pd.to_datetime(row['Date'])
        
        # Update cyclic date features (match normalization from feature engineering)
        # Day of week: 0-6 (already normalized, no change needed)
        pred_df.at[idx, 'Day_of_week_sin'] = np.sin(2 * np.pi * pred_date.dayofweek / 7)
        pred_df.at[idx, 'Day_of_week_cos'] = np.cos(2 * np.pi * pred_date.dayofweek / 7)
        # Day of month: normalize to 0-30 (subtract 1 to match training)
        pred_df.at[idx, 'Day_of_month_sin'] = np.sin(2 * np.pi * (pred_date.day - 1) / 31)
        pred_df.at[idx, 'Day_of_month_cos'] = np.cos(2 * np.pi * (pred_date.day - 1) / 31)
        # Month: normalize to 0-11 (subtract 1 to match training)
        pred_df.at[idx, 'Month_sin'] = np.sin(2 * np.pi * (pred_date.month - 1) / 12)
        pred_df.at[idx, 'Month_cos'] = np.cos(2 * np.pi * (pred_date.month - 1) / 12)
        pred_df.at[idx, 'Weekend'] = 1 if pred_date.dayofweek >= 5 else 0
        
        # Update day-of-week one-hot encoding (if present)
        for i in range(1, 7):
            col_name = f'Day_of_week_{i}'
            if col_name in pred_df.columns:
                pred_df.at[idx, col_name] = 1 if pred_date.dayofweek == i else 0
        
        # Update weather features with forecast for this specific date/horizon
        if weather_forecast is not None:
            forecast_date = pred_date.date() if hasattr(pred_date, 'date') else pred_date
            
            # Find forecast for this target date
            forecast_match = weather_forecast[
                weather_forecast['Date'].dt.date == forecast_date
            ]
            
            if len(forecast_match) > 0:
                forecast_row = forecast_match.iloc[0]
                for col in weather_cols:
                    if col in pred_df.columns and col in forecast_row.index:
                        pred_df.at[idx, col] = forecast_row[col]
                horizon = idx + 1
                logger.debug(f"Horizon {horizon} ({forecast_date}): Using forecast weather - "
                            f"Temp={forecast_row.get('Temp_Mean', 'N/A'):.1f}C, "
                            f"Precip={forecast_row.get('Precipitation', 'N/A'):.1f}mm")
            else:
                horizon = idx + 1
                logger.warning(f"Horizon {horizon} ({forecast_date}): No forecast available, "
                              f"using last known weather")
    
    # Convert Day_of_week columns to int (they may be object type after copying)
    for i in range(1, 7):
        col_name = f'Day_of_week_{i}'
        if col_name in pred_df.columns:
            pred_df[col_name] = pred_df[col_name].astype(int)
    
    # Log summary
    logger.info(f"Prepared features for {len(pred_df)} prediction days")
    logger.info(f"Prediction date range: {pred_df['Date'].min().date()} to {pred_df['Date'].max().date()}")
    
    if weather_forecast is not None:
        logger.info("Weather forecasts applied to prediction features")
        for col in weather_cols:
            if col in pred_df.columns:
                logger.debug(f"  {col} range: {pred_df[col].min():.1f} to {pred_df[col].max():.1f}")
    else:
        logger.warning("No weather forecast provided - using last known weather for all horizons")
    
    logger.info(f"Total columns in prediction features: {len(pred_df.columns)}")
    
    return pred_df


def generate_predictions(
    models: Dict[int, LightGBMForecaster],
    features: pd.DataFrame,
    base_date: Optional[date] = None
) -> pd.DataFrame:
    """
    Generate batch predictions for the next 7 days with confidence intervals.
    
    Uses the 7 trained models (one per horizon) to generate predictions.
    Each prediction includes point estimate and 95% confidence interval.
    
    Each horizon uses its corresponding row from the features DataFrame,
    which should contain horizon-specific weather forecasts.
    
    Args:
        models: Dictionary mapping horizon -> trained model
        features: DataFrame with 7 rows of features (one per horizon).
            Row 0 = horizon 1, Row 6 = horizon 7.
            Each row should have weather features for its target date.
        base_date: Date predictions are made from. Defaults to inferring from the
            features DataFrame (first prediction date - 1 day). This ensures
            prediction dates align with the feature dates prepared by
            prepare_prediction_features().
    
    Returns:
        DataFrame with columns:
            - prediction_timestamp: When prediction was made
            - prediction_date: Date being predicted
            - horizon: Forecast horizon (1-7 days)
            - model_version: MLflow model version
            - point_prediction: Expected patient count
            - lower_bound: Lower 95% CI bound
            - upper_bound: Upper 95% CI bound
    """
    logger.info("Generating batch predictions for next 7 days")
    
    if base_date is None:
        # Infer base_date from features: first row is horizon 1, so base_date = first_date - 1
        if 'Date' in features.columns and len(features) > 0:
            first_pred_date = pd.to_datetime(features['Date'].iloc[0]).date()
            base_date = first_pred_date - timedelta(days=1)
            logger.info(f"Inferred base_date from features: {base_date}")
        else:
            # Fallback to today if no Date column (shouldn't happen in normal usage)
            base_date = datetime.now().date()
            logger.warning(f"Could not infer base_date from features, using today: {base_date}")
    
    prediction_timestamp = datetime.now()
    
    # Get model version from MLflow (instead of hardcoding 'production')
    # Use horizon 1 model as reference - all horizons should have same version
    model_version = get_model_version_from_mlflow("er_forecast_horizon_1")
    logger.info(f"Using model version: {model_version}")
    
    # Get feature columns (exclude Date and target)
    feature_cols = [col for col in features.columns 
                    if col not in ['Date', 'Patients_per_day', 'target']]
    
    logger.info(f"Feature columns selected: {len(feature_cols)} features")
    logger.debug(f"All feature columns: {feature_cols}")
    
    all_predictions = []
    
    # Generate prediction for each horizon
    for horizon in range(1, 8):
        if horizon not in models:
            logger.warning(f"Model for horizon {horizon} not found, skipping")
            continue
        
        model = models[horizon]
        
        # Use the row corresponding to this horizon (row 0 = horizon 1, etc.)
        # Each row has horizon-specific features including weather forecast
        row_idx = horizon - 1
        if row_idx < len(features):
            X = features[feature_cols].iloc[row_idx:row_idx+1]
        else:
            # Fallback to last available row if features DataFrame is shorter
            logger.warning(f"Horizon {horizon}: feature row not available, using last row")
            X = features[feature_cols].iloc[-1:]
        
        # Generate prediction with intervals
        pred = model.predict(X, return_intervals=True)
        
        # Create prediction record
        prediction_date = base_date + timedelta(days=horizon)
        
        record = {
            'prediction_timestamp': prediction_timestamp,
            'prediction_date': prediction_date,
            'horizon': horizon,
            'model_version': model_version,
            'point_prediction': float(pred['point_prediction'].iloc[0]),
            'lower_bound': float(pred.get('lower_bound', [pred['point_prediction'].iloc[0]])[0]),
            'upper_bound': float(pred.get('upper_bound', [pred['point_prediction'].iloc[0]])[0]),
        }
        
        all_predictions.append(record)
        
        logger.debug(f"Horizon {horizon}: {record['point_prediction']:.1f} "
                    f"[{record['lower_bound']:.1f}, {record['upper_bound']:.1f}]")
    
    # Create DataFrame
    predictions_df = pd.DataFrame(all_predictions)
    
    logger.info(f"Generated {len(predictions_df)} predictions")
    logger.info(f"Prediction range: {predictions_df['point_prediction'].min():.1f} to "
                f"{predictions_df['point_prediction'].max():.1f} patients")
    
    return predictions_df


def get_model_version_from_mlflow(model_name: str) -> str:
    """
    Get the version number of the current production model.
    
    Helper function to retrieve actual version numbers from MLflow.
    
    Args:
        model_name: Name of registered model
    
    Returns:
        Version number as string
    """
    from mlflow.tracking import MlflowClient
    
    try:
        client = MlflowClient()
        production_versions = client.get_latest_versions(
            name=model_name,
            stages=["Production"]
        )
        
        if production_versions:
            version = production_versions[0].version
            logger.debug(f"Production version for {model_name}: {version}")
            return str(version)
        else:
            logger.warning(f"No production version found for {model_name}")
            return "unknown"
            
    except Exception as e:
        logger.warning(f"Could not retrieve model version: {e}")
        return "unknown"


def validate_predictions(predictions_df: pd.DataFrame) -> bool:
    """
    Validate prediction output before saving.
    
    Checks that:
    - All required columns are present
    - No missing values
    - Bounds are properly ordered (lower < point < upper)
    - Horizons are in valid range (1-7)
    
    Args:
        predictions_df: DataFrame with prediction results
    
    Returns:
        True if valid, raises ValueError if invalid
    
    Raises:
        ValueError: If validation fails
    """
    logger.debug("Validating prediction output")
    
    # Check required columns
    required_cols = [
        'prediction_timestamp', 'prediction_date', 'horizon',
        'model_version', 'point_prediction', 'lower_bound', 'upper_bound'
    ]
    
    missing_cols = set(required_cols) - set(predictions_df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Check for missing values
    if predictions_df[required_cols].isnull().any().any():
        raise ValueError("Predictions contain missing values")
    
    # Check horizon range
    if not predictions_df['horizon'].between(1, 7).all():
        raise ValueError("Invalid horizon values (must be 1-7)")
    
    # Check bounds ordering
    invalid_bounds = (
        (predictions_df['lower_bound'] > predictions_df['point_prediction']) |
        (predictions_df['point_prediction'] > predictions_df['upper_bound'])
    )
    
    if invalid_bounds.any():
        n_invalid = invalid_bounds.sum()
        raise ValueError(f"{n_invalid} predictions have invalid bounds (lower > point > upper)")
    
    logger.debug("Prediction validation passed")
    
    return True
