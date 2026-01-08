"""
Prediction script for ER patient forecasting.

This script orchestrates daily batch predictions:
1. Load latest patient and weather data
2. Engineer features
3. Load production models from MLflow
4. Generate predictions for next 7 days
5. Save predictions to CSV
6. Log metadata to MLflow
"""

import logging
import os
import sys
import time
from datetime import date, datetime, timedelta
from typing import Dict, Any, Optional

import mlflow
import pandas as pd

from src.data.preprocessing import (
    load_raw_data,
    load_data_from_database,
    remove_duplicates,
    aggregate_to_daily,
    handle_missing_dates,
    detect_and_handle_outliers,
    filter_incomplete_current_day,
)
from src.data.weather_integration import (
    fetch_weather_data,
    fetch_weather_forecast,
    merge_weather_data,
)
from src.data.feature_engineering import engineer_features, remove_nan_rows
from src.models.predict import (
    load_production_models,
    prepare_prediction_features,
    generate_predictions,
    validate_predictions,
)
from src.models.prediction_output import (
    save_predictions_to_csv,
    write_predictions_to_database,
    log_prediction_metadata_to_mlflow,
    create_prediction_summary,
)
from src.utils.mlflow_utils import configure_mlflow_s3_backend
from src.monitoring.metrics_collector import (
    record_prediction_metrics,
    record_flow_run_status,
    record_data_processing_metrics,
)


def validate_data_freshness(
    df: pd.DataFrame,
    max_staleness_days: int = 1,
    min_rows_required: int = 60
) -> None:
    """
    Validate that patient data is fresh enough for meaningful predictions.
    
    Why freshness matters:
    - The model uses lag features (lag_1, lag_7, etc.) relative to the last data date
    - Predictions are labeled relative to today
    - If data is stale, there's a mismatch: lag_7 won't be "same weekday last week"
      relative to the prediction target, causing incorrect weekly pattern capture
    
    For ER forecasting with strong weekly patterns, data should be at most 1 day old.
    
    Checks:
    1. Most recent data is within max_staleness_days of today
    2. DataFrame has at least min_rows_required rows (for lag features)
    
    Args:
        df: DataFrame with 'Date' column
        max_staleness_days: Maximum allowed days between last data point and today.
            Default is 1 day to preserve weekly pattern alignment in lag features.
        min_rows_required: Minimum number of rows needed for feature engineering
    
    Raises:
        ValueError: If data is too stale or insufficient
    """
    today = datetime.now().date()
    max_date = df['Date'].max()
    
    # Handle both datetime and date types
    if hasattr(max_date, 'date'):
        max_date = max_date.date()
    
    staleness_days = (today - max_date).days
    
    # Check data freshness
    if staleness_days > max_staleness_days:
        raise ValueError(
            f"Data is too stale for predictions. "
            f"Most recent data: {max_date} ({staleness_days} days old). "
            f"Maximum allowed staleness: {max_staleness_days} days. "
            f"Stale data causes lag feature misalignment with prediction dates, "
            f"breaking weekly pattern capture. "
            f"Please ensure the data pipeline is running correctly."
        )
    
    # Check minimum data size (need enough for lag features like lag_28)
    if len(df) < min_rows_required:
        raise ValueError(
            f"Insufficient data for predictions. "
            f"Found {len(df)} rows, need at least {min_rows_required}. "
            f"Lag features require historical data."
        )
    
    logger.info(f"Data validation passed: {len(df)} rows, "
                f"most recent date {max_date} ({staleness_days} days ago)")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/prediction.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Environment variables
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://mlflow:5000')
DB_CONNECTION_URL = os.getenv('DB_CONNECTION_URL', '')
DB_STORED_PROCEDURE = os.getenv('DB_STORED_PROCEDURE', '[getVPB_Data]')

# Configure MLflow S3 backend for MinIO
configure_mlflow_s3_backend()
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


# Change from CSV to sql connection
def prediction_flow(
    raw_data_path: str = "data/raw/emergency_visits.csv",
    output_path: str = "data/predictions/",
    prediction_date: Optional[date] = None,
    save_to_database: bool = False
) -> Dict[str, Any]:
    """
    Complete prediction pipeline for next 7 days.
    
    This script:
    - Loads latest patient data from SQL Server database (via stored procedure) 
      or CSV file (fallback)
    - Fetches weather data from Open-Meteo API
    - Engineers features using historical data
    - Loads production models from MLflow
    - Generates predictions with 95% confidence intervals
    - Saves predictions to CSV (and optionally database)
    - Logs metadata to MLflow
    
    Data Source:
        - If DB_CONNECTION_URL environment variable is set, loads data from
          SQL Server using the stored procedure specified in DB_STORED_PROCEDURE
          (default: '[getVPB_Data]')
        - Otherwise, loads data from the CSV file specified in raw_data_path
    
    Args:
        raw_data_path: Path to raw patient visit CSV (used as fallback if database unavailable)
        output_path: Directory to save prediction CSV files
        prediction_date: Base date for predictions. Defaults to the last date in the
            preprocessed data (after filtering incomplete current day). This ensures
            lag features align correctly with prediction targets, matching training semantics.
        save_to_database: Whether to write predictions to database (default: False)
    
    Returns:
        Dictionary with prediction results and metadata
    """
    # Note: prediction_date default is set AFTER preprocessing to match training semantics.
    # See below after filter_incomplete_current_day() for the default assignment.
    
    flow_start_time = time.time()
    flow_success = False
    
    try:
        logger.info("="*70)
        logger.info("STARTING ER PATIENT FORECAST PREDICTION FLOW")
        logger.info("="*70)
        logger.info(f"Output path: {output_path}")
        
        # =========================================================================
        # STEP 1: Load Latest Data
        # =========================================================================
        logger.info("\n" + "="*70)
        logger.info("STEP 1: Loading Latest Patient Data")
        logger.info("="*70)
        
        step_start = time.time()
        
        # Determine data source: use database if connection string is provided, otherwise CSV
        use_database = bool(DB_CONNECTION_URL)
        
        if use_database:
            logger.info("Loading data from SQL Server database using stored procedure")
            try:
                df = load_data_from_database(
                    connection_string=DB_CONNECTION_URL,
                    stored_procedure=DB_STORED_PROCEDURE
                )
                logger.info(f"Successfully loaded {len(df):,} rows from database")
            except Exception as e:
                logger.warning(f"Failed to load data from database: {e}")
                logger.info(f"Falling back to CSV file: {raw_data_path}")
                df = load_raw_data(raw_data_path)
        else:
            logger.info(f"Loading data from CSV file: {raw_data_path}")
            df = load_raw_data(raw_data_path)
        
        # Apply standard preprocessing steps
        df = remove_duplicates(df)
        df = aggregate_to_daily(df)
        df = filter_incomplete_current_day(df)  # Remove partial today data
        df = handle_missing_dates(df)
        df = detect_and_handle_outliers(df)  # Cap extreme values using 2x IQR
        step_duration = time.time() - step_start
        
        record_data_processing_metrics('load', len(df), step_duration)
        
        logger.info(f"Loaded data: {len(df)} days up to {df['Date'].max().date()}")
        
        # Set prediction_date to last date in data to match training semantics.
        # In training, each row's features (lags, rolling stats) are computed relative to
        # that row's date, and the target is row_date + horizon. For inference to match,
        # base_date must equal the last row's date so that lag features align correctly
        # with the prediction targets.
        if prediction_date is None:
            prediction_date = df['Date'].max().date()
        
        logger.info(f"Prediction base date: {prediction_date}")
        
        # Validate data freshness before proceeding
        # Default is 1 day to preserve weekly pattern alignment in lag features
        # Configurable via environment variables if looser validation is acceptable
        max_staleness = int(os.getenv('MAX_DATA_STALENESS_DAYS', '1'))
        min_rows = int(os.getenv('MIN_DATA_ROWS', '60'))
        validate_data_freshness(df, max_staleness_days=max_staleness, min_rows_required=min_rows)
        
        # =========================================================================
        # STEP 2: Fetch Latest Weather Data (Historical + Forecast)
        # =========================================================================
        logger.info("\n" + "="*70)
        logger.info("STEP 2: Fetching Weather Data (Historical + Forecast)")
        logger.info("="*70)
        
        # Get historical weather data up to today
        start_date = df['Date'].min().strftime('%Y-%m-%d')
        end_date = datetime.now().strftime('%Y-%m-%d')
        
        # Hospital coordinates
        lat = float(os.getenv('HOSPITAL_LATITUDE', '59.6099'))
        lon = float(os.getenv('HOSPITAL_LONGITUDE', '16.5448'))
        
        weather_df = fetch_weather_data(start_date, end_date, lat, lon)
        df = merge_weather_data(df, weather_df, fallback_strategy='last_known')
        
        logger.info(f"Historical weather data: {len(weather_df)} days")
        
        # Fetch weather FORECAST for the next 8 days (covers all 7 horizons)
        try:
            weather_forecast = fetch_weather_forecast(lat, lon, forecast_days=8)
            logger.info(f"Weather forecast fetched: {len(weather_forecast)} days")
            logger.info(f"Forecast date range: {weather_forecast['Date'].min().date()} to "
                       f"{weather_forecast['Date'].max().date()}")
        except Exception as e:
            logger.warning(f"Failed to fetch weather forecast: {e}")
            logger.warning("Predictions will use last known weather for all horizons")
            weather_forecast = None
        
        # =========================================================================
        # STEP 3: Engineer Features (for historical data)
        # =========================================================================
        # Note: This creates lag/rolling features from historical data.
        # The date/weather features here are for historical rows only.
        # prepare_prediction_features() will update date/weather to TARGET dates
        # for each horizon, matching the Approach B training semantics.
        logger.info("\n" + "="*70)
        logger.info("STEP 3: Engineering Features (historical data)")
        logger.info("="*70)
        
        step_start = time.time()
        df = engineer_features(df)
        df = remove_nan_rows(df)
        step_duration = time.time() - step_start
        
        record_data_processing_metrics('feature_engineering', len(df), step_duration)
        
        logger.info(f"Features engineered: {df.shape}")
        
        # =========================================================================
        # STEP 4: Load Production Models
        # =========================================================================
        logger.info("\n" + "="*70)
        logger.info("STEP 4: Loading Production Models from MLflow")
        logger.info("="*70)
        
        models = load_production_models()
        
        logger.info(f"Loaded {len(models)} production models")
        
        # =========================================================================
        # STEP 5: Prepare Features for Prediction (with Weather Forecasts)
        # =========================================================================
        logger.info("\n" + "="*70)
        logger.info("STEP 5: Preparing Prediction Features")
        logger.info("="*70)
        
        prediction_features = prepare_prediction_features(
            df,
            base_date=prediction_date,
            weather_forecast=weather_forecast
        )
        
        logger.info(f"Features prepared for next 7 days")
        if weather_forecast is not None:
            logger.info("Each horizon has weather forecast for its target date")
        
        # =========================================================================
        # STEP 6: Generate Predictions
        # =========================================================================
        logger.info("\n" + "="*70)
        logger.info("STEP 6: Generating Predictions with Confidence Intervals")
        logger.info("="*70)
        
        pred_start_time = time.time()
        predictions = generate_predictions(
            models=models,
            features=prediction_features,
            base_date=prediction_date
        )
        pred_duration = time.time() - pred_start_time
        
        # Validate predictions
        validate_predictions(predictions)
        
        # Record prediction metrics
        horizon_counts = predictions.groupby('horizon').size().to_dict()
        record_prediction_metrics(
            count=len(predictions),
            duration=pred_duration,
            success=True,
            horizon_counts=horizon_counts
        )
        
        logger.info(f"Generated {len(predictions)} predictions in {pred_duration:.2f}s")
        logger.info(f"   Prediction range: {predictions['point_prediction'].min():.1f} to "
                    f"{predictions['point_prediction'].max():.1f} patients")
        
        # =========================================================================
        # STEP 7: Save Predictions
        # =========================================================================
        logger.info("\n" + "="*70)
        logger.info("STEP 7: Saving Predictions")
        logger.info("="*70)
        
        # Save to CSV
        csv_path = save_predictions_to_csv(predictions, output_path)
        logger.info(f"Predictions saved to CSV: {csv_path}")
        
        # Save to database (if enabled)
        if save_to_database and DB_CONNECTION_URL:
            try:
                # Optional: get table name from environment variable, or use default
                db_table_name = os.getenv('DB_PREDICTION_TABLE', '[LTV_STAGE].[dbo].[ER_PREDICTION]')
                n_rows = write_predictions_to_database(
                    predictions,
                    connection_string=DB_CONNECTION_URL,
                    table_name=db_table_name
                )
                logger.info(f"Wrote {n_rows} predictions to database table: {db_table_name}")
            except Exception as e:
                logger.error(f"Failed to write to database: {e}")
                # Don't fail the flow if database write fails
        
        # =========================================================================
        # STEP 8: Log to MLflow
        # =========================================================================
        logger.info("\n" + "="*70)
        logger.info("STEP 8: Logging Metadata to MLflow")
        logger.info("="*70)
        
        mlflow_run_id = log_prediction_metadata_to_mlflow(predictions)
        logger.info(f"Logged to MLflow run: {mlflow_run_id}")
        
        # =========================================================================
        # STEP 9: Create Summary
        # =========================================================================
        summary = create_prediction_summary(predictions)
        
        logger.info("\n" + "="*70)
        logger.info("PREDICTION FLOW COMPLETE! ")
        logger.info("="*70)
        logger.info(f"Average prediction: {summary['mean_prediction']:.1f} patients")
        logger.info(f"Average interval width: {summary.get('mean_interval_width', 'N/A')}")
        
        flow_success = True
        
        return {
            'n_predictions': len(predictions),
            'csv_path': csv_path,
            'mlflow_run_id': mlflow_run_id,
            'summary': summary,
            'timestamp': datetime.now(),
        }
    
    except Exception as e:
        logger.error(f"Prediction flow failed: {e}", exc_info=True)
        flow_success = False
        record_flow_run_status('prediction_flow', flow_success)
        sys.exit(1)
    
    finally:
        # Record flow run status
        if flow_success:
            record_flow_run_status('prediction_flow', flow_success)


if __name__ == "__main__":
    try:
        result = prediction_flow(save_to_database=True)
        logger.info(f"Prediction completed successfully: {result['n_predictions']} predictions")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Prediction script failed: {e}", exc_info=True)
        sys.exit(1)
