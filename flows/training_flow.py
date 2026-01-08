"""
Training script for ER patient forecasting.

This script orchestrates the complete model training pipeline:
1. Load and preprocess raw patient data
2. Fetch weather data
3. Engineer features
4. Train models for all 7 horizons with Optuna
5. Evaluate models on test set
6. Log experiments to MLflow
7. Register models in Model Registry
8. Promote models to production
9. Archive old model versions
"""

import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
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
from src.data.weather_integration import fetch_weather_data, merge_weather_data
from src.data.feature_engineering import (
    engineer_features_for_horizon,
    remove_nan_rows,
)
from src.models.train import train_model_for_horizon
from src.models.evaluation import evaluate_model
from src.utils.mlflow_utils import (
    configure_mlflow_s3_backend,
    log_experiment_params,
    log_experiment_metrics,
    register_model,
    set_experiment,
)
from src.models.model_promotion import (
    should_promote_model,
    promote_to_production,
    archive_old_models,
    get_production_model_mae,
)
from src.monitoring.metrics_collector import (
    record_training_metrics,
    record_model_registration,
    record_model_promotion,
    record_flow_run_status,
    record_data_processing_metrics,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/training.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Environment variables
MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://host.docker.internal:5050')
DB_CONNECTION_URL = os.getenv('DB_CONNECTION_URL', '')
DB_STORED_PROCEDURE = os.getenv('DB_STORED_PROCEDURE', '[getVPB_Data]')

# Configure MLflow S3 backend for MinIO
configure_mlflow_s3_backend()
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)


def training_flow(
    raw_data_path: str = "data/raw/emergency_visits.csv",
    n_optuna_trials: int = 300,
    horizons: Optional[list] = None,
    mae_threshold: float = 15.0
) -> Dict[str, Any]:
    """
    Complete training pipeline for all forecast horizons.
    
    This script:
    - Loads and preprocesses raw patient data
    - Fetches weather data from API
    - Engineers ~50 features
    - Trains 7 LightGBM models with Optuna
    - Evaluates and logs to MLflow
    - Promotes models to production if MAE < threshold
    - Archives old model versions
    
    Args:
        raw_data_path: Path to raw patient visit CSV
        n_optuna_trials: Number of Optuna trials per horizon
        horizons: List of horizons to train (default: [1,2,3,4,5,6,7])
        mae_threshold: MAE threshold for promotion to production
    
    Returns:
        Dictionary with training results and metadata
    """
    if horizons is None:
        horizons = [1, 2, 3, 4, 5, 6, 7]
    
    flow_start_time = time.time()
    flow_success = False
    
    try:
        logger.info("="*70)
        logger.info("STARTING ER PATIENT FORECAST TRAINING FLOW")
        logger.info("="*70)
        logger.info(f"Raw data path: {raw_data_path}")
        logger.info(f"Horizons to train: {horizons}")
        logger.info(f"Optuna trials per horizon: {n_optuna_trials}")
        logger.info(f"MAE threshold for promotion: {mae_threshold}")
        
        # =========================================================================
        # STEP 1: Load and Preprocess Data
        # =========================================================================
        logger.info("\n" + "="*70)
        logger.info("STEP 1: Loading and Preprocessing Patient Data")
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
        
        df = remove_duplicates(df)
        df = aggregate_to_daily(df)
        df = filter_incomplete_current_day(df)  # Remove partial today data
        df = handle_missing_dates(df)
        df = detect_and_handle_outliers(df)  # Cap extreme values using 2x IQR
        step_duration = time.time() - step_start
        
        # Record data processing metrics
        record_data_processing_metrics('preprocess', len(df), step_duration)
        
        logger.info(f"Preprocessed data: {len(df)} days from {df['Date'].min().date()} to {df['Date'].max().date()}")
        
        # =========================================================================
        # STEP 2: Fetch Weather Data
        # =========================================================================
        logger.info("\n" + "="*70)
        logger.info("STEP 2: Fetching Weather Data")
        logger.info("="*70)
        
        start_date = df['Date'].min().strftime('%Y-%m-%d')
        end_date = df['Date'].max().strftime('%Y-%m-%d')
        
        # Hospital coordinates from environment or config
        lat = float(os.getenv('HOSPITAL_LATITUDE', '59.6099'))
        lon = float(os.getenv('HOSPITAL_LONGITUDE', '16.5448'))
        
        weather_df = fetch_weather_data(start_date, end_date, lat, lon)
        df = merge_weather_data(df, weather_df, fallback_strategy='last_known')
        
        logger.info(f"Weather data merged: {len(weather_df)} days of weather")
        
        # Store base data for horizon-specific feature engineering
        df_base = df.copy()
        
        # =========================================================================
        # STEP 3 & 4: Engineer Features and Train Models for All Horizons
        # =========================================================================
        # Note: Feature engineering is now horizon-specific (Approach B)
        # Each horizon gets date/weather features from its TARGET date
        logger.info("\n" + "="*70)
        logger.info("STEP 3 & 4: Training Models with Horizon-Specific Features")
        logger.info("="*70)
        logger.info("Using TARGET-DAY features: date/weather from target date, lags from prediction day")
        
        # Set MLflow experiment
        experiment_name = "ER_Patient_Forecasting_MinIO"
        experiment_id = set_experiment(experiment_name)
        logger.info(f"Set experiment to: {experiment_name} (ID: {experiment_id})")
        
        # Verify the experiment is set correctly
        current_experiment = mlflow.get_experiment_by_name(experiment_name)
        logger.info(f"Current experiment artifact location: {current_experiment.artifact_location}")
        
        training_results = {}
        
        for horizon in horizons:
            logger.info(f"\n{'─'*70}")
            logger.info(f"Training Horizon {horizon} Days Ahead")
            logger.info(f"{'─'*70}")
            
            # Engineer features for this specific horizon (Approach B)
            # Date and weather features are from TARGET date (row_date + horizon)
            # Lag and rolling features are from row_date (prediction day)
            step_start = time.time()
            df = engineer_features_for_horizon(df_base.copy(), weather_df, horizon)
            df = remove_nan_rows(df)
            step_duration = time.time() - step_start
            
            record_data_processing_metrics(f'feature_engineering_h{horizon}', len(df), step_duration)
            
            n_features = len([col for col in df.columns if col not in ['Date', 'Patients_per_day']])
            logger.info(f"Feature engineering complete for horizon {horizon}: {n_features} features, {len(df)} samples")
            
            # Start MLflow run for this horizon
            run_name = f"train_horizon_{horizon}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            logger.info(f"Starting run: {run_name}")
            with mlflow.start_run(run_name=run_name):
                logger.info(f"Run experiment ID: {mlflow.active_run().info.experiment_id}")
                logger.info(f"Run artifact URI: {mlflow.active_run().info.artifact_uri}")
                
                # Log run metadata
                mlflow.log_param("horizon", horizon)
                mlflow.log_param("n_optuna_trials", n_optuna_trials)
                mlflow.log_param("data_start_date", df['Date'].min())
                mlflow.log_param("data_end_date", df['Date'].max())
                mlflow.log_param("n_samples", len(df))
                mlflow.log_param("n_features", n_features)
                mlflow.log_param("feature_approach", "target_day")  # Log that we use Approach B
                
                # Train model with timing
                training_start = time.time()
                model, metadata = train_model_for_horizon(
                    df=df,
                    horizon=horizon,
                    n_trials=n_optuna_trials
                )
                training_duration = time.time() - training_start
                
                # Log best parameters from Optuna
                log_experiment_params(metadata['best_params'])
                
                # Log evaluation metrics
                metrics = {
                    'best_val_mae': metadata['best_val_mae'],
                    'test_mae': metadata['test_mae'],
                    'test_rmse': metadata['test_rmse'],
                }
                
                if metadata.get('test_coverage'):
                    metrics['test_coverage'] = metadata['test_coverage']
                    metrics['test_interval_width'] = metadata['test_interval_width']
                
                log_experiment_metrics(metrics, step=horizon)
                
                # Record training metrics to Prometheus
                record_training_metrics(
                    horizon=horizon,
                    mae=metadata['test_mae'],
                    rmse=metadata['test_rmse'],
                    coverage=metadata.get('test_coverage', 0.0),
                    interval_width=metadata.get('test_interval_width', 0.0),
                    duration=training_duration,
                    n_trials=n_optuna_trials,
                    dataset='test',
                    success=True
                )
                
                # Save model locally
                model_dir = Path("models")
                model_dir.mkdir(exist_ok=True)
                model_path = model_dir / f"forecaster_horizon_{horizon}.pkl"
                model.save_model(str(model_path))
                
                # Log model to MLflow as pickle file
                model_name = f"er_forecast_horizon_{horizon}"
                run_id = mlflow.active_run().info.run_id
                
                try:
                    # Load the model and log it using sklearn flavor
                    import joblib
                    model = joblib.load(model_path)
                    
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path=f"horizon_{horizon}",
                        registered_model_name=model_name
                    )

                    client = mlflow.tracking.MlflowClient()
                    latest_version = client.get_latest_versions(model_name, stages=["None"])[0].version
                    version = latest_version
                except Exception as e:
                    logger.warning(f"Could not register model in MLflow: {e}")
                    version = "local"
                
                logger.info(f"Model registered: {model_name} v{version}")
                
                # Record model registration
                record_model_registration(horizon=horizon, stage='Staging')
                
                # ================================================================
                # STEP 5: Evaluate for Production Promotion
                # ================================================================
                test_mae = metadata['test_mae']
                
                # Get current production model MAE for comparison
                baseline_mae = get_production_model_mae(model_name)
                
                # Decide on promotion
                should_promote = should_promote_model(
                    test_mae=test_mae,
                    threshold=mae_threshold,
                    baseline_mae=baseline_mae
                )
                
                if should_promote and version != "local":
                    logger.info(f"[PROMOTE] Model to production (MAE: {test_mae:.4f} < {mae_threshold:.4f})")
                    try:
                        promote_to_production(
                            model_name=model_name,
                            version=version,
                            archive_existing=True
                        )
                        
                        # Record model promotion
                        record_model_promotion(horizon=horizon)
                        record_model_registration(horizon=horizon, stage='Production')
                        
                        mlflow.set_tag("promoted_to_production", "true")
                    except Exception as e:
                        logger.warning(f"Could not promote model (running locally): {e}")
                        mlflow.set_tag("promoted_to_production", "skipped_local")
                elif should_promote:
                    logger.info(f"Model meets promotion criteria (MAE: {test_mae:.4f} < {mae_threshold:.4f}) but skipping MLflow promotion (running locally)")
                    mlflow.set_tag("promoted_to_production", "skipped_local")
                else:
                    logger.warning(f"[SKIP] Model not promoted (MAE: {test_mae:.4f} >= {mae_threshold:.4f})")
                    mlflow.set_tag("promoted_to_production", "false")
                
                # Store results
                training_results[horizon] = {
                    'model': model,
                    'metadata': metadata,
                    'version': version,
                    'promoted': should_promote,
                    'mlflow_run_id': run_id,
                }
        
        # =========================================================================
        # STEP 6: Archive Old Model Versions
        # =========================================================================
        logger.info("\n" + "="*70)
        logger.info("STEP 6: Archiving Old Model Versions")
        logger.info("="*70)
        
        total_archived = 0
        
        try:
            for horizon in horizons:
                model_name = f"er_forecast_horizon_{horizon}"
                n_archived = archive_old_models(model_name, keep_n=52)
                total_archived += n_archived
            
            logger.info(f"Archived {total_archived} old model versions across all horizons")
        except Exception as e:
            logger.warning(f"Could not archive models (running locally): {e}")
        
        # =========================================================================
        # STEP 7: Create Summary Report
        # =========================================================================
        logger.info("\n" + "="*70)
        logger.info("STEP 7: Creating Summary Report")
        logger.info("="*70)
        
        # Create markdown summary
        summary_md = _create_training_summary(training_results, n_optuna_trials, mae_threshold)
        
        # Save summary to file
        summary_path = Path("logs") / f"training_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        summary_path.write_text(summary_md)
        logger.info(f"Training summary saved to: {summary_path}")
        
        logger.info("\n" + "="*70)
        logger.info("TRAINING FLOW COMPLETE! ")
        logger.info("="*70)
        
        flow_success = True
        
        # Return summary
        return {
            'models_trained': len(training_results),
            'models_promoted': sum(1 for r in training_results.values() if r['promoted']),
            'versions_archived': total_archived,
            'training_results': training_results,
            'timestamp': datetime.now(),
        }
    
    except Exception as e:
        logger.error(f"Training flow failed: {e}", exc_info=True)
        flow_success = False
        record_flow_run_status('training_flow', flow_success)
        sys.exit(1)
    
    finally:
        # Record flow run status
        if flow_success:
            record_flow_run_status('training_flow', flow_success)


def _create_training_summary(
    results: Dict[int, Dict[str, Any]],
    n_trials: int,
    threshold: float
) -> str:
    """
    Create markdown summary of training results.
    
    Args:
        results: Dictionary of training results per horizon
        n_trials: Number of Optuna trials used
        threshold: MAE threshold for promotion
    
    Returns:
        Markdown formatted summary
    """
    lines = [
        "# ER Patient Forecast - Training Summary",
        "",
        f"**Training Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"**Optuna Trials per Horizon**: {n_trials}",
        f"**MAE Promotion Threshold**: {threshold:.2f} patients",
        "",
        "## Results by Horizon",
        "",
        "| Horizon | Test MAE | Test RMSE | Coverage | Interval Width | Promoted | Version |",
        "|---------|----------|-----------|----------|----------------|----------|---------|",
    ]
    
    for horizon in sorted(results.keys()):
        r = results[horizon]
        meta = r['metadata']
        
        coverage = meta.get('test_coverage', 0) * 100 if meta.get('test_coverage') else 'N/A'
        interval = meta.get('test_interval_width', 0) if meta.get('test_interval_width') else 'N/A'
        promoted = "Yes" if r['promoted'] else "No"
        
        if isinstance(coverage, float):
            coverage_str = f"{coverage:.1f}%"
        else:
            coverage_str = coverage
        
        if isinstance(interval, float):
            interval_str = f"{interval:.2f}"
        else:
            interval_str = interval
        
        lines.append(
            f"| {horizon} day{'s' if horizon > 1 else ''} | "
            f"{meta['test_mae']:.4f} | "
            f"{meta['test_rmse']:.4f} | "
            f"{coverage_str} | "
            f"{interval_str} | "
            f"{promoted} | "
            f"{r['version']} |"
        )
    
    lines.extend([
        "",
        "## Summary Statistics",
        "",
        f"- **Models Trained**: {len(results)}",
        f"- **Models Promoted**: {sum(1 for r in results.values() if r['promoted'])}",
        f"- **Average Test MAE**: {sum(r['metadata']['test_mae'] for r in results.values()) / len(results):.4f}",
        "",
        "## Next Steps",
        "",
        "- Models are now available in MLflow Model Registry",
        "- Promoted models are in Production stage",
        "- Daily prediction flow will use production models",
        "",
    ])
    
    return "\n".join(lines)


if __name__ == "__main__":
    try:
        result = training_flow(
            raw_data_path="data/raw/emergency_visits.csv",
            n_optuna_trials=10  # Use fewer trials for testing
        )
        logger.info(f"Training completed successfully: {result['models_trained']} models trained, "
                   f"{result['models_promoted']} promoted")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Training script failed: {e}", exc_info=True)
        sys.exit(1)
