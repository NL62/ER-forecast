#!/usr/bin/env python3
"""
Full system end-to-end test script.

This script tests the complete MLOps pipeline:
1. Training flow with reduced Optuna trials (faster)
2. Prediction flow using trained models
3. Validates outputs and metrics
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set Prefect API URL to localhost for local execution
os.environ['PREFECT_API_URL'] = 'http://localhost:4200/api'
os.environ['MLFLOW_TRACKING_URI'] = 'http://localhost:5050'

# Set MLflow artifact root to local directory (not Docker's /mlflow)
mlflow_artifacts_dir = str(Path(__file__).parent.parent / 'mlruns')
os.environ['MLFLOW_ARTIFACT_ROOT'] = f'file://{mlflow_artifacts_dir}'

from src.utils.logging_config import setup_logging
from flows.training_flow import training_flow
from flows.prediction_flow import prediction_flow


def main():
    """Run full system test."""
    print("="*80)
    print("FULL SYSTEM END-TO-END TEST")
    print("="*80)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Setup logging
    setup_logging(level='INFO')
    
    # =========================================================================
    # STEP 1: Training Flow
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 1: Running Training Flow")
    print("="*80)
    print("This will train 7 models with reduced Optuna trials for speed...")
    print("Expected duration: ~5-10 minutes (instead of hours)")
    print()
    
    training_start = time.time()
    
    try:
        training_results = training_flow(
            raw_data_path="data/raw/emergency_visits.csv",
            n_optuna_trials=10,  # Reduced from 300 for testing
            horizons=[1, 2, 3, 4, 5, 6, 7],
            mae_threshold=15.0
        )
        
        training_duration = time.time() - training_start
        
        print("\nTraining flow completed successfully!")
        print(f"Duration: {training_duration:.1f} seconds ({training_duration/60:.1f} minutes)")
        print(f"Models trained: {training_results['models_trained']}")
        print(f"Models promoted: {training_results['models_promoted']}")
        print(f"Versions archived: {training_results['versions_archived']}")
        
        # Validate outputs
        models_dir = Path("models")
        model_files = list(models_dir.glob("forecaster_horizon_*.pkl"))
        print(f"\nFound {len(model_files)} model files in {models_dir}")
        
    except Exception as e:
        print(f"\n Training flow failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # =========================================================================
    # STEP 2: Prediction Flow
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 2: Running Prediction Flow")
    print("="*80)
    print("This will generate predictions for next 7 days...")
    print("Expected duration: ~30-60 seconds")
    print()
    
    prediction_start = time.time()
    
    try:
        prediction_results = prediction_flow(
            raw_data_path="data/raw/emergency_visits.csv",
            output_path="data/predictions/",
            save_to_database=False
        )
        
        prediction_duration = time.time() - prediction_start
        
        print("\nPrediction flow completed successfully!")
        print(f"Duration: {prediction_duration:.1f} seconds")
        print(f"Predictions generated: {prediction_results['n_predictions']}")
        print(f"CSV saved to: {prediction_results['csv_path']}")
        print(f"MLflow run ID: {prediction_results['mlflow_run_id']}")
        
        # Validate predictions
        import pandas as pd
        predictions_df = pd.read_csv(prediction_results['csv_path'])
        
        print(f"\nPrediction file validation:")
        print(f"   - Rows: {len(predictions_df)}")
        print(f"   - Columns: {list(predictions_df.columns)}")
        print(f"   - Horizons: {sorted(predictions_df['horizon'].unique())}")
        print(f"   - Date range: {predictions_df['prediction_date'].min()} to {predictions_df['prediction_date'].max()}")
        print(f"\n   Sample predictions:")
        print(predictions_df[['prediction_date', 'horizon', 'point_prediction', 'lower_bound', 'upper_bound']].head(10))
        
    except Exception as e:
        print(f"\n Prediction flow failed: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # =========================================================================
    # STEP 3: Validate Outputs
    # =========================================================================
    print("\n" + "="*80)
    print("STEP 3: Validating System Outputs")
    print("="*80)
    
    # Check models directory
    models_dir = Path("models")
    if models_dir.exists():
        model_files = list(models_dir.glob("*.pkl"))
        print(f"Models directory: {len(model_files)} files")
    else:
        print(f" Models directory not found")
    
    # Check predictions directory
    predictions_dir = Path("data/predictions")
    if predictions_dir.exists():
        prediction_files = list(predictions_dir.glob("*.csv"))
        print(f"Predictions directory: {len(prediction_files)} files")
    else:
        print(f" Predictions directory not found")
    
    # Check logs
    logs_dir = Path("logs")
    if logs_dir.exists():
        log_files = list(logs_dir.glob("*.log"))
        print(f"Logs directory: {len(log_files)} files")
    else:
        print(f" Logs directory not found")
    
    # =========================================================================
    # Summary
    # =========================================================================
    total_duration = training_duration + prediction_duration
    
    print("\n" + "="*80)
    print("TEST COMPLETE! ")
    print("="*80)
    print(f"Total duration: {total_duration:.1f} seconds ({total_duration/60:.1f} minutes)")
    print(f"\nNext steps:")
    print(f"1. Check MLflow UI: http://localhost:5000")
    print(f"2. Check Prefect UI: http://localhost:4200")
    print(f"3. Check Grafana dashboards: http://localhost:3000")
    print(f"   - System Overview dashboard")
    print(f"   - Model Performance dashboard")
    print(f"   - Prefect Monitoring dashboard")
    print(f"4. Check Prometheus metrics: http://localhost:9090")
    print(f"5. Check Prometheus metrics endpoint: http://localhost:8000/metrics")
    print(f"\nCredentials:")
    print(f"   Grafana - admin / admin")
    print()
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

