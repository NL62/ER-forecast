#!/usr/bin/env python3
"""
Prediction and visualization script.

This script:
1. Loads trained models
2. Generates predictions on test data
3. Creates comprehensive visualizations
4. Saves plots and prediction results

Usage:
    python scripts/predict_and_visualize.py
    python scripts/predict_and_visualize.py --horizons 1 2 3
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from src.data.preprocessing import (
    load_raw_data,
    remove_duplicates,
    aggregate_to_daily,
    handle_missing_dates,
)
from src.data.weather_integration import fetch_weather_data, merge_weather_data
from src.data.feature_engineering import engineer_features_for_horizon, remove_nan_rows
from src.models.train import split_data, create_shifted_target
from src.models.lightgbm_model import LightGBMForecaster
from src.models.evaluation import evaluate_model

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


def load_model(horizon: int) -> LightGBMForecaster:
    """Load a trained model for specific horizon."""
    model_path = Path(f"models/forecaster_horizon_{horizon}.pkl")
    
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Please train the model first: python scripts/train_standalone.py"
        )
    
    # Use classmethod to load model
    model = LightGBMForecaster.load_model(str(model_path))
    
    return model


def create_prediction_plots(results_df: pd.DataFrame, horizon: int, output_dir: Path):
    """Create prediction time series visualization."""
    
    fig, ax = plt.subplots(figsize=(16, 6))
    
    # Use dates as x-axis
    dates = pd.to_datetime(results_df['date']) if 'date' in results_df.columns else results_df.index
    
    # Highlight weekends with subtle shading
    weekend_patch_added = False
    if 'date' in results_df.columns:
        for i, date in enumerate(dates):
            if date.dayofweek >= 5:  # Saturday (5) or Sunday (6)
                label = 'Weekend' if not weekend_patch_added else None
                # Use date-based spans
                ax.axvspan(date - pd.Timedelta(hours=12), date + pd.Timedelta(hours=12), 
                          alpha=0.12, color='#666666', zorder=0, label=label)
                weekend_patch_added = True
    
    ax.plot(dates, results_df['actual'], 'o-', label='Actual', 
             color='#2E86AB', linewidth=2, markersize=4, alpha=0.7)
    ax.plot(dates, results_df['predicted'], 's-', label='Predicted', 
             color='#A23B72', linewidth=2, markersize=4, alpha=0.7)
    ax.fill_between(dates, 
                      results_df['lower_bound'], 
                      results_df['upper_bound'],
                      alpha=0.2, color='#F18F01', label='95% CI')
    ax.set_xlabel('Date')
    ax.set_ylabel('Number of Patients')
    ax.set_title(f'Horizon {horizon} Day{"s" if horizon > 1 else ""} Ahead - Actual vs Predicted (with 95% CI)', 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Rotate date labels for readability
    plt.xticks(rotation=45, ha='right')
    
    # Add metrics annotation
    mae = np.abs(results_df['actual'] - results_df['predicted']).mean()
    rmse = np.sqrt(((results_df['actual'] - results_df['predicted']) ** 2).mean())
    coverage = ((results_df['actual'] >= results_df['lower_bound']) & 
                (results_df['actual'] <= results_df['upper_bound'])).mean() * 100
    ax.text(0.01, 0.02, f'MAE: {mae:.2f}  |  RMSE: {rmse:.2f}  |  Coverage: {coverage:.1f}%', 
            transform=ax.transAxes, fontsize=10, 
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / f'horizon_{horizon}_predictions.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"   Saved plot: {plot_path}")
    
    plt.close()


def create_coverage_plot(all_results: dict, output_dir: Path):
    """Create coverage plot across all horizons."""
    
    horizons = sorted(all_results.keys())
    coverages = [all_results[h]['coverage'] * 100 for h in horizons]
    maes = [all_results[h]['mae'] for h in horizons]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Model Performance Across Forecast Horizons', fontsize=16, fontweight='bold')
    
    # Coverage plot - zoom in to relevant range (80-100%)
    ax1.bar(horizons, coverages, color='#2E86AB', alpha=0.7, edgecolor='black')
    ax1.axhline(y=95, color='red', linestyle='--', linewidth=2, label='Target 95%')
    ax1.set_xlabel('Forecast Horizon (days)')
    ax1.set_ylabel('Coverage (%)')
    ax1.set_title('Confidence Interval Coverage')
    ax1.set_ylim([80, 102])  # Zoom in to show detail
    ax1.legend(loc='lower right')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (h, c) in enumerate(zip(horizons, coverages)):
        ax1.text(h, c + 0.5, f'{c:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    # MAE plot - set reasonable y-limits based on data
    mae_min, mae_max = min(maes), max(maes)
    mae_margin = (mae_max - mae_min) * 0.3
    ax2.plot(horizons, maes, 'o-', color='#A23B72', linewidth=2, markersize=10)
    ax2.set_xlabel('Forecast Horizon (days)')
    ax2.set_ylabel('Mean Absolute Error (patients)')
    ax2.set_title('Prediction Error by Horizon')
    ax2.set_ylim([mae_min - mae_margin, mae_max + mae_margin * 1.5])
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for h, mae in zip(horizons, maes):
        ax2.text(h, mae + mae_margin * 0.2, f'{mae:.2f}', ha='center', va='bottom', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = output_dir / 'all_horizons_performance.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved summary plot: {plot_path}")
    
    plt.close()


def main():
    """Run prediction and visualization."""
    
    parser = argparse.ArgumentParser(description='Generate predictions and visualizations')
    parser.add_argument('--horizons', type=int, nargs='+', 
                       help='Horizons to predict (default: all available models)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("PREDICTION AND VISUALIZATION")
    print("="*70)
    
    # Create output directories
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)
    
    predictions_dir = Path("data/predictions")
    predictions_dir.mkdir(parents=True, exist_ok=True)
    
    # Find available models
    models_dir = Path("models")
    available_models = sorted([
        int(p.stem.split('_')[-1]) 
        for p in models_dir.glob("forecaster_horizon_*.pkl")
    ])
    
    if not available_models:
        print(" No trained models found in models/")
        print("   Please train models first: python scripts/train_standalone.py --quick")
        return
    
    print(f"\nAvailable trained models: {available_models}")
    
    # Determine which horizons to use
    if args.horizons:
        horizons = [h for h in args.horizons if h in available_models]
        if not horizons:
            print(f" None of the requested horizons {args.horizons} have trained models")
            return
    else:
        horizons = available_models
    
    print(f"Generating predictions for horizons: {horizons}")
    
    # =================================================================
    # Load and Process Data (base data - features engineered per horizon)
    # =================================================================
    print("\nLoading data...")
    
    df_base = load_raw_data('data/raw/emergency_visits.csv')
    df_base = remove_duplicates(df_base)
    df_base = aggregate_to_daily(df_base)
    df_base = handle_missing_dates(df_base)
    
    # Remove days with 0 visits (data quality issue)
    df_base = df_base[df_base['Patients_per_day'] > 0].copy()
    
    # Fetch weather (keep separate for per-horizon feature engineering)
    start_date = df_base['Date'].min().strftime('%Y-%m-%d')
    end_date = df_base['Date'].max().strftime('%Y-%m-%d')
    weather_df = fetch_weather_data(start_date, end_date, 59.6099, 16.5448)
    
    # Merge weather into base (for lag/rolling features that need it)
    df_base = merge_weather_data(df_base, weather_df)
    
    print(f"Base data loaded: {len(df_base)} days")
    
    # =================================================================
    # Generate Predictions for Each Horizon
    # =================================================================
    all_results = {}
    all_predictions = []
    
    for horizon in horizons:
        print(f"\n{'─'*70}")
        print(f"Horizon {horizon} Day{'s' if horizon > 1 else ''} Ahead")
        print(f"{'─'*70}")
        
        # Load model
        print(f"Loading model...")
        model = load_model(horizon)
        
        # Engineer features for this specific horizon (target-day features)
        print(f"Engineering features for horizon {horizon}...")
        df = engineer_features_for_horizon(df_base.copy(), weather_df, horizon)
        df = remove_nan_rows(df)
        
        # Create shifted target and split data
        df_horizon = create_shifted_target(df, horizon, 'Patients_per_day')
        train_df, val_df, calib_df, test_df = split_data(df_horizon)
        
        # Get features
        feature_cols = [col for col in df_horizon.columns 
                       if col not in ['target', 'Date', 'Patients_per_day']]
        
        X_test = test_df[feature_cols]
        y_test = test_df['target']
        dates_test = test_df['Date']
        
        # Calculate target dates (the actual date being predicted)
        target_dates = pd.to_datetime(dates_test) + pd.Timedelta(days=horizon)
        
        print(f"Test set: {len(X_test)} samples, Features: {len(feature_cols)}")
        
        # Generate predictions
        print(f"Generating predictions...")
        predictions = model.predict(X_test, return_intervals=True)
        
        # Create results dataframe with TARGET dates (not prediction dates)
        results_df = pd.DataFrame({
            'date': target_dates.values,
            'actual': y_test.values,
            'predicted': predictions['point_prediction'].values,
            'lower_bound': predictions['lower_bound'].values,
            'upper_bound': predictions['upper_bound'].values,
            'horizon': horizon
        })
        
        # Calculate metrics
        metrics = evaluate_model(model, X_test, y_test)
        
        all_results[horizon] = {
            'mae': metrics['mae'],
            'rmse': metrics['rmse'],
            'coverage': metrics['coverage'],
            'interval_width': metrics['interval_width']
        }
        
        print(f"\nPerformance:")
        print(f"   MAE:              {metrics['mae']:.4f} patients")
        print(f"   RMSE:             {metrics['rmse']:.4f} patients")
        print(f"   Coverage:         {metrics['coverage']*100:.1f}% (target: 95%)")
        print(f"   Interval Width:   {metrics['interval_width']:.2f} patients")
        
        # Create visualizations
        print(f"\nCreating visualizations...")
        create_prediction_plots(results_df.set_index(results_df.index), horizon, output_dir)
        
        # Save predictions
        pred_path = predictions_dir / f'horizon_{horizon}_test_predictions.csv'
        results_df.to_csv(pred_path, index=False)
        print(f"   Saved predictions: {pred_path}")
        
        all_predictions.append(results_df)
    
    # =================================================================
    # Create Summary Visualizations
    # =================================================================
    if len(horizons) > 1:
        print(f"\n{'─'*70}")
        print("Creating summary visualizations...")
        print(f"{'─'*70}")
        create_coverage_plot(all_results, output_dir)
    
    # Save combined predictions
    combined_df = pd.concat(all_predictions, ignore_index=True)
    combined_path = predictions_dir / 'all_horizons_test_predictions.csv'
    combined_df.to_csv(combined_path, index=False)
    print(f"Saved combined predictions: {combined_path}")
    
    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "="*70)
    print("PREDICTION AND VISUALIZATION COMPLETE!")
    print("="*70)
    
    print(f"\nSummary:")
    print(f"{'Horizon':<12} {'MAE':<12} {'RMSE':<12} {'Coverage':<12}")
    print("-" * 50)
    for h in sorted(all_results.keys()):
        r = all_results[h]
        print(f"{h} day{'s' if h > 1 else '':<11} "
              f"{r['mae']:<12.4f} {r['rmse']:<12.4f} {r['coverage']*100:<11.1f}%")
    
    print(f"\nFiles created:")
    print(f"   Plots: {output_dir}/ ({len(horizons)} individual + 1 summary)")
    print(f"   Predictions: {predictions_dir}/")
    
    print(f"\nNext steps:")
    print(f"   1. Review plots: open {output_dir}/")
    print(f"   2. Analyze predictions: open {predictions_dir}/")
    print(f"   3. Train more horizons: python scripts/train_standalone.py --full")
    
    print("="*70)


if __name__ == "__main__":
    main()

