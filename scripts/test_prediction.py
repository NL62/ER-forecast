#!/usr/bin/env python3
"""
Simple prediction test script without Prefect dependencies.
Simulates predictions for 7 days after latest data point.
"""

import sys
from pathlib import Path
from datetime import date, datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import os

from src.data.preprocessing import (
    load_raw_data,
    remove_duplicates,
    aggregate_to_daily,
    handle_missing_dates,
    filter_incomplete_current_day,
)
from src.data.weather_integration import fetch_weather_data, fetch_weather_forecast, merge_weather_data
from src.data.feature_engineering import engineer_features, remove_nan_rows
from src.models.lightgbm_model import LightGBMForecaster


def load_production_models_simple():
    """Load all 7 horizon models from disk."""
    models = {}
    for horizon in range(1, 8):
        model_path = f"models/forecaster_horizon_{horizon}.pkl"
        print(f"Loading model from {model_path}...")
        model = LightGBMForecaster.load_model(model_path)
        models[horizon] = model
    return models


def prepare_features_simple(df, base_date, weather_forecast=None):
    """
    Prepare features for prediction with horizon-specific weather forecasts.
    Creates 7 rows, one per horizon, each with its target date's weather.
    """
    import numpy as np
    
    # Get the most recent complete row as starting point
    latest_row = df.iloc[[-1]].copy()
    
    print(f"\nLatest historical date: {latest_row['Date'].iloc[0]}")
    print(f"Predicting from base date: {base_date}")
    
    # Create 7 rows, one for each horizon
    prediction_rows = []
    weather_cols = ['Temp_Max', 'Temp_Min', 'Temp_Mean', 'Precipitation', 'Snowfall']
    
    for horizon in range(1, 8):
        row = latest_row.copy()
        pred_date = base_date + timedelta(days=horizon)
        row['Date'] = pd.Timestamp(pred_date)
        
        # Update date features for this prediction date
        row['Day_of_week_sin'] = np.sin(2 * np.pi * pred_date.weekday() / 7)
        row['Day_of_week_cos'] = np.cos(2 * np.pi * pred_date.weekday() / 7)
        row['Day_of_month_sin'] = np.sin(2 * np.pi * pred_date.day / 31)
        row['Day_of_month_cos'] = np.cos(2 * np.pi * pred_date.day / 31)
        row['Month_sin'] = np.sin(2 * np.pi * pred_date.month / 12)
        row['Month_cos'] = np.cos(2 * np.pi * pred_date.month / 12)
        row['Weekend'] = 1 if pred_date.weekday() >= 5 else 0
        
        # Update day-of-week one-hot encoding
        for i in range(1, 7):
            col_name = f'Day_of_week_{i}'
            if col_name in row.columns:
                row[col_name] = 1 if pred_date.weekday() == i else 0
        
        # Apply weather forecast for this horizon's target date
        if weather_forecast is not None:
            weather_forecast['Date'] = pd.to_datetime(weather_forecast['Date'])
            forecast_match = weather_forecast[
                weather_forecast['Date'].dt.date == pred_date
            ]
            if len(forecast_match) > 0:
                forecast_row = forecast_match.iloc[0]
                for col in weather_cols:
                    if col in row.columns and col in forecast_row.index:
                        row[col] = forecast_row[col]
                print(f"  Horizon {horizon} ({pred_date}): Forecast Temp={forecast_row['Temp_Mean']:.1f}C, "
                      f"Precip={forecast_row['Precipitation']:.1f}mm")
            else:
                print(f"  Horizon {horizon} ({pred_date}): No forecast, using last known weather")
        
        prediction_rows.append(row)
    
    return pd.concat(prediction_rows, ignore_index=True)


def generate_predictions_simple(models, features, base_date):
    """Generate predictions for all 7 horizons using horizon-specific features."""
    predictions = []
    
    # Get feature columns (exclude Date, Patients_per_day, and target columns)
    feature_cols = [col for col in features.columns 
                   if col not in ['Date', 'Patients_per_day'] 
                   and not col.startswith('target_horizon_')]
    
    for horizon in range(1, 8):
        model = models[horizon]
        prediction_date = base_date + timedelta(days=horizon)
        
        # Use the row corresponding to this horizon (row 0 = horizon 1, etc.)
        row_idx = horizon - 1
        X = features[feature_cols].iloc[row_idx:row_idx+1]
        
        # Make prediction with intervals
        pred_result = model.predict(X, return_intervals=True)
        
        point_pred = pred_result['point_prediction'][0]
        lower = pred_result['lower_bound'][0]
        upper = pred_result['upper_bound'][0]
        
        predictions.append({
            'prediction_timestamp': datetime.now(),
            'prediction_date': prediction_date,
            'horizon': horizon,
            'model_version': f'v1_horizon_{horizon}',
            'point_prediction': point_pred,
            'lower_bound': lower,
            'upper_bound': upper
        })
        
        print(f"  Horizon {horizon} ({prediction_date}): {point_pred:.1f} patients "
              f"[{lower:.1f}, {upper:.1f}]")
    
    return pd.DataFrame(predictions)


def main():
    print("="*70)
    print("ER PATIENT FORECAST - PREDICTION TEST (with Weather Forecasts)")
    print("="*70)
    
    # Step 1: Load and prepare data
    print("\n" + "="*70)
    print("STEP 1: Loading Patient Data")
    print("="*70)
    
    df = load_raw_data("data/raw/emergency_visits.csv")
    df = remove_duplicates(df)
    df = aggregate_to_daily(df)
    df = filter_incomplete_current_day(df)  # Remove partial today data
    df = handle_missing_dates(df)
    
    # Set base_date to last date in data to match training semantics.
    # This ensures lag features align correctly with prediction targets.
    base_date = df['Date'].max().date()
    
    print(f"Loaded {len(df)} days of data")
    print(f"Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"\nBase date (last data): {base_date}")
    print(f"Predicting for: {base_date + timedelta(days=1)} to {base_date + timedelta(days=7)}")
    print(f"Latest patient count: {df['Patients_per_day'].iloc[-1]}")
    
    # Step 2: Fetch weather data (historical + forecast)
    print("\n" + "="*70)
    print("STEP 2: Fetching Weather Data (Historical + Forecast)")
    print("="*70)
    
    start_date = df['Date'].min().strftime('%Y-%m-%d')
    end_date = datetime.now().strftime('%Y-%m-%d')  # Use today for historical weather
    
    # Hospital coordinates (Sweden)
    lat = 59.6099
    lon = 16.5448
    
    # Fetch historical weather
    weather_df = fetch_weather_data(start_date, end_date, lat, lon)
    df = merge_weather_data(df, weather_df, fallback_strategy='last_known')
    print(f"Historical weather data: {len(weather_df)} days")
    
    # Fetch weather FORECAST for the next 8 days (covers all 7 horizons)
    print("\nFetching weather forecast for prediction horizons...")
    try:
        weather_forecast = fetch_weather_forecast(lat, lon, forecast_days=8)
        print(f"Weather forecast: {len(weather_forecast)} days")
        print(f"Forecast date range: {weather_forecast['Date'].min().date()} to "
              f"{weather_forecast['Date'].max().date()}")
    except Exception as e:
        print(f"Warning: Could not fetch weather forecast: {e}")
        print("Will use last known weather for all horizons")
        weather_forecast = None
    
    # Step 3: Engineer features
    print("\n" + "="*70)
    print("STEP 3: Engineering Features")
    print("="*70)
    
    df = engineer_features(df)
    df = remove_nan_rows(df)
    
    print(f"Final dataset shape: {df.shape}")
    print(f"Number of features: {df.shape[1] - 1}")  # Exclude Date column
    
    # Step 4: Load models
    print("\n" + "="*70)
    print("STEP 4: Loading Production Models")
    print("="*70)
    
    models = load_production_models_simple()
    print(f"Loaded {len(models)} models")
    
    # Step 5: Prepare features (with weather forecasts for each horizon)
    print("\n" + "="*70)
    print("STEP 5: Preparing Features for Prediction (with Weather Forecasts)")
    print("="*70)
    
    features = prepare_features_simple(df, base_date, weather_forecast=weather_forecast)
    
    # Step 6: Generate predictions
    print("\n" + "="*70)
    print("STEP 6: Generating Predictions")
    print("="*70)
    
    predictions = generate_predictions_simple(models, features, base_date)
    
    # Step 7: Save predictions
    print("\n" + "="*70)
    print("STEP 7: Saving Predictions")
    print("="*70)
    
    output_dir = Path("data/predictions")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_path = output_dir / f"test_predictions_{timestamp}.csv"
    
    predictions.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")
    
    # Summary
    print("\n" + "="*70)
    print("PREDICTION SUMMARY")
    print("="*70)
    
    print(f"\nPredictions for {base_date + timedelta(days=1)} to {base_date + timedelta(days=7)}:")
    print(f"  Mean prediction: {predictions['point_prediction'].mean():.1f} patients")
    print(f"  Min prediction: {predictions['point_prediction'].min():.1f} patients")
    print(f"  Max prediction: {predictions['point_prediction'].max():.1f} patients")
    print(f"  Mean interval width: {(predictions['upper_bound'] - predictions['lower_bound']).mean():.1f}")
    
    print("\nDetailed predictions:")
    print(predictions[['prediction_date', 'horizon', 'point_prediction', 'lower_bound', 'upper_bound']].to_string(index=False))
    
    print("\n" + "="*70)
    print("PREDICTION TEST COMPLETE!")
    print("="*70)


if __name__ == "__main__":
    main()


