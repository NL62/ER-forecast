#!/usr/bin/env python3
"""
Process raw data and save intermediate results.

This script:
1. Loads raw patient visit data
2. Preprocesses (dedup, aggregate, fill missing dates)
3. Fetches weather data
4. Engineers features
5. Saves all intermediate steps for faster training

Run this before training to cache processed data.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.preprocessing import (
    load_raw_data,
    remove_duplicates,
    aggregate_to_daily,
    handle_missing_dates,
    filter_incomplete_current_day,
)
from src.data.weather_integration import fetch_weather_data, merge_weather_data
from src.data.feature_engineering import engineer_features, remove_nan_rows
from src.utils.logging_config import setup_logging


def main():
    """Process and save all data."""
    
    # Setup logging
    setup_logging(level='INFO', log_file='logs/data_processing.log')
    
    print("="*70)
    print("DATA PROCESSING AND CACHING PIPELINE")
    print("="*70)
    
    # Create output directories
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # =================================================================
    # STEP 1: Load and Preprocess Raw Data
    # =================================================================
    print("\nSTEP 1: Loading raw data...")
    df = load_raw_data('data/raw/emergency_visits.csv')
    print(f"   Loaded: {len(df):,} patient visits")
    
    print("\nSTEP 2: Removing duplicates...")
    df = remove_duplicates(df)
    print(f"   After dedup: {len(df):,} visits")
    
    print("\nSTEP 3: Aggregating to daily counts...")
    df = aggregate_to_daily(df)
    print(f"   Daily data: {len(df):,} days")
    print(f"   Date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"   Avg patients/day: {df['Patients_per_day'].mean():.1f}")
    
    print("\nSTEP 3b: Filtering incomplete current day...")
    df = filter_incomplete_current_day(df)
    print(f"   After filter: {len(df):,} days")
    
    print("\n📅 STEP 4: Filling missing dates...")
    df = handle_missing_dates(df)
    print(f"   Complete: {len(df):,} days")
    
    # Remove days with 0 visits (data quality issue - likely gaps in data collection)
    zero_days = len(df[df['Patients_per_day'] == 0])
    if zero_days > 0:
        print(f"\nSTEP 4b: Removing {zero_days} days with 0 visits (data gaps)...")
        df = df[df['Patients_per_day'] > 0].copy()
        print(f"   Clean data: {len(df):,} days")
    
    # Save daily aggregated data
    daily_path = processed_dir / "daily_patients.csv"
    df.to_csv(daily_path, index=False)
    print(f"\nSaved: {daily_path}")
    
    # =================================================================
    # STEP 5: Fetch Weather Data
    # =================================================================
    print("\nSTEP 5: Fetching weather data from Open-Meteo API...")
    
    start_date = df['Date'].min().strftime('%Y-%m-%d')
    end_date = df['Date'].max().strftime('%Y-%m-%d')
    
    # Hospital coordinates (Sweden - from config)
    lat = 59.6099
    lon = 16.5448
    
    weather_df = fetch_weather_data(start_date, end_date, lat, lon)
    print(f"   Weather data: {len(weather_df):,} days")
    
    # Save weather data
    weather_path = processed_dir / "weather_data.csv"
    weather_df.to_csv(weather_path, index=False)
    print(f"Saved: {weather_path}")
    
    print("\nSTEP 6: Merging weather with patient data...")
    df = merge_weather_data(df, weather_df, fallback_strategy='last_known')
    print(f"   Merged: {len(df):,} days with weather")
    
    # =================================================================
    # STEP 7: Engineer Features
    # =================================================================
    print("\nSTEP 7: Engineering features...")
    print("   This may take a minute...")
    
    df_featured = engineer_features(df)
    n_features = len([col for col in df_featured.columns if col not in ['Date', 'Patients_per_day']])
    print(f"   Created: {n_features} features")
    print(f"   Total columns: {len(df_featured.columns)}")
    
    print("\nSTEP 8: Removing NaN rows (from lag features)...")
    df_clean = remove_nan_rows(df_featured)
    print(f"   Clean data: {len(df_clean):,} samples")
    print(f"   Removed: {len(df_featured) - len(df_clean)} rows with NaN")
    
    # Save featured data
    featured_path = processed_dir / "featured_data.csv"
    df_clean.to_csv(featured_path, index=False)
    print(f"\nSaved: {featured_path}")
    
    # =================================================================
    # Summary
    # =================================================================
    print("\n" + "="*70)
    print("DATA PROCESSING COMPLETE!")
    print("="*70)
    
    print(f"\nSaved files:")
    print(f"   - {daily_path} ({daily_path.stat().st_size / 1024:.1f} KB)")
    print(f"   - {weather_path} ({weather_path.stat().st_size / 1024:.1f} KB)")
    print(f"   - {featured_path} ({featured_path.stat().st_size / 1024 / 1024:.1f} MB)")
    
    print(f"\nFinal dataset:")
    print(f"   - Samples: {len(df_clean):,}")
    print(f"   - Features: {n_features}")
    print(f"   - Date range: {df_clean['Date'].min().date()} to {df_clean['Date'].max().date()}")
    print(f"   - Time span: {(df_clean['Date'].max() - df_clean['Date'].min()).days} days")
    
    print(f"\nNext steps:")
    print(f"   1. Review data: open data/processed/featured_data.csv")
    print(f"   2. Run training: uv run python scripts/train_quick.py")
    print(f"   3. Or use flows: from flows.training_flow import training_flow")
    print("="*70)


if __name__ == "__main__":
    main()

