"""
Feature engineering for time series forecasting.
"""

import logging
from typing import List, Optional

import numpy as np
import pandas as pd

# Configure module logger
logger = logging.getLogger(__name__)


def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add date and time features with cyclic encoding (sin/cos for day, month, etc).
    """
    logger.info("Adding date and time features with cyclic encoding")
    
    df = df.copy()
    
    df['Date'] = pd.to_datetime(df['Date'])
    
    df['Day_of_week'] = df['Date'].dt.dayofweek
    df['Month'] = df['Date'].dt.month
    df['Day_of_month'] = df['Date'].dt.day
    
    df['Day_of_week_sin'] = np.sin(2 * np.pi * df['Day_of_week'] / 7)
    df['Day_of_week_cos'] = np.cos(2 * np.pi * df['Day_of_week'] / 7)
    
    # Normalize to 0-11 for proper cyclic encoding
    month_normalized = df['Month'] - 1
    df['Month_sin'] = np.sin(2 * np.pi * month_normalized / 12)
    df['Month_cos'] = np.cos(2 * np.pi * month_normalized / 12)
    
    dom_normalized = df['Day_of_month'] - 1
    df['Day_of_month_sin'] = np.sin(2 * np.pi * dom_normalized / 31)
    df['Day_of_month_cos'] = np.cos(2 * np.pi * dom_normalized / 31)
    
    day_dummies = pd.get_dummies(df['Day_of_week'], prefix='Day_of_week', drop_first=True)
    df = pd.concat([df, day_dummies], axis=1)
    
    df = df.drop(columns=['Day_of_week', 'Month', 'Day_of_month'])
    
    n_features = len([col for col in df.columns if 'sin' in col or 'cos' in col or 'Day_of_week' in col])
    logger.info(f"Added {n_features} date/time features")
    
    return df


def add_weekend_indicator(df: pd.DataFrame) -> pd.DataFrame:
    """Add weekend indicator (1 for Sat/Sun, 0 otherwise)."""
    logger.info("Adding weekend indicator")
    
    df = df.copy()
    
    df['Date'] = pd.to_datetime(df['Date'])
    
    df['Weekend'] = df['Date'].dt.dayofweek.isin([5, 6]).astype(int)
    
    weekend_count = df['Weekend'].sum()
    logger.info(f"Weekend days: {weekend_count}/{len(df)} ({weekend_count/len(df)*100:.1f}%)")
    
    return df


def add_fourier_features(
    df: pd.DataFrame,
    periods: Optional[List[float]] = None,
    n_harmonics: Optional[dict] = None
) -> pd.DataFrame:
    """Add Fourier features for capturing seasonal patterns (annual, weekly, etc)."""
    if periods is None:
        periods = [365.25, 7, 3.5, 2.3]
    
    if n_harmonics is None:
        n_harmonics = {365.25: 3, 7: 2, 3.5: 1, 2.3: 1}
    
    logger.info(f"Adding Fourier features for periods: {periods}")
    
    df = df.copy()
    
    # Ensure Date is datetime
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Calculate days since start (for Fourier features)
    min_date = df['Date'].min()
    df['days_since_start'] = (df['Date'] - min_date).dt.days
    
    feature_count = 0
    
    # Create Fourier features for each period
    for period in periods:
        n_harm = n_harmonics.get(period, 1)
        
        # Determine name prefix based on period
        if period >= 365:
            prefix = 'annual'
        elif period >= 6 and period <= 8:
            prefix = 'weekly'
        elif period >= 3 and period < 4:
            prefix = 'half_week'
        else:
            prefix = 'short_cycle'
        
        # Create harmonics for this period
        for k in range(1, n_harm + 1):
            df[f'{prefix}_sin_{k}'] = np.sin(2 * np.pi * k * df['days_since_start'] / period)
            df[f'{prefix}_cos_{k}'] = np.cos(2 * np.pi * k * df['days_since_start'] / period)
            feature_count += 2
    
    # Drop temporary column
    df = df.drop(columns=['days_since_start'])
    
    logger.info(f"Added {feature_count} Fourier features")
    
    return df


def add_lag_features(
    df: pd.DataFrame,
    lags: Optional[List[int]] = None,
    target_col: str = 'Patients_per_day'
) -> pd.DataFrame:
    """Add lag features (previous days' patient counts)."""
    if lags is None:
        lags = [1, 2, 3, 7, 14, 21, 28]
    
    logger.info(f"Adding lag features: {lags}")
    
    df = df.copy()
    
    # Verify target column exists
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame")
    
    # Create lag features
    for lag in lags:
        col_name = f'{target_col.lower().replace("_per_day", "")}_lag_{lag}'
        df[col_name] = df[target_col].shift(lag)
    
    # Also add lagged 7-day average (shifted to prevent leakage)
    if f'{target_col}_7d_avg' in df.columns:
        df['patients_7d_avg_lag_7'] = df[f'{target_col}_7d_avg'].shift(7)
    
    logger.info(f"Added {len(lags)} lag features")
    logger.debug(f"Lag features will create NaN for first {max(lags)} rows")
    
    return df


def add_rolling_features(
    df: pd.DataFrame,
    windows: Optional[List[int]] = None,
    target_col: str = 'Patients_per_day'
) -> pd.DataFrame:
    """Add rolling mean and std for different windows (shifted to prevent leakage)."""
    if windows is None:
        windows = [3, 14, 30]
    
    logger.info(f"Adding rolling statistics for windows: {windows}")
    
    df = df.copy()
    
    # Verify target column exists
    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame")
    
    feature_count = 0
    
    # Create rolling features for each window
    for window in windows:
        # Shift by 1 to prevent leakage (use only past data)
        target_shifted = df[target_col].shift(1)
        
        # Rolling mean
        col_name_mean = f'patients_{window}d_avg'
        df[col_name_mean] = target_shifted.rolling(window=window, min_periods=1).mean()
        
        # Rolling standard deviation
        col_name_std = f'patients_{window}d_std'
        df[col_name_std] = target_shifted.rolling(window=window, min_periods=1).std()
        
        feature_count += 2
    
    logger.info(f"Added {feature_count} rolling statistics features")
    
    return df


def add_change_features(
    df: pd.DataFrame,
    target_col: str = 'Patients_per_day'
) -> pd.DataFrame:
    """Add change features (day-over-day, week-over-week differences)."""
    logger.info("Adding change features")
    
    df = df.copy()
    
    target_prefix = target_col.lower().replace("_per_day", "")
    
    # Day-over-day change (lag_1 - lag_2)
    lag_1_col = f'{target_prefix}_lag_1'
    lag_2_col = f'{target_prefix}_lag_2'
    
    if lag_1_col in df.columns and lag_2_col in df.columns:
        df['patients_change_1d'] = df[lag_1_col] - df[lag_2_col]
        logger.debug("Added 1-day change feature")
    else:
        logger.warning(f"Cannot create 1-day change: missing {lag_1_col} or {lag_2_col}")
    
    # Week-over-week change (lag_7 - lag_14)
    lag_7_col = f'{target_prefix}_lag_7'
    lag_14_col = f'{target_prefix}_lag_14'
    
    if lag_7_col in df.columns and lag_14_col in df.columns:
        df['patients_change_7d'] = df[lag_7_col] - df[lag_14_col]
        logger.debug("Added 7-day change feature")
    else:
        logger.warning(f"Cannot create 7-day change: missing {lag_7_col} or {lag_14_col}")
    
    logger.info("Added change features")
    
    return df


def engineer_features(
    df: pd.DataFrame,
    include_date: bool = True,
    include_weekend: bool = True,
    include_fourier: bool = False,  # Disabled: double cyclic encoding with day_of_week sin/cos causes mid-week peaks
    include_lags: bool = True,
    include_rolling: bool = True,
    include_changes: bool = True
) -> pd.DataFrame:
    """Run complete feature engineering pipeline."""
    logger.info("Starting comprehensive feature engineering pipeline")
    original_cols = len(df.columns)
    
    df = df.copy()
    
    # 1. Add date features with cyclic encoding
    if include_date:
        df = add_date_features(df)
    
    # 2. Add weekend indicator
    if include_weekend:
        df = add_weekend_indicator(df)
    
    # 3. Add rolling statistics (before lags to avoid using future data)
    if include_rolling:
        df = add_rolling_features(df)
    
    # 4. Add Fourier features for periodic patterns
    if include_fourier:
        df = add_fourier_features(df)
    
    # 5. Add lag features
    if include_lags:
        df = add_lag_features(df)
    
    # 6. Add change features (requires lag features)
    if include_changes and include_lags:
        df = add_change_features(df)
    
    new_features = len(df.columns) - original_cols
    logger.info(f"Feature engineering complete: added {new_features} features")
    logger.info(f"Final shape: {df.shape}")
    
    return df


def add_date_features_for_target(
    df: pd.DataFrame,
    horizon: int
) -> pd.DataFrame:
    """
    Add date features computed from TARGET DATE (row_date + horizon).
    
    This is used for Approach B training where date features should
    represent the day being predicted, not the day of prediction.
    
    Args:
        df: DataFrame with 'Date' column (the prediction/row date)
        horizon: Forecast horizon in days (1-7)
    
    Returns:
        DataFrame with date features computed from target date
    """
    logger.info(f"Adding date features for TARGET date (horizon={horizon})")
    
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Calculate target dates
    target_dates = df['Date'] + pd.Timedelta(days=horizon)
    
    # Cyclic day of week
    df['Day_of_week_sin'] = np.sin(2 * np.pi * target_dates.dt.dayofweek / 7)
    df['Day_of_week_cos'] = np.cos(2 * np.pi * target_dates.dt.dayofweek / 7)
    
    # Cyclic month (normalize to 0-11)
    month_normalized = target_dates.dt.month - 1
    df['Month_sin'] = np.sin(2 * np.pi * month_normalized / 12)
    df['Month_cos'] = np.cos(2 * np.pi * month_normalized / 12)
    
    # Cyclic day of month (normalize to 0-30)
    dom_normalized = target_dates.dt.day - 1
    df['Day_of_month_sin'] = np.sin(2 * np.pi * dom_normalized / 31)
    df['Day_of_month_cos'] = np.cos(2 * np.pi * dom_normalized / 31)
    
    # Weekend indicator for target date
    df['Weekend'] = (target_dates.dt.dayofweek >= 5).astype(int)
    
    # Day-of-week one-hot dummies for target date (drop first = Monday)
    for i in range(1, 7):
        df[f'Day_of_week_{i}'] = (target_dates.dt.dayofweek == i).astype(int)
    
    n_features = len([col for col in df.columns if 'sin' in col or 'cos' in col or 'Day_of_week' in col])
    logger.info(f"Added {n_features} date/time features for target date")
    
    return df


def shift_weather_to_target(
    df: pd.DataFrame,
    weather_df: pd.DataFrame,
    horizon: int
) -> pd.DataFrame:
    """
    Replace weather features with weather from TARGET DATE (row_date + horizon).
    
    This is used for Approach B training where weather features should
    represent the weather on the day being predicted.
    
    Args:
        df: DataFrame with 'Date' column and weather features
        weather_df: Full weather DataFrame with all dates
        horizon: Forecast horizon in days (1-7)
    
    Returns:
        DataFrame with weather features from target date
    """
    logger.info(f"Shifting weather features to TARGET date (horizon={horizon})")
    
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Calculate target dates
    target_dates = df['Date'] + pd.Timedelta(days=horizon)
    
    # Create weather lookup
    weather_df = weather_df.copy()
    weather_df['Date'] = pd.to_datetime(weather_df['Date'])
    weather_lookup = weather_df.set_index('Date')
    
    weather_cols = ['Temp_Max', 'Temp_Min', 'Temp_Mean', 'Precipitation', 'Snowfall']
    
    # Replace each weather column with target-date values
    for col in weather_cols:
        if col in df.columns and col in weather_lookup.columns:
            new_values = []
            for target_date in target_dates:
                if target_date in weather_lookup.index:
                    new_values.append(weather_lookup.loc[target_date, col])
                else:
                    new_values.append(np.nan)
            df[col] = new_values
            
    n_missing = df[weather_cols].isnull().any(axis=1).sum()
    if n_missing > 0:
        logger.warning(f"{n_missing} rows have missing target-date weather (will be dropped)")
    
    logger.info(f"Shifted weather features to target date for horizon {horizon}")
    
    return df


def engineer_features_for_horizon(
    df: pd.DataFrame,
    weather_df: pd.DataFrame,
    horizon: int,
    include_fourier: bool = False,
    include_lags: bool = True,
    include_rolling: bool = True,
    include_changes: bool = True
) -> pd.DataFrame:
    """
    Engineer features for a specific horizon using TARGET-DAY date/weather features.
    
    This implements Approach B: date and weather features are computed from the
    target date (row_date + horizon), while lag and rolling features remain
    computed from the row date (when the prediction is made).
    
    This ensures training and inference use consistent feature semantics.
    
    Args:
        df: DataFrame with 'Date' and 'Patients_per_day' columns
        weather_df: Full weather DataFrame for target-date lookup
        horizon: Forecast horizon in days (1-7)
        include_fourier: Whether to include Fourier features
        include_lags: Whether to include lag features
        include_rolling: Whether to include rolling statistics
        include_changes: Whether to include change features
    
    Returns:
        DataFrame with all features engineered for this specific horizon
    """
    logger.info(f"Engineering features for horizon {horizon} (target-day approach)")
    original_cols = len(df.columns)
    
    df = df.copy()
    
    # 1. Add date features for TARGET date (not row date!)
    df = add_date_features_for_target(df, horizon)
    
    # 2. Add rolling statistics (from row_date - captures recent trends)
    if include_rolling:
        df = add_rolling_features(df)
    
    # 3. Add Fourier features (disabled by default)
    if include_fourier:
        df = add_fourier_features(df)
    
    # 4. Add lag features (from row_date - relative to prediction day)
    if include_lags:
        df = add_lag_features(df)
    
    # 5. Add change features (requires lag features)
    if include_changes and include_lags:
        df = add_change_features(df)
    
    # 6. Shift weather to TARGET date
    df = shift_weather_to_target(df, weather_df, horizon)
    
    new_features = len(df.columns) - original_cols
    logger.info(f"Feature engineering complete for horizon {horizon}: added {new_features} features")
    logger.info(f"Final shape: {df.shape}")
    
    return df


def remove_nan_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with NaN values created by lag/rolling features."""
    logger.info("Removing rows with NaN values from feature engineering")
    
    original_rows = len(df)
    
    # Find columns with NaN values
    nan_cols = df.columns[df.isnull().any()].tolist()
    
    if nan_cols:
        logger.info(f"Found NaN values in {len(nan_cols)} columns: {nan_cols[:5]}...")
        
        # Count NaN per row
        nan_per_row = df.isnull().sum(axis=1)
        rows_with_nan = (nan_per_row > 0).sum()
        
        logger.info(f"Rows with NaN: {rows_with_nan}/{original_rows} ({rows_with_nan/original_rows*100:.1f}%)")
        
        # Drop rows with any NaN
        df = df.dropna().reset_index(drop=True)
        
        removed_rows = original_rows - len(df)
        logger.info(f"Removed {removed_rows} rows with NaN values")
        logger.info(f"Remaining rows: {len(df)}")
    else:
        logger.info("No NaN values found - no rows removed")
    
    return df
