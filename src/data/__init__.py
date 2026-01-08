"""
Data pipeline modules for ER patient forecasting.

This package contains modules for:
- Data preprocessing (loading, cleaning, aggregating)
- Weather data integration (API calls)
- Feature engineering (time features, lags, rolling stats, Fourier features)
"""

from src.data.preprocessing import (
    load_raw_data,
    remove_duplicates,
    aggregate_to_daily,
    handle_missing_dates,
    filter_incomplete_current_day,
)

from src.data.weather_integration import (
    fetch_weather_data,
    merge_weather_data,
)

from src.data.feature_engineering import (
    add_date_features,
    add_weekend_indicator,
    add_fourier_features,
    add_lag_features,
    add_rolling_features,
    add_change_features,
    engineer_features,
    remove_nan_rows,
    # New target-day feature functions (Approach B)
    add_date_features_for_target,
    shift_weather_to_target,
    engineer_features_for_horizon,
)

__all__ = [
    # Preprocessing
    "load_raw_data",
    "remove_duplicates",
    "aggregate_to_daily",
    "handle_missing_dates",
    "filter_incomplete_current_day",
    # Weather integration
    "fetch_weather_data",
    "merge_weather_data",
    # Feature engineering
    "add_date_features",
    "add_weekend_indicator",
    "add_fourier_features",
    "add_lag_features",
    "add_rolling_features",
    "add_change_features",
    "engineer_features",
    "remove_nan_rows",
    # Target-day feature functions (Approach B)
    "add_date_features_for_target",
    "shift_weather_to_target",
    "engineer_features_for_horizon",
]
