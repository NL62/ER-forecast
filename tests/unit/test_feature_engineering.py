"""Tests for feature engineering module."""

import pandas as pd
import pytest
import numpy as np

from src.data.feature_engineering import (
    add_date_features,
    add_weekend_indicator,
    add_fourier_features,
    add_lag_features,
    add_rolling_features,
    add_change_features,
    engineer_features,
    remove_nan_rows,
)


@pytest.fixture
def sample_daily_data():
    dates = pd.date_range('2022-01-01', periods=60, freq='D')
    df = pd.DataFrame({
        'Date': dates,
        'Patients_per_day': np.random.randint(40, 80, size=60)
    })
    return df


class TestAddDateFeatures:
    
    def test_creates_cyclic_features(self, sample_daily_data):
        result = add_date_features(sample_daily_data)
        
        assert 'Day_of_week_sin' in result.columns
        assert 'Day_of_week_cos' in result.columns
        assert 'Month_sin' in result.columns
        assert 'Month_cos' in result.columns
        assert 'Day_of_month_sin' in result.columns
        assert 'Day_of_month_cos' in result.columns
    
    def test_creates_one_hot_day_of_week(self, sample_daily_data):
        result = add_date_features(sample_daily_data)
        
        # Should have dummy variables (drop_first=True, so 6 columns)
        day_cols = [col for col in result.columns if 'Day_of_week_' in col and col not in ['Day_of_week_sin', 'Day_of_week_cos']]
        assert len(day_cols) == 6  # Monday dropped, so 6 remaining (Tue-Sun)
    
    def test_drops_original_categorical(self, sample_daily_data):
        result = add_date_features(sample_daily_data)
        
        assert 'Day_of_week' not in result.columns
        assert 'Month' not in result.columns
        assert 'Day_of_month' not in result.columns
    
    def test_cyclic_encoding_range(self, sample_daily_data):
        result = add_date_features(sample_daily_data)
        
        assert result['Day_of_week_sin'].between(-1, 1).all()
        assert result['Day_of_week_cos'].between(-1, 1).all()
        assert result['Month_sin'].between(-1, 1).all()
        assert result['Month_cos'].between(-1, 1).all()


class TestAddWeekendIndicator:
    
    def test_creates_weekend_column(self, sample_daily_data):
        result = add_weekend_indicator(sample_daily_data)
        
        assert 'Weekend' in result.columns
    
    def test_weekend_values(self):
        df = pd.DataFrame({
            'Date': pd.to_datetime([
                '2022-01-01',  # Saturday
                '2022-01-02',  # Sunday
                '2022-01-03',  # Monday
                '2022-01-04',  # Tuesday
            ])
        })
        
        result = add_weekend_indicator(df)
        
        assert result['Weekend'].tolist() == [1, 1, 0, 0]
    
    def test_weekend_is_binary(self, sample_daily_data):
        result = add_weekend_indicator(sample_daily_data)
        
        assert set(result['Weekend'].unique()).issubset({0, 1})


class TestAddFourierFeatures:
    
    def test_creates_default_periods(self, sample_daily_data):
        result = add_fourier_features(sample_daily_data)
        
        # Check for annual features (3 harmonics)
        assert 'annual_sin_1' in result.columns
        assert 'annual_cos_1' in result.columns
        assert 'annual_sin_2' in result.columns
        assert 'annual_cos_2' in result.columns
        assert 'annual_sin_3' in result.columns
        
        # Check for weekly features (2 harmonics)
        assert 'weekly_sin_1' in result.columns
        assert 'weekly_cos_1' in result.columns
        assert 'weekly_sin_2' in result.columns
    
    def test_fourier_value_range(self, sample_daily_data):
        result = add_fourier_features(sample_daily_data)
        
        fourier_cols = [col for col in result.columns if 'sin_' in col or 'cos_' in col]
        for col in fourier_cols:
            assert result[col].between(-1, 1).all(), f"{col} not in [-1, 1] range"
    
    def test_custom_periods(self, sample_daily_data):
        result = add_fourier_features(
            sample_daily_data,
            periods=[7.0],
            n_harmonics={7.0: 1}
        )
        
        # Should only have weekly features with 1 harmonic
        assert 'weekly_sin_1' in result.columns
        assert 'weekly_cos_1' in result.columns
        assert 'weekly_sin_2' not in result.columns


class TestAddLagFeatures:
    
    def test_creates_lag_columns(self, sample_daily_data):
        result = add_lag_features(sample_daily_data, lags=[1, 7])
        
        assert 'patients_lag_1' in result.columns
        assert 'patients_lag_7' in result.columns
    
    def test_lag_values(self):
        df = pd.DataFrame({
            'Date': pd.date_range('2022-01-01', periods=5),
            'Patients_per_day': [10, 20, 30, 40, 50]
        })
        
        result = add_lag_features(df, lags=[1, 2])
        
        # lag_1 should be previous day's value
        assert pd.isna(result['patients_lag_1'].iloc[0])  # First row is NaN
        assert result['patients_lag_1'].iloc[1] == 10
        assert result['patients_lag_1'].iloc[2] == 20
        
        # lag_2 should be 2 days ago
        assert pd.isna(result['patients_lag_2'].iloc[0])
        assert pd.isna(result['patients_lag_2'].iloc[1])
        assert result['patients_lag_2'].iloc[2] == 10
    
    def test_lag_creates_nans(self, sample_daily_data):
        result = add_lag_features(sample_daily_data, lags=[7])
        
        # First 7 rows should have NaN for lag_7
        assert result['patients_lag_7'].iloc[:7].isnull().all()
        # 8th row onwards should have values
        assert result['patients_lag_7'].iloc[7:].notnull().all()


class TestAddRollingFeatures:
    
    def test_creates_rolling_columns(self, sample_daily_data):
        result = add_rolling_features(sample_daily_data, windows=[7, 14])
        
        assert 'patients_7d_avg' in result.columns
        assert 'patients_7d_std' in result.columns
        assert 'patients_14d_avg' in result.columns
        assert 'patients_14d_std' in result.columns
    
    def test_rolling_mean_values(self):
        df = pd.DataFrame({
            'Date': pd.date_range('2022-01-01', periods=10),
            'Patients_per_day': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
        })
        
        result = add_rolling_features(df, windows=[3])
        
        # 3-day rolling mean (shifted by 1 to prevent leakage)
        # Row 3 should be mean of rows 0-2 (10, 20, 30) = 20
        assert result['patients_3d_avg'].iloc[3] == 20.0
    
    def test_rolling_prevents_leakage(self):
        df = pd.DataFrame({
            'Date': pd.date_range('2022-01-01', periods=5),
            'Patients_per_day': [10, 20, 30, 40, 50]
        })
        
        result = add_rolling_features(df, windows=[2])
        
        # At index 2, should use only data from index 0-1 (due to shift)
        # Not including index 2 (current day)
        assert result['patients_2d_avg'].iloc[2] == 15.0  # (10 + 20) / 2


class TestAddChangeFeatures:
    
    def test_creates_change_columns(self):
        df = pd.DataFrame({
            'Date': pd.date_range('2022-01-01', periods=20),
            'Patients_per_day': range(20)
        })
        
        # First add lags (required for change features)
        df = add_lag_features(df, lags=[1, 2, 7, 14])
        result = add_change_features(df)
        
        assert 'patients_change_1d' in result.columns
        assert 'patients_change_7d' in result.columns
    
    def test_change_values(self):
        df = pd.DataFrame({
            'Date': pd.date_range('2022-01-01', periods=20),
            'Patients_per_day': [i * 10 for i in range(20)]
        })
        
        df = add_lag_features(df, lags=[1, 2, 7, 14])
        result = add_change_features(df)
        
        # 1-day change at index 2 should be lag_1 - lag_2
        # lag_1[2] = 10, lag_2[2] = 0, so change = 10
        assert result['patients_change_1d'].iloc[2] == 10


class TestEngineerFeatures:
    
    def test_complete_pipeline(self, sample_daily_data):
        result = engineer_features(sample_daily_data)
        
        # Should have original columns plus many new features
        assert len(result.columns) > len(sample_daily_data.columns)
        
        # Check that features from each category exist
        # Note: Fourier features are disabled by default (include_fourier=False)
        assert 'Month_sin' in result.columns  # Date features
        assert 'Weekend' in result.columns  # Weekend indicator
        assert 'patients_lag_1' in result.columns  # Lag features
        assert 'patients_3d_avg' in result.columns  # Rolling features
        assert 'patients_change_1d' in result.columns  # Change features
    
    def test_selective_feature_groups(self, sample_daily_data):
        result = engineer_features(
            sample_daily_data,
            include_date=True,
            include_weekend=False,
            include_fourier=False,
            include_lags=False,
            include_rolling=False,
            include_changes=False
        )
        
        # Should only have date features
        assert 'Month_sin' in result.columns
        assert 'Weekend' not in result.columns
        assert 'annual_sin_1' not in result.columns


class TestRemoveNanRows:
    
    def test_removes_rows_with_nan(self):
        df = pd.DataFrame({
            'A': [1, 2, np.nan, 4, 5],
            'B': [10, 20, 30, np.nan, 50],
            'C': [100, 200, 300, 400, 500]
        })
        
        result = remove_nan_rows(df)
        
        assert len(result) == 3  # Only 3 rows without NaN
        assert result['A'].notnull().all()
        assert result['B'].notnull().all()
    
    def test_no_nans_no_change(self, sample_daily_data):
        result = remove_nan_rows(sample_daily_data)
        
        assert len(result) == len(sample_daily_data)
    
    def test_resets_index(self):
        df = pd.DataFrame({
            'A': [1, np.nan, 3, 4],
            'B': [10, 20, 30, 40]
        })
        
        result = remove_nan_rows(df)
        
        # Index should be 0, 1, 2 (not 0, 2, 3)
        assert result.index.tolist() == [0, 1, 2]


def test_complete_feature_engineering_pipeline(sample_daily_data):
    # Add all features (Fourier disabled by default)
    result = engineer_features(sample_daily_data)
    
    # Remove NaN rows
    result = remove_nan_rows(result)
    
    # Should have many features and fewer rows (due to lags)
    # Without Fourier features, expect ~30 features (not 50)
    assert len(result.columns) > 25  # Expect ~30 features without Fourier
    assert len(result) < len(sample_daily_data)  # Rows removed due to lags
    assert not result.isnull().any().any()  # No NaN values
