"""
Unit tests for model training module.

Tests cover:
- Data splitting
- Shifted target creation
- Training functions
"""

import numpy as np
import pandas as pd
import pytest

from src.models.train import (
    split_data,
    create_shifted_target,
    train_model_for_horizon,
)


@pytest.fixture
def sample_timeseries_data():
    """Generate sample time series data."""
    np.random.seed(42)
    n_days = 100
    
    dates = pd.date_range('2022-01-01', periods=n_days, freq='D')
    
    df = pd.DataFrame({
        'Date': dates,
        'Patients_per_day': np.random.randint(40, 80, n_days),
        'feature_1': np.random.randn(n_days),
        'feature_2': np.random.randn(n_days),
        'feature_3': np.random.rand(n_days) * 100,
    })
    
    return df


class TestSplitData:
    """Tests for split_data function."""
    
    def test_default_split_ratios(self, sample_timeseries_data):
        """Test data splitting with default ratios (80/10/10)."""
        df = sample_timeseries_data
        
        train, val, test = split_data(df)
        
        n = len(df)
        assert len(train) == int(0.80 * n)
        assert len(val) == int(0.10 * n)
        # Test gets the remainder
        assert len(train) + len(val) + len(test) == n
    
    def test_custom_split_ratios(self, sample_timeseries_data):
        """Test data splitting with custom ratios."""
        df = sample_timeseries_data
        
        train, val, test = split_data(
            df,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15
        )
        
        n = len(df)
        assert len(train) == int(0.70 * n)
        assert len(val) == int(0.15 * n)
    
    def test_temporal_order_preserved(self, sample_timeseries_data):
        """Test that temporal order is preserved (no shuffling)."""
        df = sample_timeseries_data
        
        train, val, test = split_data(df)
        
        # Check that dates are consecutive and in order
        assert train['Date'].iloc[-1] < val['Date'].iloc[0]
        assert val['Date'].iloc[-1] < test['Date'].iloc[0]
    
    def test_invalid_ratios_raises_error(self, sample_timeseries_data):
        """Test that invalid ratios raise error."""
        df = sample_timeseries_data
        
        with pytest.raises(ValueError, match="must sum to 1.0"):
            split_data(df, train_ratio=0.5, val_ratio=0.2, test_ratio=0.2)
    
    def test_no_data_loss(self, sample_timeseries_data):
        """Test that all rows are included in splits."""
        df = sample_timeseries_data
        
        train, val, test = split_data(df)
        
        total_rows = len(train) + len(val) + len(test)
        assert total_rows == len(df)


class TestCreateShiftedTarget:
    """Tests for create_shifted_target function."""
    
    def test_horizon_1_shift(self, sample_timeseries_data):
        """Test creating 1-day ahead target."""
        df = sample_timeseries_data
        
        result = create_shifted_target(df, horizon=1)
        
        # Should have target column
        assert 'target' in result.columns
        
        # Should have fewer rows (last row has no future value)
        assert len(result) == len(df) - 1
        
        # Check that target is correctly shifted
        assert result['target'].iloc[0] == df['Patients_per_day'].iloc[1]
        assert result['target'].iloc[1] == df['Patients_per_day'].iloc[2]
    
    def test_horizon_7_shift(self, sample_timeseries_data):
        """Test creating 7-day ahead target."""
        df = sample_timeseries_data
        
        result = create_shifted_target(df, horizon=7)
        
        # Should lose last 7 rows
        assert len(result) == len(df) - 7
        
        # Check shifting is correct
        assert result['target'].iloc[0] == df['Patients_per_day'].iloc[7]
        assert result['target'].iloc[10] == df['Patients_per_day'].iloc[17]
    
    def test_different_horizons_different_lengths(self, sample_timeseries_data):
        """Test that different horizons result in different lengths."""
        df = sample_timeseries_data
        
        result_h1 = create_shifted_target(df, horizon=1)
        result_h3 = create_shifted_target(df, horizon=3)
        result_h7 = create_shifted_target(df, horizon=7)
        
        assert len(result_h1) > len(result_h3) > len(result_h7)
    
    def test_invalid_horizon_raises_error(self, sample_timeseries_data):
        """Test that invalid horizons raise errors."""
        df = sample_timeseries_data
        
        with pytest.raises(ValueError, match="must be between 1 and 7"):
            create_shifted_target(df, horizon=0)
        
        with pytest.raises(ValueError, match="must be between 1 and 7"):
            create_shifted_target(df, horizon=8)
    
    def test_missing_target_column_raises_error(self, sample_timeseries_data):
        """Test that missing target column raises error."""
        df = sample_timeseries_data.drop(columns=['Patients_per_day'])
        
        with pytest.raises(KeyError, match="Target column"):
            create_shifted_target(df, horizon=1)
    
    def test_no_nan_in_result(self, sample_timeseries_data):
        """Test that result has no NaN values."""
        df = sample_timeseries_data
        
        result = create_shifted_target(df, horizon=3)
        
        assert not result['target'].isnull().any()


class TestTrainModelForHorizon:
    """Tests for train_model_for_horizon function."""
    
    def test_train_with_minimal_trials(self, sample_timeseries_data):
        """Test training with minimal Optuna trials (for speed)."""
        df = sample_timeseries_data
        
        # Use very few trials for quick test
        model, metadata = train_model_for_horizon(
            df,
            horizon=1,
            n_trials=2  # Just 2 trials for speed
        )
        
        # Check model is trained
        assert model is not None
        assert model.point_model is not None
        assert model.horizon == 1
        
        # Check metadata
        assert metadata['horizon'] == 1
        assert metadata['n_trials'] == 2
        assert 'best_params' in metadata
        assert 'test_mae' in metadata
        assert 'best_val_mae' in metadata
    
    def test_metadata_contains_expected_fields(self, sample_timeseries_data):
        """Test that metadata contains all expected fields."""
        df = sample_timeseries_data
        
        _, metadata = train_model_for_horizon(df, horizon=2, n_trials=2)
        
        expected_fields = [
            'horizon', 'n_trials', 'best_params', 'best_val_mae',
            'test_mae', 'test_rmse', 'n_train', 'n_val',
            'n_test', 'n_features', 'feature_names'
        ]
        
        for field in expected_fields:
            assert field in metadata, f"Missing field: {field}"
    
    def test_different_horizons_train_different_models(self, sample_timeseries_data):
        """Test that different horizons produce different models."""
        df = sample_timeseries_data
        
        model_h1, meta_h1 = train_model_for_horizon(df, horizon=1, n_trials=2)
        model_h3, meta_h3 = train_model_for_horizon(df, horizon=3, n_trials=2)
        
        assert model_h1.horizon == 1
        assert model_h3.horizon == 3
        
        # Metadata should differ
        assert meta_h1['horizon'] != meta_h3['horizon']
    
    def test_quantile_models_are_trained(self, sample_timeseries_data):
        """Test that quantile models are trained for confidence intervals."""
        df = sample_timeseries_data
        
        model, _ = train_model_for_horizon(df, horizon=1, n_trials=2)
        
        # Check that quantile models exist
        assert model.lower_model is not None
        assert model.upper_model is not None
    
    def test_feature_names_stored(self, sample_timeseries_data):
        """Test that feature names are stored in metadata."""
        df = sample_timeseries_data
        
        _, metadata = train_model_for_horizon(df, horizon=1, n_trials=2)
        
        assert 'feature_names' in metadata
        assert isinstance(metadata['feature_names'], list)
        assert len(metadata['feature_names']) > 0
        
        # Should not include target or date
        assert 'target' not in metadata['feature_names']
        assert 'Date' not in metadata['feature_names']
        assert 'Patients_per_day' not in metadata['feature_names']


def test_integration_split_and_shift(sample_timeseries_data):
    """Integration test: split data then create shifted target."""
    df = sample_timeseries_data
    
    # First create shifted target
    df_shifted = create_shifted_target(df, horizon=3)
    
    # Then split
    train, val, test = split_data(df_shifted)
    
    # Verify shapes make sense
    assert len(train) + len(val) + len(test) == len(df_shifted)
    
    # Verify target column exists in all splits
    for split in [train, val, test]:
        assert 'target' in split.columns
