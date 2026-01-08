"""
Unit tests for model evaluation module.

Tests cover:
- MAE calculation
- RMSE calculation
- Coverage calculation
- Interval width calculation
- Comprehensive model evaluation
"""

import numpy as np
import pandas as pd
import pytest

from src.models.evaluation import (
    calculate_mae,
    calculate_rmse,
    calculate_coverage,
    calculate_interval_width,
    calculate_mape,
    evaluate_model,
)


class TestCalculateMAE:
    """Tests for calculate_mae function."""
    
    def test_perfect_predictions(self):
        """Test MAE with perfect predictions."""
        y_true = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([10, 20, 30, 40, 50])
        
        mae = calculate_mae(y_true, y_pred)
        
        assert mae == 0.0
    
    def test_constant_error(self):
        """Test MAE with constant error."""
        y_true = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([12, 22, 32, 42, 52])  # Always +2
        
        mae = calculate_mae(y_true, y_pred)
        
        assert mae == 2.0
    
    def test_mixed_errors(self):
        """Test MAE with mixed positive and negative errors."""
        y_true = np.array([10, 20, 30, 40])
        y_pred = np.array([8, 22, 28, 42])  # Errors: -2, +2, -2, +2
        
        mae = calculate_mae(y_true, y_pred)
        
        assert mae == 2.0
    
    def test_mae_returns_float(self):
        """Test that MAE returns float type."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1.5, 2.5, 3.5])
        
        mae = calculate_mae(y_true, y_pred)
        
        assert isinstance(mae, float)


class TestCalculateRMSE:
    """Tests for calculate_rmse function."""
    
    def test_perfect_predictions(self):
        """Test RMSE with perfect predictions."""
        y_true = np.array([10, 20, 30, 40, 50])
        y_pred = np.array([10, 20, 30, 40, 50])
        
        rmse = calculate_rmse(y_true, y_pred)
        
        assert rmse == 0.0
    
    def test_constant_error(self):
        """Test RMSE with constant error."""
        y_true = np.array([10, 20, 30, 40])
        y_pred = np.array([13, 23, 33, 43])  # Always +3
        
        rmse = calculate_rmse(y_true, y_pred)
        
        assert rmse == 3.0
    
    def test_rmse_penalizes_large_errors(self):
        """Test that RMSE penalizes large errors more than small ones."""
        y_true = np.array([10, 10])
        
        # Case 1: Two small errors (1, 1)
        y_pred_small = np.array([11, 11])
        rmse_small = calculate_rmse(y_true, y_pred_small)
        
        # Case 2: One large error (2, 0)
        y_pred_large = np.array([12, 10])
        rmse_large = calculate_rmse(y_true, y_pred_large)
        
        # RMSE should be larger for case with one large error
        # even though both have same total error
        assert rmse_large > rmse_small
    
    def test_rmse_returns_float(self):
        """Test that RMSE returns float type."""
        y_true = np.array([1, 2, 3])
        y_pred = np.array([1.5, 2.5, 3.5])
        
        rmse = calculate_rmse(y_true, y_pred)
        
        assert isinstance(rmse, float)


class TestCalculateCoverage:
    """Tests for calculate_coverage function."""
    
    def test_perfect_coverage(self):
        """Test with all actuals within intervals."""
        y_true = np.array([10, 20, 30, 40, 50])
        y_lower = np.array([8, 18, 28, 38, 48])
        y_upper = np.array([12, 22, 32, 42, 52])
        
        coverage = calculate_coverage(y_true, y_lower, y_upper)
        
        assert coverage == 1.0
    
    def test_no_coverage(self):
        """Test with no actuals within intervals."""
        y_true = np.array([10, 20, 30, 40])
        y_lower = np.array([20, 30, 40, 50])  # All lower bounds above actuals
        y_upper = np.array([25, 35, 45, 55])
        
        coverage = calculate_coverage(y_true, y_lower, y_upper)
        
        assert coverage == 0.0
    
    def test_partial_coverage(self):
        """Test with some actuals within intervals."""
        y_true = np.array([10, 20, 30, 40])
        y_lower = np.array([8, 18, 40, 38])  # First two cover, last two don't
        y_upper = np.array([12, 22, 45, 39])
        
        coverage = calculate_coverage(y_true, y_lower, y_upper)
        
        assert coverage == 0.5  # 2 out of 4
    
    def test_coverage_at_bounds(self):
        """Test that values exactly at bounds are considered covered."""
        y_true = np.array([10, 20, 30])
        y_lower = np.array([10, 15, 25])  # First value at lower bound
        y_upper = np.array([15, 20, 35])  # Second value at upper bound
        
        coverage = calculate_coverage(y_true, y_lower, y_upper)
        
        assert coverage == 1.0  # All should be covered
    
    def test_coverage_returns_float(self):
        """Test that coverage returns float type."""
        y_true = np.array([10, 20, 30])
        y_lower = np.array([8, 18, 28])
        y_upper = np.array([12, 22, 32])
        
        coverage = calculate_coverage(y_true, y_lower, y_upper)
        
        assert isinstance(coverage, float)


class TestCalculateIntervalWidth:
    """Tests for calculate_interval_width function."""
    
    def test_constant_width(self):
        """Test with constant interval widths."""
        y_lower = np.array([10, 20, 30, 40])
        y_upper = np.array([20, 30, 40, 50])  # Always width of 10
        
        width = calculate_interval_width(y_lower, y_upper)
        
        assert width == 10.0
    
    def test_variable_widths(self):
        """Test with variable interval widths."""
        y_lower = np.array([10, 20, 30])
        y_upper = np.array([15, 30, 50])  # Widths: 5, 10, 20
        
        width = calculate_interval_width(y_lower, y_upper)
        
        expected_mean = (5 + 10 + 20) / 3
        assert width == expected_mean
    
    def test_zero_width(self):
        """Test with zero-width intervals."""
        y_lower = np.array([10, 20, 30])
        y_upper = np.array([10, 20, 30])  # Same as lower
        
        width = calculate_interval_width(y_lower, y_upper)
        
        assert width == 0.0
    
    def test_interval_width_returns_float(self):
        """Test that interval width returns float type."""
        y_lower = np.array([10, 20, 30])
        y_upper = np.array([15, 25, 35])
        
        width = calculate_interval_width(y_lower, y_upper)
        
        assert isinstance(width, float)


class TestCalculateMAPE:
    """Tests for calculate_mape function."""
    
    def test_perfect_predictions(self):
        """Test MAPE with perfect predictions."""
        y_true = np.array([10, 20, 30, 40])
        y_pred = np.array([10, 20, 30, 40])
        
        mape = calculate_mape(y_true, y_pred)
        
        assert mape < 1.0  # Should be very close to 0
    
    def test_constant_percentage_error(self):
        """Test MAPE with constant percentage error."""
        y_true = np.array([100, 200])
        y_pred = np.array([110, 220])  # 10% over-prediction
        
        mape = calculate_mape(y_true, y_pred)
        
        # Should be around 10% (with epsilon adjustment)
        assert 9.0 < mape < 11.0
    
    def test_mape_with_epsilon(self):
        """Test that epsilon prevents division by zero."""
        y_true = np.array([0, 0, 0])
        y_pred = np.array([1, 1, 1])
        
        # Should not raise division by zero error
        mape = calculate_mape(y_true, y_pred, epsilon=1.0)
        
        assert mape == 100.0  # All predictions are 100% of (true + epsilon)


class TestEvaluateModel:
    """Tests for evaluate_model function."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model that makes perfect predictions."""
        class MockModel:
            def predict(self, X, return_intervals=False):
                # Return predictions equal to index (for simplicity)
                n = len(X)
                results = pd.DataFrame({
                    'point_prediction': np.arange(n)
                })
                
                if return_intervals:
                    results['lower_bound'] = np.arange(n) - 2
                    results['upper_bound'] = np.arange(n) + 2
                
                return results
        
        return MockModel()
    
    def test_evaluate_returns_dict(self, mock_model):
        """Test that evaluate_model returns a dictionary."""
        X_test = pd.DataFrame({'feature': range(10)})
        y_test = pd.Series(range(10))
        
        metrics = evaluate_model(mock_model, X_test, y_test)
        
        assert isinstance(metrics, dict)
    
    def test_evaluate_contains_required_metrics(self, mock_model):
        """Test that result contains all required metrics."""
        X_test = pd.DataFrame({'feature': range(10)})
        y_test = pd.Series(range(10))
        
        metrics = evaluate_model(mock_model, X_test, y_test)
        
        assert 'mae' in metrics
        assert 'rmse' in metrics
        assert 'n_samples' in metrics
    
    def test_evaluate_with_intervals(self, mock_model):
        """Test evaluation with confidence intervals."""
        X_test = pd.DataFrame({'feature': range(10)})
        y_test = pd.Series(range(10))
        
        metrics = evaluate_model(mock_model, X_test, y_test)
        
        # Should include interval metrics
        assert 'coverage' in metrics
        assert 'interval_width' in metrics
    
    def test_evaluate_with_return_predictions(self, mock_model):
        """Test that predictions can be returned."""
        X_test = pd.DataFrame({'feature': range(10)})
        y_test = pd.Series(range(10))
        
        metrics = evaluate_model(mock_model, X_test, y_test, return_predictions=True)
        
        assert 'predictions' in metrics
        assert isinstance(metrics['predictions'], pd.DataFrame)
        assert 'actual' in metrics['predictions'].columns
        assert 'error' in metrics['predictions'].columns
        assert 'abs_error' in metrics['predictions'].columns
    
    def test_evaluate_n_samples_correct(self, mock_model):
        """Test that n_samples is correctly reported."""
        X_test = pd.DataFrame({'feature': range(25)})
        y_test = pd.Series(range(25))
        
        metrics = evaluate_model(mock_model, X_test, y_test)
        
        assert metrics['n_samples'] == 25


def test_metrics_with_known_values():
    """Integration test with known values."""
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 32.0, 38.0])
    y_lower = np.array([8.0, 14.0, 28.0, 34.0])
    y_upper = np.array([16.0, 22.0, 36.0, 42.0])
    
    # Calculate metrics
    mae = calculate_mae(y_true, y_pred)
    rmse = calculate_rmse(y_true, y_pred)
    coverage = calculate_coverage(y_true, y_lower, y_upper)
    width = calculate_interval_width(y_lower, y_upper)
    
    # Verify expected values
    assert mae == 2.0  # Average absolute error is 2
    assert rmse == 2.0  # RMS error is also 2 (constant errors)
    assert coverage == 1.0  # All within intervals
    assert width == 8.0  # All intervals have width 8
