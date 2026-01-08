"""
Unit tests for LightGBM model wrapper.

Tests cover:
- Model initialization
- Point model training
- Quantile model training
- Prediction generation
- Model persistence (save/load)
- Feature importance
"""

import numpy as np
import pandas as pd
import pytest
import tempfile
from pathlib import Path

from src.models.lightgbm_model import LightGBMForecaster


@pytest.fixture
def sample_train_data():
    """Generate sample training data."""
    np.random.seed(42)
    n_samples = 200
    
    X = pd.DataFrame({
        'feature_1': np.random.randn(n_samples),
        'feature_2': np.random.randn(n_samples),
        'feature_3': np.random.rand(n_samples) * 100,
        'feature_4': np.random.randint(0, 10, n_samples),
    })
    
    # Create target with some relationship to features
    y = pd.Series(
        50 + 
        2 * X['feature_1'] + 
        -1.5 * X['feature_2'] + 
        0.3 * X['feature_3'] +
        np.random.randn(n_samples) * 5
    )
    
    return X, y


class TestLightGBMForecaster:
    """Tests for LightGBMForecaster class."""
    
    def test_initialization(self):
        """Test model initialization."""
        forecaster = LightGBMForecaster(horizon=3)
        
        assert forecaster.horizon == 3
        assert forecaster.point_model is None
        assert forecaster.lower_model is None
        assert forecaster.upper_model is None
        assert forecaster.feature_names is None
    
    def test_train_point_model(self, sample_train_data):
        """Test training point prediction model."""
        X, y = sample_train_data
        forecaster = LightGBMForecaster(horizon=1)
        
        params = {
            'n_estimators': 10,
            'max_depth': 3,
            'learning_rate': 0.1
        }
        
        forecaster.train_point_model(X, y, params)
        
        assert forecaster.point_model is not None
        assert forecaster.feature_names == X.columns.tolist()
    
    def test_train_with_default_params(self, sample_train_data):
        """Test training with default parameters."""
        X, y = sample_train_data
        forecaster = LightGBMForecaster()
        
        forecaster.train_point_model(X, y)  # Uses defaults
        
        assert forecaster.point_model is not None
    
    def test_train_quantile_models(self, sample_train_data):
        """Test training quantile regression models."""
        X, y = sample_train_data
        forecaster = LightGBMForecaster()
        
        params = {'n_estimators': 10, 'max_depth': 3}
        
        forecaster.train_quantile_models(X, y, params)
        
        assert forecaster.lower_model is not None
        assert forecaster.upper_model is not None
    
    def test_predict_point_only(self, sample_train_data):
        """Test prediction without confidence intervals."""
        X, y = sample_train_data
        forecaster = LightGBMForecaster()
        
        # Train
        forecaster.train_point_model(X[:150], y[:150])
        
        # Predict
        X_test = X[150:]
        predictions = forecaster.predict(X_test, return_intervals=False)
        
        assert 'point_prediction' in predictions.columns
        assert len(predictions) == len(X_test)
        assert 'lower_bound' not in predictions.columns
        assert 'upper_bound' not in predictions.columns
    
    def test_predict_with_intervals(self, sample_train_data):
        """Test prediction with confidence intervals."""
        X, y = sample_train_data
        forecaster = LightGBMForecaster()
        
        # Train both point and quantile models
        params = {'n_estimators': 10, 'max_depth': 3}
        forecaster.train_point_model(X[:150], y[:150], params)
        forecaster.train_quantile_models(X[:150], y[:150], params)
        
        # Predict with intervals
        X_test = X[150:]
        predictions = forecaster.predict(X_test, return_intervals=True)
        
        assert 'point_prediction' in predictions.columns
        assert 'lower_bound' in predictions.columns
        assert 'upper_bound' in predictions.columns
        assert len(predictions) == len(X_test)
        
        # Check that bounds make sense
        assert (predictions['lower_bound'] <= predictions['point_prediction']).all()
        assert (predictions['point_prediction'] <= predictions['upper_bound']).all()
    
    def test_predict_without_training_raises_error(self, sample_train_data):
        """Test that predicting without training raises error."""
        X, y = sample_train_data
        forecaster = LightGBMForecaster()
        
        with pytest.raises(ValueError, match="Point model not trained"):
            forecaster.predict(X)
    
    def test_save_and_load_model(self, sample_train_data, tmp_path):
        """Test model persistence."""
        X, y = sample_train_data
        
        # Train a model
        forecaster = LightGBMForecaster(horizon=5)
        params = {'n_estimators': 10, 'max_depth': 3}
        forecaster.train_point_model(X[:150], y[:150], params)
        forecaster.train_quantile_models(X[:150], y[:150], params)
        
        # Save model
        model_path = tmp_path / "test_model.pkl"
        forecaster.save_model(str(model_path))
        
        assert model_path.exists()
        
        # Load model
        loaded_forecaster = LightGBMForecaster.load_model(str(model_path))
        
        assert loaded_forecaster.horizon == 5
        assert loaded_forecaster.point_model is not None
        assert loaded_forecaster.lower_model is not None
        assert loaded_forecaster.upper_model is not None
        assert loaded_forecaster.feature_names == X.columns.tolist()
        
        # Test that loaded model makes same predictions
        X_test = X[150:]
        original_preds = forecaster.predict(X_test, return_intervals=True)
        loaded_preds = loaded_forecaster.predict(X_test, return_intervals=True)
        
        np.testing.assert_array_almost_equal(
            original_preds['point_prediction'].values,
            loaded_preds['point_prediction'].values
        )
    
    def test_load_nonexistent_model_raises_error(self):
        """Test that loading nonexistent model raises error."""
        with pytest.raises(FileNotFoundError):
            LightGBMForecaster.load_model("nonexistent_model.pkl")
    
    def test_get_feature_importance(self, sample_train_data):
        """Test feature importance extraction."""
        X, y = sample_train_data
        forecaster = LightGBMForecaster()
        
        forecaster.train_point_model(X, y)
        
        importance = forecaster.get_feature_importance(top_n=3)
        
        assert len(importance) == 3
        assert 'feature' in importance.columns
        assert 'importance' in importance.columns
        assert importance['importance'].iloc[0] >= importance['importance'].iloc[1]  # Sorted
    
    def test_get_feature_importance_without_training_raises_error(self):
        """Test that getting importance without training raises error."""
        forecaster = LightGBMForecaster()
        
        with pytest.raises(ValueError, match="Point model not trained"):
            forecaster.get_feature_importance()
    
    def test_repr(self):
        """Test string representation."""
        forecaster = LightGBMForecaster(horizon=2)
        repr_str = repr(forecaster)
        
        assert 'horizon=2' in repr_str
        assert 'none' in repr_str  # No models trained yet
    
    def test_repr_with_trained_models(self, sample_train_data):
        """Test string representation with trained models."""
        X, y = sample_train_data
        forecaster = LightGBMForecaster(horizon=4)
        
        forecaster.train_point_model(X, y)
        repr_str = repr(forecaster)
        
        assert 'horizon=4' in repr_str
        assert 'point' in repr_str
    
    def test_quantile_models_coverage(self, sample_train_data):
        """Test that quantile models produce reasonable coverage."""
        X, y = sample_train_data
        forecaster = LightGBMForecaster()
        
        # Train on more data for better convergence
        params = {'n_estimators': 50, 'max_depth': 5}
        forecaster.train_point_model(X[:150], y[:150], params)
        forecaster.train_quantile_models(X[:150], y[:150], params, quantiles=(0.025, 0.975))
        
        # Predict on test set
        X_test = X[150:]
        y_test = y[150:].reset_index(drop=True)
        predictions = forecaster.predict(X_test, return_intervals=True)
        
        # Check coverage (should be around 95%)
        within_interval = (
            (y_test >= predictions['lower_bound']) & 
            (y_test <= predictions['upper_bound'])
        )
        coverage = within_interval.mean()
        
        # Coverage should be reasonably close to 95% (allow some deviation)
        assert 0.70 <= coverage <= 1.0, f"Coverage {coverage:.2%} is outside reasonable range"
