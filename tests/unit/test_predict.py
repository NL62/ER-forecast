"""
Unit tests for batch prediction module.

Tests cover:
- Loading production models
- Preparing prediction features
- Generating predictions
- Validating predictions
"""

import pandas as pd
import pytest
from datetime import date, datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from src.models.predict import (
    load_production_models,
    prepare_prediction_features,
    generate_predictions,
    validate_predictions,
    get_model_version_from_mlflow,
)


class TestLoadProductionModels:
    """Tests for load_production_models function."""
    
    @patch('src.models.predict.get_latest_production_model')
    def test_load_all_horizons(self, mock_get_model):
        """Test loading models for all 7 horizons."""
        # Mock model loading
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        
        models = load_production_models()
        
        # Should load 7 models
        assert len(models) == 7
        assert all(h in models for h in range(1, 8))
        
        # Should have called get_latest_production_model 7 times
        assert mock_get_model.call_count == 7
    
    @patch('src.models.predict.get_latest_production_model')
    def test_load_with_custom_pattern(self, mock_get_model):
        """Test loading with custom model name pattern."""
        mock_model = Mock()
        mock_get_model.return_value = mock_model
        
        models = load_production_models(
            model_name_pattern="custom_model_h{horizon}"
        )
        
        # Check that custom pattern was used
        first_call_args = mock_get_model.call_args_list[0][0]
        assert "custom_model_h1" == first_call_args[0]
    
    @patch('src.models.predict.get_latest_production_model')
    @patch('src.models.predict.Path')
    def test_load_fails_when_no_models_available(self, mock_path, mock_get_model):
        """Test that loading fails when neither MLflow nor local files are available."""
        # MLflow fails for all horizons
        mock_get_model.side_effect = Exception("Model not found")
        
        # Mock Path.exists() to return False (no local files)
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = False
        mock_path.return_value = mock_path_instance
        
        # Should raise when both MLflow and local files fail
        with pytest.raises(Exception, match="Cannot load production model"):
            load_production_models()


class TestPreparePredictionFeatures:
    """Tests for prepare_prediction_features function."""
    
    @pytest.fixture
    def sample_historical_data(self):
        """Create sample historical data with features."""
        dates = pd.date_range('2025-09-01', periods=30, freq='D')
        df = pd.DataFrame({
            'Date': dates,
            'Patients_per_day': [50 + i for i in range(30)],
            'patients_lag_1': [49 + i for i in range(30)],
            'patients_lag_7': [43 + i for i in range(30)],
            'patients_3d_avg': [50.0] * 30,
            'Month_sin': [0.5] * 30,
            'Month_cos': [0.5] * 30,
            'Weekend': [0, 0, 0, 0, 0, 1, 1] * 4 + [0, 0],
        })
        return df
    
    def test_creates_7_prediction_rows(self, sample_historical_data):
        """Test that 7 prediction rows are created."""
        features = prepare_prediction_features(sample_historical_data)
        
        assert len(features) == 7
    
    def test_prediction_dates_are_consecutive(self, sample_historical_data):
        """Test that prediction dates are consecutive days."""
        base = date(2025, 9, 30)
        features = prepare_prediction_features(sample_historical_data, base_date=base)
        
        dates = pd.to_datetime(features['Date']).dt.date.tolist()
        
        # Should be Oct 1-7
        expected_dates = [base + timedelta(days=i) for i in range(1, 8)]
        assert dates == expected_dates
    
    def test_uses_last_date_as_base_by_default(self, sample_historical_data):
        """Test that last date in data is used as base if not specified."""
        features = prepare_prediction_features(sample_historical_data)
        
        last_historical_date = sample_historical_data['Date'].max().date()
        first_prediction_date = pd.to_datetime(features['Date'].iloc[0]).date()
        
        assert first_prediction_date == last_historical_date + timedelta(days=1)
    
    def test_preserves_lag_features(self, sample_historical_data):
        """Test that lag features are preserved for prediction."""
        features = prepare_prediction_features(sample_historical_data)
        
        # Lag features should exist
        assert 'patients_lag_1' in features.columns
        assert 'patients_lag_7' in features.columns


class TestGeneratePredictions:
    """Tests for generate_predictions function."""
    
    @pytest.fixture
    def mock_models(self):
        """Create mock models for all horizons."""
        models = {}
        
        for horizon in range(1, 8):
            mock_model = Mock()
            # Mock predict method to return DataFrame
            mock_model.predict.return_value = pd.DataFrame({
                'point_prediction': [60.0 + horizon],
                'lower_bound': [50.0 + horizon],
                'upper_bound': [70.0 + horizon]
            })
            models[horizon] = mock_model
        
        return models
    
    @pytest.fixture
    def sample_features(self):
        """Create sample features for prediction."""
        return pd.DataFrame({
            'Date': pd.date_range('2025-10-01', periods=7),
            'patients_lag_1': [50] * 7,
            'patients_lag_7': [48] * 7,
            'feature_1': [1.0] * 7,
            'feature_2': [2.0] * 7,
        })
    
    def test_generates_7_predictions(self, mock_models, sample_features):
        """Test that 7 predictions are generated."""
        predictions = generate_predictions(mock_models, sample_features)
        
        assert len(predictions) == 7
    
    def test_predictions_have_required_columns(self, mock_models, sample_features):
        """Test that all required columns are present."""
        predictions = generate_predictions(mock_models, sample_features)
        
        required_cols = [
            'prediction_timestamp', 'prediction_date', 'horizon',
            'model_version', 'point_prediction', 'lower_bound', 'upper_bound'
        ]
        
        for col in required_cols:
            assert col in predictions.columns
    
    def test_horizons_are_correct(self, mock_models, sample_features):
        """Test that horizon values are 1-7."""
        predictions = generate_predictions(mock_models, sample_features)
        
        assert predictions['horizon'].tolist() == [1, 2, 3, 4, 5, 6, 7]
    
    def test_prediction_dates_are_sequential(self, mock_models, sample_features):
        """Test that prediction dates are consecutive."""
        base = date(2025, 9, 30)
        predictions = generate_predictions(mock_models, sample_features, base_date=base)
        
        pred_dates = pd.to_datetime(predictions['prediction_date']).dt.date.tolist()
        
        for i, pred_date in enumerate(pred_dates, start=1):
            expected_date = base + timedelta(days=i)
            assert pred_date == expected_date
    
    def test_calls_model_predict_for_each_horizon(self, mock_models, sample_features):
        """Test that each model's predict method is called."""
        generate_predictions(mock_models, sample_features)
        
        for horizon, model in mock_models.items():
            model.predict.assert_called_once()


class TestValidatePredictions:
    """Tests for validate_predictions function."""
    
    def test_valid_predictions_pass(self):
        """Test that valid predictions pass validation."""
        predictions = pd.DataFrame({
            'prediction_timestamp': [datetime(2025, 9, 30, 2, 0)] * 3,
            'prediction_date': [date(2025, 10, 1), date(2025, 10, 2), date(2025, 10, 3)],
            'horizon': [1, 2, 3],
            'model_version': ['5', '5', '5'],
            'point_prediction': [60.0, 62.0, 64.0],
            'lower_bound': [50.0, 52.0, 54.0],
            'upper_bound': [70.0, 72.0, 74.0]
        })
        
        result = validate_predictions(predictions)
        
        assert result is True
    
    def test_missing_column_raises_error(self):
        """Test that missing required column raises error."""
        predictions = pd.DataFrame({
            'prediction_timestamp': [datetime.now()],
            'prediction_date': [date.today()],
            # Missing other columns
        })
        
        with pytest.raises(ValueError, match="Missing required columns"):
            validate_predictions(predictions)
    
    def test_null_values_raise_error(self):
        """Test that NULL values raise error."""
        predictions = pd.DataFrame({
            'prediction_timestamp': [datetime.now()],
            'prediction_date': [None],  # NULL value
            'horizon': [1],
            'model_version': ['5'],
            'point_prediction': [60.0],
            'lower_bound': [50.0],
            'upper_bound': [70.0]
        })
        
        with pytest.raises(ValueError, match="contain missing values"):
            validate_predictions(predictions)
    
    def test_invalid_horizon_raises_error(self):
        """Test that invalid horizon values raise error."""
        predictions = pd.DataFrame({
            'prediction_timestamp': [datetime.now()],
            'prediction_date': [date.today()],
            'horizon': [8],  # Invalid horizon
            'model_version': ['5'],
            'point_prediction': [60.0],
            'lower_bound': [50.0],
            'upper_bound': [70.0]
        })
        
        with pytest.raises(ValueError, match="Invalid horizon values"):
            validate_predictions(predictions)
    
    def test_invalid_bounds_raise_error(self):
        """Test that invalid bounds raise error."""
        predictions = pd.DataFrame({
            'prediction_timestamp': [datetime.now()],
            'prediction_date': [date.today()],
            'horizon': [1],
            'model_version': ['5'],
            'point_prediction': [60.0],
            'lower_bound': [65.0],  # Lower > point (invalid)
            'upper_bound': [70.0]
        })
        
        with pytest.raises(ValueError, match="invalid bounds"):
            validate_predictions(predictions)


class TestGetModelVersionFromMLflow:
    """Tests for get_model_version_from_mlflow function."""
    
    @patch('mlflow.tracking.MlflowClient')
    def test_returns_production_version(self, mock_client_class):
        """Test retrieving production model version."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        # Mock model version
        mock_version = Mock()
        mock_version.version = "5"
        mock_client.get_latest_versions.return_value = [mock_version]
        
        version = get_model_version_from_mlflow("er_forecast_horizon_1")
        
        assert version == "5"
    
    @patch('mlflow.tracking.MlflowClient')
    def test_returns_unknown_if_no_production(self, mock_client_class):
        """Test returns 'unknown' if no production model exists."""
        mock_client = Mock()
        mock_client_class.return_value = mock_client
        
        mock_client.get_latest_versions.return_value = []
        
        version = get_model_version_from_mlflow("er_forecast_horizon_1")
        
        assert version == "unknown"
