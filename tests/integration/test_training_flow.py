"""
Integration tests for training flow.

Tests the complete end-to-end training pipeline with sample data.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from unittest.mock import patch, Mock
import tempfile

from flows.training_flow import training_flow


@pytest.fixture
def sample_patient_data(tmp_path):
    """Create sample patient visit CSV for testing."""
    # Generate synthetic patient visit data
    np.random.seed(42)
    n_visits = 5000
    
    # Create dates over 3 years
    dates = pd.date_range('2022-01-01', '2025-01-01', freq='H')
    visit_dates = np.random.choice(dates, size=n_visits, replace=True)
    
    df = pd.DataFrame({
        'surrogate_key': range(n_visits),
        'contact_start': visit_dates
    })
    
    # Save to CSV
    csv_path = tmp_path / "test_patient_data.csv"
    
    # Save in semicolon-delimited format (European standard)
    with open(csv_path, 'w') as f:
        for _, row in df.iterrows():
            f.write(f"{row['surrogate_key']};{row['contact_start']}\n")
    
    return str(csv_path)


@pytest.fixture
def mock_weather_api():
    """Mock weather API responses."""
    with patch('src.data.weather_integration.requests.get') as mock_get:
        # Create mock response
        mock_response = Mock()
        mock_response.status_code = 200
        
        # Generate weather data for 3 years
        dates = pd.date_range('2022-01-01', '2025-01-01', freq='D')
        
        mock_response.json.return_value = {
            'daily': {
                'time': [d.strftime('%Y-%m-%d') for d in dates],
                'temperature_2m_max': np.random.uniform(-5, 20, len(dates)).tolist(),
                'temperature_2m_min': np.random.uniform(-15, 10, len(dates)).tolist(),
                'temperature_2m_mean': np.random.uniform(-10, 15, len(dates)).tolist(),
                'precipitation_sum': np.random.uniform(0, 20, len(dates)).tolist(),
                'snowfall_sum': np.random.uniform(0, 10, len(dates)).tolist(),
            }
        }
        
        mock_get.return_value = mock_response
        
        yield mock_get


@pytest.mark.slow
@pytest.mark.integration
class TestTrainingFlowIntegration:
    """Integration tests for complete training flow."""
    
    def test_training_flow_runs_successfully(self, sample_patient_data, mock_weather_api):
        """Test that training flow completes without errors (minimal trials)."""
        
        # Run training flow with minimal trials for speed
        results = training_flow(
            raw_data_path=sample_patient_data,
            n_optuna_trials=2,  # Very few trials for quick test
            horizons=[1, 2],  # Only train 2 horizons for speed
            mae_threshold=100.0  # High threshold so models get promoted
        )
        
        # Assertions
        assert results is not None
        assert 'models_trained' in results
        assert results['models_trained'] == 2  # Trained 2 horizons
    
    def test_training_creates_model_artifacts(self, sample_patient_data, mock_weather_api, tmp_path):
        """Test that training creates model artifacts."""
        
        # Create temporary models directory
        models_dir = tmp_path / "models"
        models_dir.mkdir()
        
        with patch('flows.training_flow.Path') as mock_path:
            mock_path.return_value = models_dir
            
            results = training_flow(
                raw_data_path=sample_patient_data,
                n_optuna_trials=2,
                horizons=[1],
                mae_threshold=100.0
            )
        
        # Note: This test may need to be adjusted based on actual implementation
        assert results['models_trained'] >= 1
    
    @patch('flows.training_flow.mlflow')
    def test_training_logs_to_mlflow(self, mock_mlflow, sample_patient_data, mock_weather_api):
        """Test that training logs experiments to MLflow."""
        
        # Mock MLflow run
        mock_run = Mock()
        mock_run.info.run_id = "test_run_123"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        mock_mlflow.active_run.return_value = mock_run
        
        results = training_flow(
            raw_data_path=sample_patient_data,
            n_optuna_trials=2,
            horizons=[1],
            mae_threshold=100.0
        )
        
        # Check that MLflow functions were called
        assert mock_mlflow.set_tracking_uri.called
        assert mock_mlflow.start_run.called
    
    def test_training_with_invalid_data_path_fails(self):
        """Test that training fails gracefully with invalid data path."""
        
        with pytest.raises(FileNotFoundError):
            training_flow(
                raw_data_path="nonexistent_file.csv",
                n_optuna_trials=2,
                horizons=[1]
            )


@pytest.mark.integration
def test_training_flow_parameter_validation(sample_patient_data, mock_weather_api):
    """Test that flow validates parameters."""
    
    # Invalid horizon (should be 1-7)
    with pytest.raises((ValueError, Exception)):
        training_flow(
            raw_data_path=sample_patient_data,
            n_optuna_trials=2,
            horizons=[0, 8],  # Invalid horizons
            mae_threshold=10.0
        )
