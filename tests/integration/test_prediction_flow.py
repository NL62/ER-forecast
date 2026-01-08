"""
Integration tests for prediction flow.

Tests the complete end-to-end prediction pipeline with sample data.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch, Mock

from flows.prediction_flow import prediction_flow


@pytest.fixture
def sample_patient_data(tmp_path):
    """Create sample patient visit CSV for testing."""
    np.random.seed(42)
    n_visits = 2000
    
    # Create dates over 1 year
    dates = pd.date_range('2024-01-01', '2025-01-01', freq='H')
    visit_dates = np.random.choice(dates, size=n_visits, replace=True)
    
    df = pd.DataFrame({
        'surrogate_key': range(n_visits),
        'contact_start': visit_dates
    })
    
    # Save to CSV
    csv_path = tmp_path / "test_patient_data.csv"
    
    with open(csv_path, 'w') as f:
        for _, row in df.iterrows():
            f.write(f"{row['surrogate_key']};{row['contact_start']}\n")
    
    return str(csv_path)


@pytest.fixture
def mock_weather_api():
    """Mock weather API responses."""
    with patch('src.data.weather_integration.requests.get') as mock_get:
        mock_response = Mock()
        mock_response.status_code = 200
        
        dates = pd.date_range('2024-01-01', '2025-01-01', freq='D')
        
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


@pytest.fixture
def mock_production_models():
    """Mock production models from MLflow."""
    with patch('src.models.predict.get_latest_production_model') as mock_get:
        # Create mock models that return predictions
        mock_models = {}
        
        for horizon in range(1, 8):
            mock_model = Mock()
            mock_model.predict.return_value = pd.DataFrame({
                'point_prediction': [60.0 + horizon],
                'lower_bound': [50.0 + horizon],
                'upper_bound': [70.0 + horizon]
            })
            mock_models[horizon] = mock_model
        
        mock_get.side_effect = lambda name: mock_models[int(name.split('_')[-1])]
        
        yield mock_get


@pytest.mark.integration
class TestPredictionFlowIntegration:
    """Integration tests for complete prediction flow."""
    
    def test_prediction_flow_runs_successfully(
        self,
        sample_patient_data,
        mock_weather_api,
        mock_production_models,
        tmp_path
    ):
        """Test that prediction flow completes without errors."""
        
        output_path = tmp_path / "predictions"
        output_path.mkdir()
        
        results = prediction_flow(
            raw_data_path=sample_patient_data,
            output_path=str(output_path),
            save_to_database=False
        )
        
        # Assertions
        assert results is not None
        assert 'n_predictions' in results
        assert results['n_predictions'] == 7  # 7-day forecast
        assert 'csv_path' in results
    
    def test_prediction_creates_csv_file(
        self,
        sample_patient_data,
        mock_weather_api,
        mock_production_models,
        tmp_path
    ):
        """Test that prediction flow creates CSV file."""
        
        output_path = tmp_path / "predictions"
        output_path.mkdir()
        
        results = prediction_flow(
            raw_data_path=sample_patient_data,
            output_path=str(output_path),
            save_to_database=False
        )
        
        # Check that CSV file was created
        csv_path = Path(results['csv_path'])
        assert csv_path.exists()
        
        # Read and validate CSV
        df = pd.read_csv(csv_path)
        assert len(df) == 7
        assert 'prediction_date' in df.columns
        assert 'point_prediction' in df.columns
    
    @patch('flows.prediction_flow.log_prediction_metadata_to_mlflow')
    def test_prediction_logs_to_mlflow(
        self,
        mock_log_mlflow,
        sample_patient_data,
        mock_weather_api,
        mock_production_models,
        tmp_path
    ):
        """Test that prediction flow logs to MLflow."""
        
        mock_log_mlflow.return_value = "test_run_456"
        
        output_path = tmp_path / "predictions"
        output_path.mkdir()
        
        results = prediction_flow(
            raw_data_path=sample_patient_data,
            output_path=str(output_path)
        )
        
        # Check that MLflow logging was called
        assert mock_log_mlflow.called
        assert results['mlflow_run_id'] == "test_run_456"
    
    def test_prediction_with_specific_date(
        self,
        sample_patient_data,
        mock_weather_api,
        mock_production_models,
        tmp_path
    ):
        """Test prediction flow with specific base date."""
        
        output_path = tmp_path / "predictions"
        output_path.mkdir()
        
        test_date = date(2025, 9, 30)
        
        results = prediction_flow(
            raw_data_path=sample_patient_data,
            output_path=str(output_path),
            prediction_date=test_date
        )
        
        # Read predictions
        df = pd.read_csv(results['csv_path'])
        
        # Check that prediction dates start from test_date + 1
        first_pred_date = pd.to_datetime(df['prediction_date'].iloc[0]).date()
        assert first_pred_date == date(2025, 10, 1)  # Day after base date
    
    def test_prediction_summary_is_created(
        self,
        sample_patient_data,
        mock_weather_api,
        mock_production_models,
        tmp_path
    ):
        """Test that prediction summary is generated."""
        
        output_path = tmp_path / "predictions"
        output_path.mkdir()
        
        results = prediction_flow(
            raw_data_path=sample_patient_data,
            output_path=str(output_path)
        )
        
        # Check summary
        assert 'summary' in results
        assert 'mean_prediction' in results['summary']
        assert 'n_predictions' in results['summary']


@pytest.mark.integration
def test_prediction_with_missing_models_fails(
    sample_patient_data,
    mock_weather_api,
    tmp_path
):
    """Test that prediction fails if production models are not available."""
    
    with patch('src.models.predict.get_latest_production_model') as mock_get:
        # Mock model loading to fail
        mock_get.side_effect = Exception("No production model found")
        
        output_path = tmp_path / "predictions"
        output_path.mkdir()
        
        with pytest.raises(Exception, match="Cannot load production model"):
            prediction_flow(
                raw_data_path=sample_patient_data,
                output_path=str(output_path)
            )
