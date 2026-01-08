"""
Unit tests for prediction output module.

Tests cover:
- Saving predictions to CSV
- Writing predictions to database (stub)
- Logging to MLflow
- Formatting for database
- Creating summaries
"""

import pandas as pd
import pytest
from datetime import datetime, date
from pathlib import Path
from unittest.mock import patch, Mock, MagicMock

from src.models.prediction_output import (
    save_predictions_to_csv,
    write_predictions_to_database,
    log_prediction_metadata_to_mlflow,
    format_predictions_for_database,
    create_prediction_summary,
)


@pytest.fixture
def sample_predictions():
    """Create sample predictions DataFrame."""
    return pd.DataFrame({
        'prediction_timestamp': [datetime(2025, 9, 30, 2, 0)] * 7,
        'prediction_date': [date(2025, 10, i) for i in range(1, 8)],
        'horizon': [1, 2, 3, 4, 5, 6, 7],
        'model_version': ['5'] * 7,
        'point_prediction': [60.0, 62.0, 64.0, 66.0, 68.0, 70.0, 72.0],
        'lower_bound': [50.0, 52.0, 54.0, 56.0, 58.0, 60.0, 62.0],
        'upper_bound': [70.0, 72.0, 74.0, 76.0, 78.0, 80.0, 82.0]
    })


class TestSavePredictionsToCSV:
    """Tests for save_predictions_to_csv function."""
    
    def test_creates_csv_file(self, sample_predictions, tmp_path):
        """Test that CSV file is created."""
        filepath = save_predictions_to_csv(
            sample_predictions,
            output_path=str(tmp_path),
            include_timestamp=False
        )
        
        assert Path(filepath).exists()
        assert Path(filepath).name == "predictions_latest.csv"
    
    def test_csv_contains_correct_data(self, sample_predictions, tmp_path):
        """Test that saved CSV contains correct data."""
        filepath = save_predictions_to_csv(
            sample_predictions,
            output_path=str(tmp_path),
            include_timestamp=False
        )
        
        # Read back the CSV
        loaded = pd.read_csv(filepath)
        
        assert len(loaded) == 7
        assert list(loaded.columns) == list(sample_predictions.columns)
        assert loaded['horizon'].tolist() == [1, 2, 3, 4, 5, 6, 7]
    
    def test_filename_with_timestamp(self, sample_predictions, tmp_path):
        """Test that filename includes timestamp when requested."""
        filepath = save_predictions_to_csv(
            sample_predictions,
            output_path=str(tmp_path),
            include_timestamp=True
        )
        
        filename = Path(filepath).name
        assert filename.startswith("predictions_")
        assert filename.endswith(".csv")
        assert len(filename) > len("predictions_.csv")  # Has timestamp
    
    def test_creates_output_directory(self, sample_predictions, tmp_path):
        """Test that output directory is created if it doesn't exist."""
        new_dir = tmp_path / "new_directory" / "predictions"
        
        filepath = save_predictions_to_csv(
            sample_predictions,
            output_path=str(new_dir)
        )
        
        assert Path(filepath).parent.exists()


def _pyodbc_available():
    """Check if pyodbc is properly installed and usable."""
    try:
        import pyodbc
        return True
    except (ImportError, OSError):
        # ImportError if not installed, OSError if unixodbc lib missing
        return False


class TestWritePredictionsToDatabase:
    """Tests for write_predictions_to_database function."""
    
    @pytest.mark.skipif(not _pyodbc_available(), reason="pyodbc not available or unixodbc not installed")
    def test_returns_row_count(self, sample_predictions):
        """Test that function returns number of rows."""
        n_rows = write_predictions_to_database(
            sample_predictions,
            connection_string="postgresql://user:pwd@localhost/db"
        )
        
        assert n_rows == 7
    
    @pytest.mark.skipif(not _pyodbc_available(), reason="pyodbc not available or unixodbc not installed")
    def test_accepts_table_name(self, sample_predictions):
        """Test that custom table name is accepted."""
        n_rows = write_predictions_to_database(
            sample_predictions,
            connection_string="postgresql://user:pwd@localhost/db",
            table_name="custom_predictions"
        )
        
        assert n_rows == len(sample_predictions)


class TestLogPredictionMetadataToMLflow:
    """Tests for log_prediction_metadata_to_mlflow function."""
    
    @patch('src.models.prediction_output.mlflow')
    def test_creates_mlflow_run(self, mock_mlflow, sample_predictions):
        """Test that MLflow run is created."""
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_123"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        
        run_id = log_prediction_metadata_to_mlflow(sample_predictions)
        
        assert run_id == "test_run_123"
        mock_mlflow.set_experiment.assert_called_once()
        mock_mlflow.start_run.assert_called_once()
    
    @patch('src.models.prediction_output.mlflow')
    def test_logs_parameters(self, mock_mlflow, sample_predictions):
        """Test that parameters are logged."""
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_123"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        
        log_prediction_metadata_to_mlflow(sample_predictions)
        
        # Check that log_param was called
        assert mock_mlflow.log_param.call_count >= 3
    
    @patch('src.models.prediction_output.mlflow')
    def test_logs_metrics(self, mock_mlflow, sample_predictions):
        """Test that metrics are logged."""
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_123"
        mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
        
        log_prediction_metadata_to_mlflow(sample_predictions)
        
        # Check that log_metric was called
        assert mock_mlflow.log_metric.call_count >= 4


class TestFormatPredictionsForDatabase:
    """Tests for format_predictions_for_database function."""
    
    def test_formats_data_types(self, sample_predictions):
        """Test that data types are correctly formatted."""
        formatted = format_predictions_for_database(sample_predictions)
        
        assert formatted['horizon'].dtype == int
        assert formatted['point_prediction'].dtype == float
        assert formatted['lower_bound'].dtype == float
        assert formatted['upper_bound'].dtype == float
    
    def test_column_order_is_correct(self, sample_predictions):
        """Test that columns are in correct order."""
        formatted = format_predictions_for_database(sample_predictions)
        
        expected_order = [
            'prediction_timestamp', 'prediction_date', 'horizon',
            'model_version', 'point_prediction', 'lower_bound', 'upper_bound'
        ]
        
        assert list(formatted.columns) == expected_order
    
    def test_preserves_all_rows(self, sample_predictions):
        """Test that all rows are preserved."""
        formatted = format_predictions_for_database(sample_predictions)
        
        assert len(formatted) == len(sample_predictions)


class TestCreatePredictionSummary:
    """Tests for create_prediction_summary function."""
    
    def test_creates_summary_dict(self, sample_predictions):
        """Test that summary dictionary is created."""
        summary = create_prediction_summary(sample_predictions)
        
        assert isinstance(summary, dict)
    
    def test_summary_contains_required_fields(self, sample_predictions):
        """Test that summary contains expected fields."""
        summary = create_prediction_summary(sample_predictions)
        
        required_fields = [
            'n_predictions', 'mean_prediction', 'median_prediction',
            'min_prediction', 'max_prediction', 'std_prediction'
        ]
        
        for field in required_fields:
            assert field in summary
    
    def test_summary_includes_interval_stats(self, sample_predictions):
        """Test that interval statistics are included."""
        summary = create_prediction_summary(sample_predictions)
        
        assert 'mean_interval_width' in summary
        assert 'median_interval_width' in summary
    
    def test_summary_calculations_correct(self, sample_predictions):
        """Test that summary calculations are correct."""
        summary = create_prediction_summary(sample_predictions)
        
        assert summary['n_predictions'] == 7
        assert summary['min_prediction'] == 60.0
        assert summary['max_prediction'] == 72.0
        # Interval width should be 20.0 for first prediction
        assert summary['mean_interval_width'] == 20.0
