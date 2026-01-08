"""
Unit tests for weather integration module.

Tests cover:
- Fetching weather data from API (with mocking)
- Parsing API responses
- Error handling for API failures
- Merging weather data with patient data
"""

import pandas as pd
import pytest
from unittest.mock import patch, Mock
import requests

from src.data.weather_integration import (
    fetch_weather_data,
    _parse_weather_response,
    merge_weather_data,
)


class TestFetchWeatherData:
    """Tests for fetch_weather_data function."""
    
    @patch('src.data.weather_integration.requests.get')
    def test_successful_api_call(self, mock_get):
        """Test successful weather API call."""
        # Mock API response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'daily': {
                'time': ['2022-01-01', '2022-01-02'],
                'temperature_2m_max': [5.0, 7.0],
                'temperature_2m_min': [-2.0, 0.0],
                'temperature_2m_mean': [1.5, 3.5],
                'precipitation_sum': [2.0, 0.5],
                'snowfall_sum': [1.0, 0.0]
            }
        }
        mock_get.return_value = mock_response
        
        # Call function
        df = fetch_weather_data(
            start_date='2022-01-01',
            end_date='2022-01-02',
            lat=59.6099,
            lon=16.5448
        )
        
        # Assertions
        assert len(df) == 2
        assert list(df.columns) == ['Date', 'Temp_Max', 'Temp_Min', 'Temp_Mean', 'Precipitation', 'Snowfall']
        assert df['Temp_Max'].tolist() == [5.0, 7.0]
        assert df['Temp_Min'].tolist() == [-2.0, 0.0]
    
    @patch('src.data.weather_integration.requests.get')
    def test_api_timeout(self, mock_get):
        """Test handling of API timeout."""
        mock_get.side_effect = requests.exceptions.Timeout("Connection timeout")
        
        with pytest.raises(requests.exceptions.Timeout):
            fetch_weather_data(
                start_date='2022-01-01',
                end_date='2022-01-02',
                lat=59.6099,
                lon=16.5448
            )
    
    @patch('src.data.weather_integration.requests.get')
    def test_api_http_error(self, mock_get):
        """Test handling of HTTP errors (4xx, 5xx)."""
        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.raise_for_status.side_effect = requests.exceptions.HTTPError("Server error")
        mock_get.return_value = mock_response
        
        with pytest.raises(requests.exceptions.HTTPError):
            fetch_weather_data(
                start_date='2022-01-01',
                end_date='2022-01-02',
                lat=59.6099,
                lon=16.5448
            )
    
    @patch('src.data.weather_integration.requests.get')
    def test_invalid_response_structure(self, mock_get):
        """Test handling of invalid API response (missing 'daily' key)."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {'error': 'Invalid request'}
        mock_get.return_value = mock_response
        
        with pytest.raises(ValueError, match="missing 'daily' key"):
            fetch_weather_data(
                start_date='2022-01-01',
                end_date='2022-01-02',
                lat=59.6099,
                lon=16.5448
            )


class TestParseWeatherResponse:
    """Tests for _parse_weather_response function."""
    
    def test_parse_valid_response(self):
        """Test parsing valid API response."""
        response_data = {
            'daily': {
                'time': ['2022-01-01', '2022-01-02', '2022-01-03'],
                'temperature_2m_max': [5.0, 7.0, 6.0],
                'temperature_2m_min': [-2.0, 0.0, -1.0],
                'temperature_2m_mean': [1.5, 3.5, 2.5],
                'precipitation_sum': [2.0, 0.5, 1.0],
                'snowfall_sum': [1.0, 0.0, 0.5]
            }
        }
        
        df = _parse_weather_response(response_data)
        
        assert len(df) == 3
        assert list(df.columns) == ['Date', 'Temp_Max', 'Temp_Min', 'Temp_Mean', 'Precipitation', 'Snowfall']
        assert df['Temp_Max'].tolist() == [5.0, 7.0, 6.0]
    
    def test_parse_with_missing_values(self):
        """Test parsing response with null values."""
        response_data = {
            'daily': {
                'time': ['2022-01-01', '2022-01-02'],
                'temperature_2m_max': [5.0, None],
                'temperature_2m_min': [-2.0, 0.0],
                'temperature_2m_mean': [1.5, None],
                'precipitation_sum': [2.0, 0.5],
                'snowfall_sum': [1.0, 0.0]
            }
        }
        
        df = _parse_weather_response(response_data)
        
        # Should fill missing values
        assert not df.isnull().any().any()
    
    def test_parse_missing_key(self):
        """Test parsing response with missing required key."""
        response_data = {
            'daily': {
                'time': ['2022-01-01'],
                'temperature_2m_max': [5.0],
                # Missing other required keys
            }
        }
        
        with pytest.raises(ValueError):
            _parse_weather_response(response_data)


class TestMergeWeatherData:
    """Tests for merge_weather_data function."""
    
    def test_perfect_merge(self):
        """Test merging with perfect date overlap."""
        patient_df = pd.DataFrame({
            'Date': pd.date_range('2022-01-01', periods=3),
            'Patients_per_day': [50, 55, 60]
        })
        
        weather_df = pd.DataFrame({
            'Date': pd.date_range('2022-01-01', periods=3),
            'Temp_Max': [5.0, 7.0, 6.0],
            'Temp_Min': [-2.0, 0.0, -1.0],
            'Temp_Mean': [1.5, 3.5, 2.5],
            'Precipitation': [2.0, 0.5, 1.0],
            'Snowfall': [1.0, 0.0, 0.5]
        })
        
        result = merge_weather_data(patient_df, weather_df)
        
        assert len(result) == 3
        assert 'Temp_Max' in result.columns
        assert 'Patients_per_day' in result.columns
        assert not result.isnull().any().any()
    
    def test_merge_with_missing_weather(self):
        """Test merging when some dates lack weather data."""
        patient_df = pd.DataFrame({
            'Date': pd.date_range('2022-01-01', periods=5),
            'Patients_per_day': [50, 55, 60, 65, 70]
        })
        
        # Weather data only for first 3 days
        weather_df = pd.DataFrame({
            'Date': pd.date_range('2022-01-01', periods=3),
            'Temp_Max': [5.0, 7.0, 6.0],
            'Temp_Min': [-2.0, 0.0, -1.0],
            'Temp_Mean': [1.5, 3.5, 2.5],
            'Precipitation': [2.0, 0.5, 1.0],
            'Snowfall': [1.0, 0.0, 0.5]
        })
        
        # Use 'last_known' strategy to fill missing weather
        result = merge_weather_data(patient_df, weather_df, fallback_strategy='last_known')
        
        assert len(result) == 5
        # Should have filled missing weather data
        assert not result.isnull().any().any()
    
    def test_merge_with_zero_fallback(self):
        """Test merging with zero fallback strategy."""
        patient_df = pd.DataFrame({
            'Date': pd.date_range('2022-01-01', periods=3),
            'Patients_per_day': [50, 55, 60]
        })
        
        # Weather data only for first day
        weather_df = pd.DataFrame({
            'Date': pd.date_range('2022-01-01', periods=1),
            'Temp_Max': [5.0],
            'Temp_Min': [-2.0],
            'Temp_Mean': [1.5],
            'Precipitation': [2.0],
            'Snowfall': [1.0]
        })
        
        result = merge_weather_data(patient_df, weather_df, fallback_strategy='zero')
        
        assert len(result) == 3
        # Missing weather should be filled with zeros
        assert result.loc[1, 'Temp_Max'] == 0.0
        assert result.loc[2, 'Temp_Max'] == 0.0
    
    def test_merge_no_overlap_raises_error(self):
        """Test that merging with no date overlap raises error."""
        patient_df = pd.DataFrame({
            'Date': pd.date_range('2022-01-01', periods=3),
            'Patients_per_day': [50, 55, 60]
        })
        
        # Weather data for completely different dates
        weather_df = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=3),
            'Temp_Max': [5.0, 7.0, 6.0],
            'Temp_Min': [-2.0, 0.0, -1.0],
            'Temp_Mean': [1.5, 3.5, 2.5],
            'Precipitation': [2.0, 0.5, 1.0],
            'Snowfall': [1.0, 0.0, 0.5]
        })
        
        with pytest.raises(ValueError, match="No date overlap"):
            merge_weather_data(patient_df, weather_df)
    
    def test_merge_error_fallback(self):
        """Test that error fallback raises exception for missing data."""
        patient_df = pd.DataFrame({
            'Date': pd.date_range('2022-01-01', periods=3),
            'Patients_per_day': [50, 55, 60]
        })
        
        weather_df = pd.DataFrame({
            'Date': pd.date_range('2022-01-01', periods=2),  # Missing one day
            'Temp_Max': [5.0, 7.0],
            'Temp_Min': [-2.0, 0.0],
            'Temp_Mean': [1.5, 3.5],
            'Precipitation': [2.0, 0.5],
            'Snowfall': [1.0, 0.0]
        })
        
        with pytest.raises(ValueError, match="Missing weather data"):
            merge_weather_data(patient_df, weather_df, fallback_strategy='error')
    
    def test_merge_missing_date_column(self):
        """Test that missing Date column raises KeyError."""
        patient_df = pd.DataFrame({
            'NotDate': pd.date_range('2022-01-01', periods=3),
            'Patients_per_day': [50, 55, 60]
        })
        
        weather_df = pd.DataFrame({
            'Date': pd.date_range('2022-01-01', periods=3),
            'Temp_Max': [5.0, 7.0, 6.0],
            'Temp_Min': [-2.0, 0.0, -1.0],
            'Temp_Mean': [1.5, 3.5, 2.5],
            'Precipitation': [2.0, 0.5, 1.0],
            'Snowfall': [1.0, 0.0, 0.5]
        })
        
        with pytest.raises(KeyError, match="'Date' column"):
            merge_weather_data(patient_df, weather_df)
