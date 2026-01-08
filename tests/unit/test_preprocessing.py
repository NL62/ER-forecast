"""
Unit tests for data preprocessing module.

Tests cover:
- Loading raw CSV data
- Removing duplicate records
- Aggregating to daily counts
- Handling missing dates
"""

import pandas as pd
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import os

from src.data.preprocessing import (
    load_raw_data,
    remove_duplicates,
    aggregate_to_daily,
    handle_missing_dates,
)


class TestLoadRawData:
    """Tests for load_raw_data function."""
    
    def test_load_valid_csv(self, tmp_path):
        """Test loading a valid CSV file."""
        # Create test CSV
        csv_path = tmp_path / "test_data.csv"
        test_data = "123;2022-01-15 14:30:00\n456;2022-01-15 15:45:00\n"
        csv_path.write_text(test_data)
        
        # Load data
        df = load_raw_data(str(csv_path))
        
        # Assertions
        assert len(df) == 2
        assert list(df.columns) == ['Surrogate_Key', 'Contact_Start']
        assert df['Surrogate_Key'].tolist() == [123, 456]
    
    def test_load_csv_with_extra_columns(self, tmp_path):
        """Test loading CSV with more than 2 columns (should use first 2)."""
        csv_path = tmp_path / "test_data.csv"
        test_data = "123;2022-01-15 14:30:00;extra;data\n456;2022-01-15 15:45:00;more;stuff\n"
        csv_path.write_text(test_data)
        
        df = load_raw_data(str(csv_path))
        
        assert len(df.columns) == 2
        assert list(df.columns) == ['Surrogate_Key', 'Contact_Start']
    
    def test_load_nonexistent_file(self):
        """Test loading a file that doesn't exist."""
        with pytest.raises(FileNotFoundError):
            load_raw_data("nonexistent_file.csv")
    
    def test_load_empty_csv(self, tmp_path):
        """Test loading an empty CSV file."""
        csv_path = tmp_path / "empty.csv"
        csv_path.write_text("")
        
        with pytest.raises(pd.errors.EmptyDataError):
            load_raw_data(str(csv_path))


class TestRemoveDuplicates:
    """Tests for remove_duplicates function."""
    
    def test_no_duplicates(self):
        """Test DataFrame with no duplicates."""
        df = pd.DataFrame({
            'Surrogate_Key': [1, 2, 3],
            'Contact_Start': ['2022-01-01', '2022-01-02', '2022-01-03']
        })
        
        result = remove_duplicates(df)
        
        assert len(result) == 3
        assert result.equals(df.reset_index(drop=True))
    
    def test_with_duplicates(self):
        """Test DataFrame with duplicate records."""
        df = pd.DataFrame({
            'Surrogate_Key': [1, 2, 2, 3],
            'Contact_Start': ['2022-01-01', '2022-01-02', '2022-01-02', '2022-01-03']
        })
        
        result = remove_duplicates(df)
        
        assert len(result) == 3
        assert list(result['Surrogate_Key']) == [1, 2, 3]
    
    def test_keeps_first_occurrence(self):
        """Test that first occurrence is kept when duplicates exist."""
        df = pd.DataFrame({
            'Surrogate_Key': [1, 1, 1],
            'Contact_Start': ['2022-01-01', '2022-01-01', '2022-01-01'],
            'Extra': ['first', 'second', 'third']
        })
        
        result = remove_duplicates(df)
        
        assert len(result) == 1
        assert result['Extra'].iloc[0] == 'first'


class TestAggregateToDaily:
    """Tests for aggregate_to_daily function."""
    
    def test_single_day_single_visit(self):
        """Test aggregation with one visit on one day."""
        df = pd.DataFrame({
            'Surrogate_Key': [1],
            'Contact_Start': ['2022-01-01 14:30:00']
        })
        
        result = aggregate_to_daily(df)
        
        assert len(result) == 1
        assert result['Patients_per_day'].iloc[0] == 1
        assert pd.to_datetime(result['Date'].iloc[0]).date() == pd.to_datetime('2022-01-01').date()
    
    def test_single_day_multiple_visits(self):
        """Test aggregation with multiple visits on the same day."""
        df = pd.DataFrame({
            'Surrogate_Key': [1, 2, 3],
            'Contact_Start': [
                '2022-01-01 08:00:00',
                '2022-01-01 14:30:00',
                '2022-01-01 20:15:00'
            ]
        })
        
        result = aggregate_to_daily(df)
        
        assert len(result) == 1
        assert result['Patients_per_day'].iloc[0] == 3
    
    def test_multiple_days(self):
        """Test aggregation across multiple days."""
        df = pd.DataFrame({
            'Surrogate_Key': [1, 2, 3, 4, 5],
            'Contact_Start': [
                '2022-01-01 08:00:00',
                '2022-01-01 14:30:00',
                '2022-01-02 09:00:00',
                '2022-01-03 10:00:00',
                '2022-01-03 11:00:00'
            ]
        })
        
        result = aggregate_to_daily(df)
        
        assert len(result) == 3
        assert list(result['Patients_per_day']) == [2, 1, 2]
    
    def test_dates_are_sorted(self):
        """Test that output is sorted by date."""
        df = pd.DataFrame({
            'Surrogate_Key': [1, 2, 3],
            'Contact_Start': [
                '2022-01-03 10:00:00',
                '2022-01-01 10:00:00',
                '2022-01-02 10:00:00'
            ]
        })
        
        result = aggregate_to_daily(df)
        
        dates = result['Date'].tolist()
        assert dates == sorted(dates)


class TestHandleMissingDates:
    """Tests for handle_missing_dates function."""
    
    def test_no_missing_dates(self):
        """Test with complete date range (no gaps)."""
        df = pd.DataFrame({
            'Date': pd.date_range('2022-01-01', periods=5),
            'Patients_per_day': [10, 15, 12, 18, 20]
        })
        
        result = handle_missing_dates(df)
        
        assert len(result) == 5
        assert result['Patients_per_day'].tolist() == [10, 15, 12, 18, 20]
    
    def test_with_missing_dates(self):
        """Test with gaps in date range - missing dates filled with weekday median."""
        df = pd.DataFrame({
            'Date': pd.to_datetime(['2022-01-01', '2022-01-03', '2022-01-05']),
            'Patients_per_day': [10, 15, 20]
        })
        # Jan 1 = Saturday, Jan 2 = Sunday (missing), Jan 3 = Monday,
        # Jan 4 = Tuesday (missing), Jan 5 = Wednesday
        
        result = handle_missing_dates(df)
        
        # Should have 5 days (Jan 1-5)
        assert len(result) == 5
        # Missing dates filled with overall median (15) since no same-weekday data exists
        # for Sunday (Jan 2) or Tuesday (Jan 4)
        assert result['Patients_per_day'].tolist() == [10, 15, 15, 15, 20]
    
    def test_single_date(self):
        """Test with only one date."""
        df = pd.DataFrame({
            'Date': pd.to_datetime(['2022-01-01']),
            'Patients_per_day': [10]
        })
        
        result = handle_missing_dates(df)
        
        assert len(result) == 1
        assert result['Patients_per_day'].iloc[0] == 10
    
    def test_dates_remain_sorted(self):
        """Test that filled dates are properly sorted."""
        df = pd.DataFrame({
            'Date': pd.to_datetime(['2022-01-01', '2022-01-05']),
            'Patients_per_day': [10, 20]
        })
        
        result = handle_missing_dates(df)
        
        dates = result['Date'].tolist()
        assert dates == sorted(dates)


@pytest.fixture
def sample_raw_data(tmp_path):
    """Fixture providing sample raw CSV data."""
    csv_path = tmp_path / "sample_data.csv"
    data = """12345;2022-01-15 08:30:00
12346;2022-01-15 14:45:00
12347;2022-01-16 09:15:00"""
    csv_path.write_text(data)
    return csv_path


def test_complete_preprocessing_pipeline(sample_raw_data):
    """Integration test for complete preprocessing pipeline."""
    # Load
    df = load_raw_data(str(sample_raw_data))
    assert len(df) == 3
    
    # Remove duplicates (none in this case)
    df = remove_duplicates(df)
    assert len(df) == 3
    
    # Aggregate
    df = aggregate_to_daily(df)
    assert len(df) == 2  # Two distinct dates
    assert df['Patients_per_day'].tolist() == [2, 1]
    
    # Handle missing dates (none in this case)
    df = handle_missing_dates(df)
    assert len(df) == 2
