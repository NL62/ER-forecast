"""
Shared pytest fixtures for all tests.

This module provides reusable fixtures for:
- Sample data generation
- Mock objects
- Temporary directories
- Database connections (future)
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import tempfile


@pytest.fixture
def temp_dir():
    """Create temporary directory that is cleaned up after test."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def sample_dates():
    """Generate sample date range."""
    return pd.date_range('2024-01-01', periods=365, freq='D')


@pytest.fixture
def sample_patient_counts():
    """Generate realistic sample patient counts."""
    np.random.seed(42)
    # Base count with weekly and seasonal patterns
    n_days = 365
    base = 60
    weekly_pattern = 10 * np.sin(2 * np.pi * np.arange(n_days) / 7)
    seasonal_pattern = 15 * np.sin(2 * np.pi * np.arange(n_days) / 365)
    noise = np.random.normal(0, 5, n_days)
    
    counts = base + weekly_pattern + seasonal_pattern + noise
    return counts.clip(min=20)  # Ensure positive counts


@pytest.fixture
def sample_daily_dataframe(sample_dates, sample_patient_counts):
    """Create sample daily patient DataFrame."""
    return pd.DataFrame({
        'Date': sample_dates,
        'Patients_per_day': sample_patient_counts
    })


@pytest.fixture
def sample_featured_dataframe(sample_daily_dataframe):
    """Create sample DataFrame with engineered features."""
    from src.data.feature_engineering import engineer_features, remove_nan_rows
    
    df = sample_daily_dataframe.copy()
    df = engineer_features(df)
    df = remove_nan_rows(df)
    
    return df


@pytest.fixture(scope="session")
def project_root():
    """Get project root directory."""
    return Path(__file__).parent.parent


@pytest.fixture
def config_dir(project_root):
    """Get configuration directory."""
    return project_root / "config"


@pytest.fixture
def models_dir(project_root):
    """Get models directory."""
    models_path = project_root / "models"
    models_path.mkdir(exist_ok=True)
    return models_path


@pytest.fixture
def data_dir(project_root):
    """Get data directory."""
    return project_root / "data"


# Mark slow tests
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
