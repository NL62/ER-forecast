"""
Weather data integration using Open-Meteo API.

Supports both:
- Historical weather data (ERA5 archive)
- Weather forecasts for prediction horizons
"""

import logging
import time
from typing import Optional, Tuple

import pandas as pd
import requests

# Configure module logger
logger = logging.getLogger(__name__)

# Open-Meteo API configuration
WEATHER_API_BASE_URL = "https://archive-api.open-meteo.com/v1/era5"
WEATHER_FORECAST_API_URL = "https://api.open-meteo.com/v1/forecast"
API_TIMEOUT = 10  # seconds
MAX_RETRIES = 3


def fetch_weather_data(
    start_date: str,
    end_date: str,
    lat: float,
    lon: float,
    base_url: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch historical weather data from Open-Meteo API.
    Returns DataFrame with temp, precipitation, and snowfall columns.
    """
    if base_url is None:
        base_url = WEATHER_API_BASE_URL
    
    logger.info(f"Fetching weather data from {start_date} to {end_date}")
    logger.info(f"Location: lat={lat}, lon={lon}")
    
    # Prepare API request parameters
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date,
        'end_date': end_date,
        'daily': 'temperature_2m_max,temperature_2m_min,temperature_2m_mean,precipitation_sum,snowfall_sum',
        'timezone': 'auto'
    }
    
    # Make API request with retry logic
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.debug(f"API request attempt {attempt}/{MAX_RETRIES}")
            
            response = requests.get(
                base_url,
                params=params,
                timeout=API_TIMEOUT
            )
            
            # Raise exception for HTTP errors (4xx, 5xx)
            response.raise_for_status()
            
            # Parse JSON response
            data = response.json()
            
            # Validate response contains expected data
            if 'daily' not in data:
                raise ValueError("API response missing 'daily' key")
            
            # Parse and convert to DataFrame
            weather_df = _parse_weather_response(data)
            
            logger.info(f"Successfully fetched weather data: {len(weather_df)} days")
            logger.debug(f"Temperature range: {weather_df['Temp_Mean'].min():.1f}°C to {weather_df['Temp_Mean'].max():.1f}°C")
            logger.debug(f"Total precipitation: {weather_df['Precipitation'].sum():.1f} mm")
            
            return weather_df
            
        except requests.exceptions.Timeout:
            logger.warning(f"API request timeout on attempt {attempt}/{MAX_RETRIES}")
            if attempt == MAX_RETRIES:
                logger.error("Max retries reached - API timeout")
                raise
            time.sleep(2 ** attempt)  # Exponential backoff
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"API request failed on attempt {attempt}/{MAX_RETRIES}: {e}")
            if attempt == MAX_RETRIES:
                logger.error(f"Max retries reached - API request failed: {e}")
                raise
            time.sleep(2 ** attempt)
            
        except (ValueError, KeyError) as e:
            logger.error(f"Invalid API response: {e}")
            raise
    
    # This should never be reached due to raise in the loop
    raise RuntimeError("Unexpected exit from retry loop")


def fetch_weather_forecast(
    lat: float,
    lon: float,
    forecast_days: int = 7,
    base_url: Optional[str] = None
) -> pd.DataFrame:
    """
    Fetch weather forecast data from Open-Meteo Forecast API.
    
    Returns forecast for the next N days with the same features as historical data.
    Used at prediction time to provide horizon-specific weather forecasts.
    
    Args:
        lat: Latitude of location
        lon: Longitude of location
        forecast_days: Number of days to forecast (1-16, default 7)
        base_url: Override API URL (for testing)
    
    Returns:
        DataFrame with columns:
            - Date: Forecast date
            - Temp_Max: Maximum temperature (C)
            - Temp_Min: Minimum temperature (C)
            - Temp_Mean: Mean temperature (C)
            - Precipitation: Total precipitation (mm)
            - Snowfall: Total snowfall (cm)
    
    Raises:
        requests.RequestException: If API request fails
        ValueError: If response cannot be parsed
    """
    if base_url is None:
        base_url = WEATHER_FORECAST_API_URL
    
    logger.info(f"Fetching weather forecast for next {forecast_days} days")
    logger.info(f"Location: lat={lat}, lon={lon}")
    
    # Prepare API request parameters
    params = {
        'latitude': lat,
        'longitude': lon,
        'daily': 'temperature_2m_max,temperature_2m_min,precipitation_sum,snowfall_sum',
        'forecast_days': forecast_days,
        'timezone': 'auto'
    }
    
    # Make API request with retry logic
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            logger.debug(f"Forecast API request attempt {attempt}/{MAX_RETRIES}")
            
            response = requests.get(
                base_url,
                params=params,
                timeout=API_TIMEOUT
            )
            
            response.raise_for_status()
            data = response.json()
            
            if 'daily' not in data:
                raise ValueError("API response missing 'daily' key")
            
            # Parse response
            forecast_df = _parse_forecast_response(data)
            
            logger.info(f"Successfully fetched weather forecast: {len(forecast_df)} days")
            logger.debug(f"Forecast date range: {forecast_df['Date'].min().date()} to "
                        f"{forecast_df['Date'].max().date()}")
            
            return forecast_df
            
        except requests.exceptions.Timeout:
            logger.warning(f"Forecast API timeout on attempt {attempt}/{MAX_RETRIES}")
            if attempt == MAX_RETRIES:
                logger.error("Max retries reached - forecast API timeout")
                raise
            time.sleep(2 ** attempt)
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Forecast API request failed on attempt {attempt}/{MAX_RETRIES}: {e}")
            if attempt == MAX_RETRIES:
                logger.error(f"Max retries reached - forecast API failed: {e}")
                raise
            time.sleep(2 ** attempt)
            
        except (ValueError, KeyError) as e:
            logger.error(f"Invalid forecast API response: {e}")
            raise
    
    raise RuntimeError("Unexpected exit from retry loop")


def _parse_forecast_response(response_data: dict) -> pd.DataFrame:
    """
    Parse Open-Meteo Forecast API response.
    
    The forecast API returns slightly different fields than the archive API,
    so we need a separate parser. Notably, it doesn't provide temperature_2m_mean
    directly, so we calculate it from max and min.
    
    Args:
        response_data: JSON response from Open-Meteo Forecast API
        
    Returns:
        DataFrame with parsed forecast data matching historical data format
    """
    try:
        daily_data = response_data['daily']
        
        # Create DataFrame
        forecast_df = pd.DataFrame({
            'Date': daily_data['time'],
            'Temp_Max': daily_data['temperature_2m_max'],
            'Temp_Min': daily_data['temperature_2m_min'],
            'Precipitation': daily_data['precipitation_sum'],
            'Snowfall': daily_data['snowfall_sum'],
        })
        
        # Convert date to datetime
        forecast_df['Date'] = pd.to_datetime(forecast_df['Date'])
        
        # Calculate mean temperature (forecast API doesn't provide it directly)
        forecast_df['Temp_Mean'] = (
            forecast_df['Temp_Max'] + forecast_df['Temp_Min']
        ) / 2
        
        # Handle any null values
        for col in ['Temp_Max', 'Temp_Min', 'Temp_Mean', 'Precipitation', 'Snowfall']:
            if forecast_df[col].isnull().any():
                n_missing = forecast_df[col].isnull().sum()
                logger.warning(f"{n_missing} missing values in forecast {col}, filling")
                forecast_df[col] = (
                    forecast_df[col]
                    .ffill()
                    .bfill()
                    .fillna(0)
                )
        
        # Reorder columns to match historical data format
        forecast_df = forecast_df[['Date', 'Temp_Max', 'Temp_Min', 'Temp_Mean', 
                                    'Precipitation', 'Snowfall']]
        
        return forecast_df
        
    except KeyError as e:
        logger.error(f"Missing expected key in forecast response: {e}")
        raise ValueError(f"Invalid forecast response structure: missing key {e}")
    except Exception as e:
        logger.error(f"Error parsing forecast data: {e}")
        raise ValueError(f"Failed to parse forecast data: {e}")


def _parse_weather_response(response_data: dict) -> pd.DataFrame:
    """
    Parse Open-Meteo API response and extract daily weather features.
    
    Internal helper function to convert API JSON response to structured DataFrame.
    
    Args:
        response_data: JSON response from Open-Meteo API
        
    Returns:
        DataFrame with parsed weather data
        
    Raises:
        KeyError: If expected keys are missing from response
        ValueError: If data cannot be parsed
    """
    try:
        daily_data = response_data['daily']
        
        # Create DataFrame from daily data
        weather_df = pd.DataFrame({
            'time': daily_data['time'],
            'temperature_2m_max': daily_data['temperature_2m_max'],
            'temperature_2m_min': daily_data['temperature_2m_min'],
            'temperature_2m_mean': daily_data['temperature_2m_mean'],
            'precipitation_sum': daily_data['precipitation_sum'],
            'snowfall_sum': daily_data['snowfall_sum'],
        })
        
        # Rename columns to match our naming convention
        weather_df.columns = [
            'Date',
            'Temp_Max',
            'Temp_Min',
            'Temp_Mean',
            'Precipitation',
            'Snowfall'
        ]
        
        # Convert date column to datetime
        weather_df['Date'] = pd.to_datetime(weather_df['Date'])
        
        # Handle missing values (API may return null for some days)
        # Forward fill first, then backward fill, then fill with 0
        for col in ['Temp_Max', 'Temp_Min', 'Temp_Mean', 'Precipitation', 'Snowfall']:
            if weather_df[col].isnull().any():
                n_missing = weather_df[col].isnull().sum()
                logger.warning(f"{n_missing} missing values in {col}, filling with interpolation")
                weather_df[col] = weather_df[col].ffill().bfill().fillna(0)
        
        return weather_df
        
    except KeyError as e:
        logger.error(f"Missing expected key in API response: {e}")
        raise ValueError(f"Invalid API response structure: missing key {e}")
    except Exception as e:
        logger.error(f"Error parsing weather data: {e}")
        raise ValueError(f"Failed to parse weather data: {e}")


def merge_weather_data(
    patient_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    fallback_strategy: str = 'last_known'
) -> pd.DataFrame:
    """
    Merge weather data with patient visit data on date.
    
    Performs a left join to ensure all patient visit dates are retained.
    Handles missing weather data using the specified fallback strategy.
    
    Args:
        patient_df: DataFrame with columns ['Date', 'Patients_per_day', ...]
        weather_df: DataFrame with columns ['Date', 'Temp_Max', 'Temp_Min', etc.]
        fallback_strategy: Strategy for handling missing weather data
            - 'last_known': Forward fill missing values
            - 'interpolate': Linear interpolation
            - 'zero': Fill with zeros
            - 'error': Raise error if any missing
            
    Returns:
        DataFrame with patient and weather data merged on Date
        
    Raises:
        ValueError: If dates don't overlap or fallback_strategy='error' and data is missing
        KeyError: If required columns are missing
    """
    logger.info("Merging weather data with patient visit data")
    
    # Validate input DataFrames
    if 'Date' not in patient_df.columns:
        raise KeyError("patient_df must contain 'Date' column")
    if 'Date' not in weather_df.columns:
        raise KeyError("weather_df must contain 'Date' column")
    
    # Ensure Date columns are datetime
    patient_df = patient_df.copy()
    weather_df = weather_df.copy()
    patient_df['Date'] = pd.to_datetime(patient_df['Date'])
    weather_df['Date'] = pd.to_datetime(weather_df['Date'])
    
    # Log date ranges
    patient_min, patient_max = patient_df['Date'].min(), patient_df['Date'].max()
    weather_min, weather_max = weather_df['Date'].min(), weather_df['Date'].max()
    
    logger.info(f"Patient data range: {patient_min.date()} to {patient_max.date()}")
    logger.info(f"Weather data range: {weather_min.date()} to {weather_max.date()}")
    
    # Check for date range overlap
    if patient_min > weather_max or patient_max < weather_min:
        raise ValueError(
            f"No date overlap between patient data ({patient_min.date()} to {patient_max.date()}) "
            f"and weather data ({weather_min.date()} to {weather_max.date()})"
        )
    
    # Perform left merge to keep all patient dates
    merged_df = patient_df.merge(weather_df, on='Date', how='left')
    
    # Check for missing weather data
    weather_cols = ['Temp_Max', 'Temp_Min', 'Temp_Mean', 'Precipitation', 'Snowfall']
    missing_weather = merged_df[weather_cols].isnull().any(axis=1).sum()
    
    if missing_weather > 0:
        logger.warning(f"Missing weather data for {missing_weather} days ({missing_weather/len(merged_df)*100:.1f}%)")
        
        # Apply fallback strategy
        if fallback_strategy == 'error':
            raise ValueError(f"Missing weather data for {missing_weather} days and fallback_strategy='error'")
        
        elif fallback_strategy == 'last_known':
            logger.info("Applying 'last_known' fallback: forward filling missing values")
            for col in weather_cols:
                merged_df[col] = merged_df[col].ffill().bfill().fillna(0)
        
        elif fallback_strategy == 'interpolate':
            logger.info("Applying 'interpolate' fallback: linear interpolation")
            for col in weather_cols:
                merged_df[col] = merged_df[col].interpolate(method='linear').bfill().ffill().fillna(0)
        
        elif fallback_strategy == 'zero':
            logger.info("Applying 'zero' fallback: filling with zeros")
            merged_df[weather_cols] = merged_df[weather_cols].fillna(0)
        
        else:
            raise ValueError(f"Unknown fallback_strategy: {fallback_strategy}. Choose from 'last_known', 'interpolate', 'zero', 'error'")
        
        logger.info(f"Filled missing weather data using '{fallback_strategy}' strategy")
    else:
        logger.info("No missing weather data - perfect merge!")
    
    # Log summary statistics
    logger.debug(f"Merged data shape: {merged_df.shape}")
    logger.debug(f"Temperature summary: min={merged_df['Temp_Mean'].min():.1f}°C, "
                f"max={merged_df['Temp_Mean'].max():.1f}°C, "
                f"mean={merged_df['Temp_Mean'].mean():.1f}°C")
    
    return merged_df
