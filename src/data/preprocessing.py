"""
Data preprocessing for patient visit data.
"""

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Union

import pandas as pd

# Configure module logger
logger = logging.getLogger(__name__)


def load_raw_data(csv_path: Union[str, Path]) -> pd.DataFrame:
    """
    Load raw patient visit data from CSV file.
    Expected format: Surrogate_Key, Contact_Start columns.
    """
    csv_path = Path(csv_path)
    
    logger.info(f"Loading raw data from: {csv_path}")
    
    # Check if file exists
    if not csv_path.exists():
        logger.error(f"File not found: {csv_path}")
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    try:
        # Read CSV with semicolon delimiter (common in European data)
        # header=None because raw data may not have headers
        df = pd.read_csv(csv_path, delimiter=';', header=None)
        
        # If DataFrame is empty, raise error
        if df.empty:
            logger.error(f"CSV file is empty: {csv_path}")
            raise pd.errors.EmptyDataError(f"CSV file is empty: {csv_path}")
        
        # Select first two columns and rename them
        df = df[[0, 1]].copy()
        df.columns = ['Surrogate_Key', 'Contact_Start']
        
        logger.info(f"Successfully loaded {len(df):,} rows from {csv_path}")
        logger.debug(f"Data shape: {df.shape}")
        logger.debug(f"Columns: {df.columns.tolist()}")
        logger.debug(f"Data types: {df.dtypes.to_dict()}")
        
        return df
        
    except pd.errors.EmptyDataError as e:
        logger.error(f"Empty data error: {e}")
        raise
    except pd.errors.ParserError as e:
        logger.error(f"Parser error while reading CSV: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading CSV: {e}")
        raise


def _normalize_odbc_connection_string(connection_string: str) -> str:
    """
    Normalize and clean an ODBC connection string.
    
    Removes duplicate keys (keeps last occurrence) from ODBC connection strings.
    This fixes the "cannot assemble with duplicate keys" error from pyodbc.
    
    Args:
        connection_string: ODBC connection string (e.g., "DRIVER={...};SERVER=...;DATABASE=...")
    
    Returns:
        Clean ODBC connection string without duplicate keys
    """
    # Parse ODBC connection string and remove duplicates
    # ODBC format: KEY1=value1;KEY2=value2;KEY3=value3
    parts = connection_string.split(';')
    seen_keys = {}  # Track last occurrence of each key (case-insensitive)
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        # Handle key=value pairs
        if '=' in part:
            key, value = part.split('=', 1)
            key = key.strip()
            value = value.strip()
            
            # Keep track of keys (case-insensitive for ODBC)
            key_upper = key.upper()
            # Store last occurrence (overwrites previous)
            seen_keys[key_upper] = f"{key}={value}"
    
    # Reconstruct connection string from unique keys (last occurrence kept)
    cleaned_parts = list(seen_keys.values())
    cleaned_string = ";".join(cleaned_parts)
    
    if cleaned_string != connection_string:
        logger.debug("Removed duplicate keys from connection string")
        logger.debug(f"Original: {connection_string[:100]}...")
        logger.debug(f"Cleaned: {cleaned_string[:100]}...")
    
    return cleaned_string


def load_data_from_database(
    connection_string: str,
    stored_procedure: str = '[getVPB_Data]'
) -> pd.DataFrame:
    """
    Load raw patient visit data from SQL Server database using a stored procedure.
    
    Executes the stored procedure and returns a DataFrame with columns
    ['Surrogate_Key', 'Contact_Start'] matching the format expected by
    the preprocessing pipeline.
    
    Args:
        connection_string: SQL Server ODBC connection string
        stored_procedure: Name of the stored procedure to execute (default: '[getVPB_Data]')
    
    Returns:
        DataFrame with columns ['Surrogate_Key', 'Contact_Start']
    
    Raises:
        ValueError: If connection_string is empty
        Exception: If database query fails
    """
    import pyodbc
    
    logger.info(f"Loading data from SQL Server using stored procedure: {stored_procedure}")
    
    if not connection_string:
        raise ValueError("Connection string cannot be empty")
    
    try:
        # Normalize connection string (remove duplicate keys)
        cleaned_conn_string = _normalize_odbc_connection_string(connection_string)
        
        # Connect to SQL Server
        logger.debug("Connecting to SQL Server...")
        conn = pyodbc.connect(cleaned_conn_string)
        cursor = conn.cursor()
        
        try:
            # Validate stored procedure name format to prevent SQL injection
            # Should be in format [schema].[procedure] or [procedure]
            import re
            # Pattern matches: [name] or [name].[name] or [name].[name].[name]
            # Allows word characters (letters, digits, underscore) inside brackets
            if not re.match(r'^(\[[\w]+\]\.)*\[[\w]+\]$', stored_procedure.replace(' ', '')):
                raise ValueError(f"Invalid stored procedure name format: {stored_procedure}. Expected format: [schema].[procedure] or [procedure]")
            
            # Execute stored procedure
            # Note: We validate the name above, but SQL Server brackets provide additional protection
            query = f"EXEC {stored_procedure}"
            logger.debug(f"Executing: {query}")
            
            cursor.execute(query)
            
            # Fetch all results
            rows = cursor.fetchall()
            
            if not rows:
                logger.warning("Stored procedure returned no rows")
                return pd.DataFrame(columns=['Surrogate_Key', 'Contact_Start'])
            
            # Extract first two columns from each row (same format as raw_real.csv)
            # Column 0: Surrogate_Key (ID), Column 1: Contact_Start (datetime)
            # This matches the format from parse_db_result - treating stored procedure 
            # results the same way as the raw_real.csv file
            data = []
            for row in rows:
                row_tuple = tuple(row)
                if len(row_tuple) < 2:
                    logger.warning(f"Row has fewer than 2 columns, skipping: {row_tuple}")
                    continue
                
                surrogate_key = row_tuple[0]
                contact_start = row_tuple[1]
                
                data.append({
                    'Surrogate_Key': surrogate_key,
                    'Contact_Start': contact_start
                })
            
            # Create DataFrame from extracted data
            df = pd.DataFrame(data)
            
            if len(df) == 0:
                logger.warning("No valid rows extracted from stored procedure results")
                return pd.DataFrame(columns=['Surrogate_Key', 'Contact_Start'])
            
            # Ensure Contact_Start is datetime
            df['Contact_Start'] = pd.to_datetime(df['Contact_Start'])
            
            # Convert Surrogate_Key to string (in case it's numeric)
            df['Surrogate_Key'] = df['Surrogate_Key'].astype(str)
            
            logger.info(f"Successfully loaded {len(df):,} rows from database")
            logger.debug(f"Data shape: {df.shape}")
            logger.debug(f"Columns: {list(df.columns)}")
            if len(df) > 0:
                logger.debug(f"Date range: {df['Contact_Start'].min()} to {df['Contact_Start'].max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to execute stored procedure: {e}")
            raise
        finally:
            cursor.close()
            conn.close()
            
    except Exception as e:
        logger.error(f"Failed to load data from database: {e}")
        raise


def parse_db_result(file_path: Union[str, Path]) -> pd.DataFrame:
    file_path = Path(file_path)
    
    logger.info(f"Parsing database result from: {file_path}")
    
    # Check if file exists
    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"Database result file not found: {file_path}")
    
    try:
        rows = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        # Skip first line if it's just a closing parenthesis
        start_idx = 0
        if lines and lines[0].strip() == ')':
            start_idx = 1
        
        # Pattern to extract ID (first number) and first datetime
        id_pattern = r'^\((\d+)'
        datetime_pattern = r'datetime\.datetime\((\d{4}),\s*(\d{1,2}),\s*(\d{1,2}),\s*(\d{1,2}),\s*(\d{1,2}),\s*(\d{1,2})(?:,\s*(\d+))?\)'
        
        for line_num, line in enumerate(lines[start_idx:], start=start_idx + 1):
            line = line.strip()
            
            # Skip empty lines or standalone closing parentheses
            if not line or line == ')':
                continue
            
            try:
                # Extract ID (first number in the tuple)
                id_match = re.search(id_pattern, line)
                if not id_match:
                    logger.warning(f"Could not find ID on line {line_num}")
                    continue
                
                surrogate_key = id_match.group(1)
                
                # Extract first datetime
                datetime_match = re.search(datetime_pattern, line)
                if not datetime_match:
                    logger.warning(f"Could not find datetime on line {line_num}")
                    continue
                
                # Parse datetime components
                year, month, day, hour, minute, second = datetime_match.groups()[:6]
                microsecond = datetime_match.groups()[6] if datetime_match.groups()[6] else '0'
                
                try:
                    contact_start = datetime(
                        int(year), int(month), int(day),
                        int(hour), int(minute), int(second),
                        int(microsecond)
                    )
                    
                    rows.append({
                        'Surrogate_Key': surrogate_key,
                        'Contact_Start': contact_start
                    })
                except ValueError as e:
                    logger.warning(f"Invalid datetime on line {line_num}: {e}")
                    continue
                    
            except Exception as e:
                logger.warning(f"Error parsing line {line_num}: {e}")
                logger.debug(f"Problematic line: {line[:200]}...")
                continue
        
        if not rows:
            logger.error("No valid rows parsed from database result file")
            raise ValueError("No valid data could be parsed from the file")
        
        # Create DataFrame
        df = pd.DataFrame(rows)
        
        # Convert Contact_Start to pandas datetime
        df['Contact_Start'] = pd.to_datetime(df['Contact_Start'])
        
        logger.info(f"Successfully parsed {len(df):,} rows from {file_path}")
        logger.debug(f"Data shape: {df.shape}")
        logger.debug(f"Columns: {df.columns.tolist()}")
        logger.debug(f"Date range: {df['Contact_Start'].min()} to {df['Contact_Start'].max()}")
        
        return df
        
    except Exception as e:
        logger.error(f"Error parsing database result file: {e}")
        raise


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate patient visit records.
    
    Duplicates are identified by matching both Surrogate_Key and Contact_Start.
    Keeps the first occurrence and removes subsequent duplicates.
    
    Args:
        df: DataFrame with columns ['Surrogate_Key', 'Contact_Start']
        
    Returns:
        DataFrame with duplicates removed
    """
    original_count = len(df)
    
    logger.info(f"Removing duplicates from {original_count:,} rows")
    
    # Identify duplicates based on both columns
    duplicates = df.duplicated(subset=['Surrogate_Key', 'Contact_Start'], keep='first')
    n_duplicates = duplicates.sum()
    
    if n_duplicates > 0:
        logger.warning(f"Found {n_duplicates:,} duplicate rows ({n_duplicates/original_count*100:.2f}%)")
        
        # Remove duplicates
        df = df[~duplicates].copy()
        df = df.reset_index(drop=True)
        
        logger.info(f"After removing duplicates: {len(df):,} rows remain")
    else:
        logger.info("No duplicates found")
    
    return df


def aggregate_to_daily(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate patient visits to daily counts.
    
    Converts Contact_Start timestamps to dates and counts the number of patient visits per day.
    
    Args:
        df: DataFrame with columns ['Surrogate_Key', 'Contact_Start']
        
    Returns:
        DataFrame with columns ['Date', 'Patients_per_day']
        Sorted by date in ascending order
    """
    logger.info("Aggregating patient visits to daily counts")
    
    # Convert Contact_Start to datetime and extract date
    df['Contact_Start'] = pd.to_datetime(df['Contact_Start'])
    df['Date'] = df['Contact_Start'].dt.date
    
    # Group by date and count visits
    df_agg = df.groupby(['Date']).size().reset_index(name='Patients_per_day')
    
    # Convert Date back to datetime for consistency
    df_agg['Date'] = pd.to_datetime(df_agg['Date'])
    
    # Sort by date
    df_agg = df_agg.sort_values('Date').reset_index(drop=True)
    
    logger.info(f"Aggregated to {len(df_agg):,} days")
    logger.info(f"Date range: {df_agg['Date'].min()} to {df_agg['Date'].max()}")
    logger.info(f"Average patients per day: {df_agg['Patients_per_day'].mean():.1f}")
    logger.info(f"Min patients per day: {df_agg['Patients_per_day'].min()}")
    logger.info(f"Max patients per day: {df_agg['Patients_per_day'].max()}")
    
    return df_agg


def handle_missing_dates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill in any missing dates using median of same weekday.
    
    Creates a complete date range from min to max date and fills missing dates
    with the median patient count for that weekday. This is more realistic than
    filling with 0 (which would indicate the hospital was closed) and preserves
    the weekly pattern in the data.
    
    Args:
        df: DataFrame with columns ['Date', 'Patients_per_day']
        
    Returns:
        DataFrame with complete date range (no gaps)
    """
    logger.info("Checking for missing dates in time series")
    
    # Get date range
    min_date = df['Date'].min()
    max_date = df['Date'].max()
    
    # Create complete date range
    complete_date_range = pd.date_range(start=min_date, end=max_date, freq='D')
    
    # Find missing dates
    existing_dates = set(df['Date'])
    missing_dates = [d for d in complete_date_range if d not in existing_dates]
    
    if missing_dates:
        n_missing = len(missing_dates)
        logger.warning(f"Found {n_missing} missing dates ({n_missing/len(complete_date_range)*100:.2f}%)")
        
        # Calculate median patients per weekday (0=Monday, 6=Sunday)
        df['_weekday'] = df['Date'].dt.dayofweek
        weekday_medians = df.groupby('_weekday')['Patients_per_day'].median().to_dict()
        df = df.drop(columns=['_weekday'])
        
        # Overall median as fallback
        overall_median = df['Patients_per_day'].median()
        
        logger.debug(f"Weekday medians: {weekday_medians}")
        
        # Create DataFrame with complete date range
        df_complete = pd.DataFrame({'Date': complete_date_range})
        
        # Merge with existing data
        df = df_complete.merge(df, on='Date', how='left')
        
        # Fill missing values with weekday median
        missing_mask = df['Patients_per_day'].isna()
        if missing_mask.any():
            df['_weekday'] = df['Date'].dt.dayofweek
            df.loc[missing_mask, 'Patients_per_day'] = df.loc[missing_mask, '_weekday'].map(
                lambda x: weekday_medians.get(x, overall_median)
            )
            df = df.drop(columns=['_weekday'])
            df['Patients_per_day'] = df['Patients_per_day'].astype(int)
        
        logger.info(f"Filled {n_missing} missing dates with weekday median patient counts")
        
        # Log which dates were filled
        for d in missing_dates[:5]:  # Log first 5
            logger.debug(f"  Filled {d.date()} ({d.strftime('%A')})")
        if len(missing_dates) > 5:
            logger.debug(f"  ... and {len(missing_dates) - 5} more")
    else:
        logger.info("No missing dates found - time series is complete")
    
    # Sort by date and reset index
    df = df.sort_values('Date').reset_index(drop=True)
    
    logger.info(f"Final time series: {len(df):,} days from {min_date.date()} to {max_date.date()}")
    
    return df


def detect_and_handle_outliers(
    df: pd.DataFrame,
    method: str = 'iqr',
    action: str = 'cap',
    iqr_multiplier: float = 2.0
) -> pd.DataFrame:
    """
    Detect and handle outliers in patient counts.
    
    Outliers can occur due to data entry errors, system issues, or genuine
    unusual events. This function identifies outliers and either flags them,
    caps them, or removes them.
    
    Args:
        df: DataFrame with columns ['Date', 'Patients_per_day']
        method: Detection method - 'iqr' (interquartile range) or 'percentile'
        action: What to do with outliers - 'flag', 'cap', or 'remove'
        iqr_multiplier: For IQR method, how many IQRs from Q1/Q3 (default 2.5)
        
    Returns:
        DataFrame with outliers handled according to action
    """
    logger.info(f"Detecting outliers using {method} method")
    
    df = df.copy()
    patients = df['Patients_per_day']
    
    # Calculate bounds based on method
    if method == 'iqr':
        Q1 = patients.quantile(0.25)
        Q3 = patients.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = max(0, Q1 - iqr_multiplier * IQR)  # Can't be negative
        upper_bound = Q3 + iqr_multiplier * IQR
    elif method == 'percentile':
        lower_bound = max(0, patients.quantile(0.01))
        upper_bound = patients.quantile(0.99)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'iqr' or 'percentile'")
    
    # Identify outliers
    lower_outliers = patients < lower_bound
    upper_outliers = patients > upper_bound
    outliers = lower_outliers | upper_outliers
    n_outliers = outliers.sum()
    
    logger.info(f"Outlier bounds: [{lower_bound:.0f}, {upper_bound:.0f}] patients")
    logger.info(f"Found {n_outliers} outliers ({n_outliers/len(df)*100:.2f}%)")
    
    if n_outliers > 0:
        # Log details about outliers
        outlier_rows = df[outliers]
        logger.warning(f"Outlier summary:")
        logger.warning(f"  Below {lower_bound:.0f}: {lower_outliers.sum()} days")
        logger.warning(f"  Above {upper_bound:.0f}: {upper_outliers.sum()} days")
        
        # Show some examples
        if upper_outliers.any():
            top_outliers = df[upper_outliers].nlargest(3, 'Patients_per_day')
            for _, row in top_outliers.iterrows():
                logger.warning(f"  High: {row['Date'].date()} = {row['Patients_per_day']:.0f} patients")
        
        if lower_outliers.any():
            low_outliers_df = df[lower_outliers].nsmallest(3, 'Patients_per_day')
            for _, row in low_outliers_df.iterrows():
                logger.warning(f"  Low: {row['Date'].date()} = {row['Patients_per_day']:.0f} patients")
        
        # Handle outliers based on action
        if action == 'flag':
            # Just add a flag column
            df['is_outlier'] = outliers
            logger.info("Outliers flagged in 'is_outlier' column")
            
        elif action == 'cap':
            # Cap values at bounds
            original_values = df.loc[outliers, 'Patients_per_day'].copy()
            df.loc[lower_outliers, 'Patients_per_day'] = int(lower_bound)
            df.loc[upper_outliers, 'Patients_per_day'] = int(upper_bound)
            logger.info(f"Capped {n_outliers} outliers to bounds [{lower_bound:.0f}, {upper_bound:.0f}]")
            
        elif action == 'remove':
            # Remove outlier rows (use with caution - creates gaps!)
            df = df[~outliers].copy()
            logger.warning(f"Removed {n_outliers} outlier rows - time series now has gaps!")
            logger.warning("Consider using 'cap' instead to preserve time series continuity")
            
        else:
            raise ValueError(f"Unknown action: {action}. Use 'flag', 'cap', or 'remove'")
    else:
        logger.info("No outliers detected")
    
    return df


def filter_incomplete_current_day(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove current day's data to avoid using incomplete/partial day counts.
    
    When running predictions or training early in the day, today's data
    is incomplete and would corrupt lag features.
    
    Args:
        df: DataFrame with 'Date' column (after aggregation to daily)
        
    Returns:
        DataFrame with current day's data removed
    """
    today = datetime.now().date()
    
    # Handle both datetime and date types in the Date column
    if hasattr(df['Date'].iloc[0], 'date'):
        df_dates = df['Date'].dt.date
    else:
        df_dates = df['Date']
    
    before_count = len(df)
    df = df[df_dates < today].copy()
    removed = before_count - len(df)
    
    if removed > 0:
        logger.info(f"Filtered out {removed} row(s) from today ({today}) to avoid incomplete data")
    
    return df
