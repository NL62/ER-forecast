"""
Prediction output - saving to CSV and logging to MLflow.
"""

import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd
import mlflow

# Configure module logger
logger = logging.getLogger(__name__)

# Database retry configuration
MAX_DB_RETRIES = 3
DB_RETRY_BASE_DELAY = 2  # seconds, will exponentially increase (2s, 4s, 8s)


def save_predictions_to_csv(
    predictions: pd.DataFrame,
    output_path: str,
    include_timestamp: bool = True
) -> str:
    """Save predictions to CSV file with optional timestamp in filename."""
    logger.info("Saving predictions to CSV")
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename with timestamp
    if include_timestamp:
        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"predictions_{timestamp_str}.csv"
    else:
        filename = "predictions_latest.csv"
    
    filepath = output_dir / filename
    
    try:
        # Save to CSV
        predictions.to_csv(filepath, index=False)
        
        file_size_kb = filepath.stat().st_size / 1024
        logger.info(f"Predictions saved to: {filepath}")
        logger.info(f"File size: {file_size_kb:.1f} KB, Rows: {len(predictions)}")
        
        return str(filepath)
        
    except Exception as e:
        logger.error(f"Failed to save predictions to CSV: {e}")
        raise


def write_predictions_to_database(
    predictions: pd.DataFrame,
    connection_string: str,
    table_name: str = '[LTV_STAGE].[dbo].[ER_PREDICTION]',
    if_exists: str = 'append'
) -> int:
    """
    Write predictions to SQL Server database.
    
    Inserts prediction records into the database table. Supports SQL Server
    with schema-qualified table names like [LTV_STAGE].[dbo].[ER_PREDICTION].
    
    Args:
        predictions: DataFrame with prediction results
        connection_string: SQL Server connection string (ODBC format or SQLAlchemy URI)
        table_name: Fully qualified table name (default: '[LTV_STAGE].[dbo].[ER_PREDICTION]')
        if_exists: What to do if table exists ('append', 'replace', 'fail')
    
    Returns:
        Number of rows written
    
    Raises:
        ValueError: If connection_string is empty or invalid
        Exception: If database write fails
    """
    import pyodbc
    
    logger.info(f"Writing predictions to database table: {table_name}")
    
    if not connection_string:
        raise ValueError("Connection string cannot be empty")
    
    if len(predictions) == 0:
        logger.warning("No predictions to write to database")
        return 0
    
    # Format predictions for database insertion
    formatted_predictions = format_predictions_for_database(predictions)
    
    # Parse connection string - support both ODBC and SQLAlchemy URI formats
    odbc_conn_string = connection_string
    
    if connection_string.startswith('mssql+pyodbc://') or connection_string.startswith('mssql://'):
        # Convert SQLAlchemy URI to ODBC connection string
        from sqlalchemy.engine.url import make_url
        from urllib.parse import unquote
        
        logger.debug("Converting SQLAlchemy URI to ODBC connection string")
        url = make_url(connection_string)
        
        # Build ODBC connection string
        odbc_parts = []
        if url.host:
            odbc_parts.append("DRIVER={ODBC Driver 17 for SQL Server}")
            server_str = url.host
            if url.port:
                server_str = f"{url.host},{url.port}"
            odbc_parts.append(f"SERVER={server_str}")
        if url.database:
            odbc_parts.append(f"DATABASE={url.database}")
        if url.username:
            odbc_parts.append(f"UID={url.username}")
        if url.password:
            odbc_parts.append(f"PWD={unquote(str(url.password))}")
        # Add encryption option for Azure SQL or modern SQL Server
        odbc_parts.append("TrustServerCertificate=yes")
        
        odbc_conn_string = ";".join(odbc_parts)
        logger.debug("Converted SQLAlchemy URI to ODBC format")
    
    # Validate table name format BEFORE retry loop (non-retryable validation)
    import re
    if not re.match(r'^(\[[\w]+\]\.)*\[[\w]+\]$', table_name.replace(' ', '')):
        raise ValueError(f"Invalid table name format: {table_name}. Expected format: [schema].[owner].[table] or [table]")
    
    # Prepare rows for bulk insert BEFORE retry loop (avoid repeated work)
    rows_to_insert = []
    for _, row in formatted_predictions.iterrows():
        rows_to_insert.append((
            pd.Timestamp(row['prediction_timestamp']),
            row['prediction_date'],
            int(row['horizon']),
            str(row['model_version']),
            float(row['point_prediction']),
            float(row['lower_bound']),
            float(row['upper_bound'])
        ))
    
    # Retry loop for transient database errors
    last_exception = None
    for attempt in range(1, MAX_DB_RETRIES + 1):
        conn = None
        cursor = None
        try:
            logger.debug(f"Database write attempt {attempt}/{MAX_DB_RETRIES}")
            conn = pyodbc.connect(odbc_conn_string)
            cursor = conn.cursor()
            
            # Handle if_exists parameter
            if if_exists == 'replace':
                logger.warning("'replace' option not supported for direct INSERT. Using 'append' instead.")
            elif if_exists == 'fail':
                check_query = f"SELECT COUNT(*) FROM {table_name}"
                try:
                    cursor.execute(check_query)
                    count = cursor.fetchone()[0]
                    if count > 0:
                        raise ValueError(f"Table {table_name} already contains data and if_exists='fail'")
                except pyodbc.Error:
                    logger.debug("Could not check if table exists, proceeding with insert")
            
            # Execute bulk insert
            insert_query = f"""
                INSERT INTO {table_name} 
                (prediction_timestamp, prediction_date, horizon, model_version, 
                 point_prediction, lower_bound, upper_bound)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            cursor.executemany(insert_query, rows_to_insert)
            conn.commit()
            
            n_rows = len(rows_to_insert)
            logger.info(f"Successfully wrote {n_rows} predictions to {table_name}")
            return n_rows
        
        except pyodbc.OperationalError as e:
            # Transient errors: connection lost, timeout, deadlock - RETRY
            last_exception = e
            logger.warning(f"Database operational error on attempt {attempt}/{MAX_DB_RETRIES}: {e}")
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            if attempt < MAX_DB_RETRIES:
                delay = DB_RETRY_BASE_DELAY ** attempt
                logger.info(f"Retrying in {delay} seconds...")
                time.sleep(delay)
            
        except pyodbc.DatabaseError as e:
            # Check if it's a transient error (deadlock, timeout)
            error_str = str(e).lower()
            if any(keyword in error_str for keyword in ['deadlock', 'timeout', 'connection']):
                last_exception = e
                logger.warning(f"Database transient error on attempt {attempt}/{MAX_DB_RETRIES}: {e}")
                if conn:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                if attempt < MAX_DB_RETRIES:
                    delay = DB_RETRY_BASE_DELAY ** attempt
                    logger.info(f"Retrying in {delay} seconds...")
                    time.sleep(delay)
            else:
                # Non-transient database error - don't retry
                logger.error(f"Database error (not retrying): {e}")
                if conn:
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                raise
        
        except (pyodbc.IntegrityError, pyodbc.ProgrammingError) as e:
            # Non-retryable: constraint violation, bad SQL - DON'T RETRY
            logger.error(f"Database error (not retrying): {e}")
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            raise
        
        except ValueError as e:
            # Validation errors - DON'T RETRY
            logger.error(f"Validation error (not retrying): {e}")
            raise
        
        except Exception as e:
            # Unexpected error - log and raise
            logger.error(f"Unexpected error writing to database: {e}")
            if conn:
                try:
                    conn.rollback()
                except Exception:
                    pass
            raise
        
        finally:
            if cursor:
                try:
                    cursor.close()
                except Exception:
                    pass
            if conn:
                try:
                    conn.close()
                except Exception:
                    pass
    
    # If we get here, all retries failed
    logger.error(f"All {MAX_DB_RETRIES} database write attempts failed")
    raise last_exception or Exception("Database write failed after all retries")


def log_prediction_metadata_to_mlflow(
    predictions: pd.DataFrame,
    run_name: Optional[str] = None
) -> str:
    """
    Log prediction run metadata to MLflow.
    
    Creates an MLflow run to track each prediction batch.
    Logs summary statistics and metadata for monitoring.
    
    Args:
        predictions: DataFrame with prediction results
        run_name: Optional custom run name
    
    Returns:
        MLflow run ID
    """
    logger.info("Logging prediction metadata to MLflow")
    
    if run_name is None:
        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        run_name = f"batch_prediction_{timestamp}"
    
    try:
        # Set experiment
        mlflow.set_experiment("ER_Patient_Predictions")
        
        with mlflow.start_run(run_name=run_name) as run:
            # Log parameters
            mlflow.log_param("n_predictions", len(predictions))
            mlflow.log_param("prediction_timestamp", predictions['prediction_timestamp'].iloc[0])
            mlflow.log_param("base_date", predictions['prediction_date'].min())
            mlflow.log_param("end_date", predictions['prediction_date'].max())
            
            # Log summary statistics
            mlflow.log_metric("mean_prediction", predictions['point_prediction'].mean())
            mlflow.log_metric("min_prediction", predictions['point_prediction'].min())
            mlflow.log_metric("max_prediction", predictions['point_prediction'].max())
            mlflow.log_metric("std_prediction", predictions['point_prediction'].std())
            
            # Log interval statistics if available
            if 'lower_bound' in predictions.columns and 'upper_bound' in predictions.columns:
                avg_interval_width = (predictions['upper_bound'] - predictions['lower_bound']).mean()
                mlflow.log_metric("avg_interval_width", avg_interval_width)
            
            # Log predictions as artifact
            predictions_csv = "predictions.csv"
            predictions.to_csv(predictions_csv, index=False)
            
            # Try to log artifact (may fail if running locally)
            try:
                mlflow.log_artifact(predictions_csv)
            except OSError as e:
                if "Read-only file system" in str(e):
                    logger.warning(f"Skipping MLflow artifact logging (running locally): {e}")
                else:
                    raise
            
            # Clean up temp file
            Path(predictions_csv).unlink(missing_ok=True)
            
            run_id = run.info.run_id
            logger.info(f"Prediction metadata logged to MLflow run: {run_id}")
            
            return run_id
            
    except Exception as e:
        logger.error(f"Failed to log prediction metadata to MLflow: {e}")
        raise


def format_predictions_for_database(predictions: pd.DataFrame) -> pd.DataFrame:
    """
    Format predictions DataFrame for database insertion.
    
    Ensures all columns match the database schema and data types are correct.
    
    Args:
        predictions: Raw predictions DataFrame
    
    Returns:
        Formatted DataFrame ready for database insertion
    """
    logger.debug("Formatting predictions for database schema")
    
    formatted = predictions.copy()
    
    # Ensure correct data types
    formatted['prediction_timestamp'] = pd.to_datetime(formatted['prediction_timestamp'])
    formatted['prediction_date'] = pd.to_datetime(formatted['prediction_date']).dt.date
    formatted['horizon'] = formatted['horizon'].astype(int)
    formatted['model_version'] = formatted['model_version'].astype(str)
    formatted['point_prediction'] = formatted['point_prediction'].astype(float)
    formatted['lower_bound'] = formatted['lower_bound'].astype(float)
    formatted['upper_bound'] = formatted['upper_bound'].astype(float)
    
    # Ensure columns are in correct order
    column_order = [
        'prediction_timestamp',
        'prediction_date',
        'horizon',
        'model_version',
        'point_prediction',
        'lower_bound',
        'upper_bound'
    ]
    
    formatted = formatted[column_order]
    
    logger.debug(f"Formatted {len(formatted)} predictions for database insertion")
    
    return formatted


def create_prediction_summary(predictions: pd.DataFrame) -> Dict[str, Any]:
    """
    Create summary statistics from predictions.
    
    Useful for quick reporting and monitoring.
    
    Args:
        predictions: DataFrame with prediction results
    
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        'n_predictions': len(predictions),
        'prediction_date_start': predictions['prediction_date'].min(),
        'prediction_date_end': predictions['prediction_date'].max(),
        'mean_prediction': predictions['point_prediction'].mean(),
        'median_prediction': predictions['point_prediction'].median(),
        'min_prediction': predictions['point_prediction'].min(),
        'max_prediction': predictions['point_prediction'].max(),
        'std_prediction': predictions['point_prediction'].std(),
    }
    
    # Add interval statistics if available
    if 'lower_bound' in predictions.columns and 'upper_bound' in predictions.columns:
        interval_widths = predictions['upper_bound'] - predictions['lower_bound']
        summary['mean_interval_width'] = interval_widths.mean()
        summary['median_interval_width'] = interval_widths.median()
    
    logger.debug(f"Created prediction summary: {len(summary)} metrics")
    
    return summary
