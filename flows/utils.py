"""
Shared utilities for flow scripts.
"""

import logging
from datetime import datetime
from typing import Dict, List, Any

import pandas as pd

# Configure module logger
logger = logging.getLogger(__name__)


def handle_task_failure(task_name: str, error: Exception) -> None:
    """Standardized error logging for failed tasks."""
    
    logger.error(f"{'='*70}")
    logger.error(f"TASK FAILURE: {task_name}")
    logger.error(f"{'='*70}")
    logger.error(f"Error type: {type(error).__name__}")
    logger.error(f"Error message: {str(error)}")
    logger.error(f"{'='*70}")
    
    # TODO: Send notification
    # send_notification(
    #     message=f"Task '{task_name}' failed: {str(error)}",
    #     level="error"
    # )


def send_notification(message: str, level: str = "info") -> None:
    """
    Send notification about flow events.
    
    This is a placeholder for future notification integration.
    Could be extended to support email, Slack, PagerDuty, etc.
    
    Args:
        message: Notification message
        level: Notification level ('info', 'warning', 'error', 'critical')
    """
    logger.info(f"[NOTIFICATION - {level.upper()}] {message}")
    
    # TODO: Implement actual notification system
    
    # Placeholder implementation
    if level in ['error', 'critical']:
        logger.warning("Critical notification triggered but not configured")
        logger.warning(f"Message: {message}")


def validate_flow_parameters(
    params: Dict[str, Any],
    required_keys: List[str]
) -> bool:
    """
    Validate flow input parameters.
    
    Checks that all required parameters are present and valid.
    
    Args:
        params: Dictionary of parameters to validate
        required_keys: List of required parameter keys
    
    Returns:
        True if valid, raises ValueError if invalid
    
    Raises:
        ValueError: If validation fails
    """
    logger.debug(f"Validating flow parameters: {list(params.keys())}")
    
    # Check for required keys
    missing_keys = set(required_keys) - set(params.keys())
    
    if missing_keys:
        raise ValueError(f"Missing required parameters: {missing_keys}")
    
    # Check for None values in required keys
    none_values = [k for k in required_keys if params.get(k) is None]
    
    if none_values:
        raise ValueError(f"Required parameters have None values: {none_values}")
    
    logger.debug("Parameter validation passed")
    
    return True


def log_flow_start(flow_name: str, params: Dict[str, Any]) -> None:
    """
    Log flow start with parameters.
    
    Provides consistent logging format for flow initialization.
    
    Args:
        flow_name: Name of the flow
        params: Flow parameters
    """
    logger.info("="*70)
    logger.info(f"STARTING FLOW: {flow_name}")
    logger.info("="*70)
    logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Parameters:")
    
    for key, value in params.items():
        logger.info(f"  {key}: {value}")
    
    logger.info("="*70)


def log_flow_end(flow_name: str, success: bool, duration_seconds: float = None) -> None:
    """
    Log flow completion.
    
    Provides consistent logging format for flow completion.
    
    Args:
        flow_name: Name of the flow
        success: Whether flow completed successfully
        duration_seconds: Optional flow duration
    """
    status = "SUCCESS" if success else " FAILED"
    
    logger.info("="*70)
    logger.info(f"FLOW COMPLETE: {flow_name} - {status}")
    logger.info("="*70)
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if duration_seconds:
        logger.info(f"Duration: {duration_seconds:.1f} seconds ({duration_seconds/60:.1f} minutes)")
    
    logger.info("="*70)


def check_data_freshness(df: pd.DataFrame, max_age_days: int = 7) -> bool:
    """
    Check if data is fresh enough for predictions.
    
    Ensures the most recent data is not too old, which could indicate
    a data pipeline issue.
    
    Args:
        df: DataFrame with 'Date' column
        max_age_days: Maximum acceptable age of most recent data
    
    Returns:
        True if data is fresh, False otherwise
    """
    latest_date = df['Date'].max()
    today = pd.Timestamp.now().normalize()
    age_days = (today - latest_date).days
    
    logger.info(f"Data freshness check:")
    logger.info(f"  Latest data: {latest_date.date()}")
    logger.info(f"  Today: {today.date()}")
    logger.info(f"  Age: {age_days} days")
    
    if age_days > max_age_days:
        logger.warning(f"Data is {age_days} days old (threshold: {max_age_days} days)")
        return False
    else:
        logger.info(f"Data is fresh ({age_days} days old)")
        return True


def create_flow_summary_table(results: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Create summary table for flow results.
    
    Converts flow results into a table format suitable for Prefect artifacts.
    
    Args:
        results: Flow results dictionary
    
    Returns:
        List of dictionaries (table rows)
    """
    table = []
    
    for key, value in results.items():
        if not isinstance(value, (dict, list, pd.DataFrame)):
            table.append({
                'metric': key,
                'value': str(value)
            })
    
    return table


def format_duration(seconds: float) -> str:
    """
    Format duration in human-readable format.
    
    Args:
        seconds: Duration in seconds
    
    Returns:
        Formatted string (e.g., "2h 15m 30s")
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    parts = []
    if hours > 0:
        parts.append(f"{hours}h")
    if minutes > 0:
        parts.append(f"{minutes}m")
    if secs > 0 or not parts:
        parts.append(f"{secs}s")
    
    return " ".join(parts)
