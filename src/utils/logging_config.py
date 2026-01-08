"""
Logging configuration for the MLOps system.

This module sets up centralized logging with:
- Console and file handlers
- Configurable log levels
- Log rotation
- Consistent formatting
"""

import logging
import logging.handlers
import os
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = "logs/app.log",
    console: bool = True,
    file_logging: bool = True,
    max_bytes: int = 104857600,  # 100 MB
    backup_count: int = 5
) -> None:
    """
    Configure logging for the entire application.
    
    Sets up:
    - Root logger with specified level
    - Console handler (stdout)
    - File handler with rotation
    - Consistent formatting
    
    Args:
        level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
        log_file: Path to log file (None to disable file logging)
        console: Whether to log to console
        file_logging: Whether to log to file
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup log files to keep
    """
    # Create logs directory if it doesn't exist
    if log_file and file_logging:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Get root logger
    root_logger = logging.getLogger()
    
    # Clear any existing handlers
    root_logger.handlers.clear()
    
    # Set level
    log_level = getattr(logging, level.upper(), logging.INFO)
    root_logger.setLevel(log_level)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)
    
    # File handler with rotation
    if file_logging and log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # Set third-party library log levels
    logging.getLogger('mlflow').setLevel(logging.WARNING)
    logging.getLogger('prefect').setLevel(logging.WARNING)
    logging.getLogger('lightgbm').setLevel(logging.ERROR)
    logging.getLogger('optuna').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    
    root_logger.info("Logging configured successfully")
    root_logger.info(f"Log level: {level}")
    if console:
        root_logger.info("Console logging: enabled")
    if file_logging and log_file:
        root_logger.info(f"File logging: enabled ({log_file})")


def get_logger(name: str, level: Optional[str] = None) -> logging.Logger:
    """
    Get a configured logger for a module.
    
    Args:
        name: Logger name (typically __name__)
        level: Optional log level override
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    if level:
        log_level = getattr(logging, level.upper(), logging.INFO)
        logger.setLevel(log_level)
    
    return logger
