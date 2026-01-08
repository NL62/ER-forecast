"""
MLflow utilities for experiment tracking and model registry.
"""

import logging
import os
from typing import Dict, Any, Optional
from pathlib import Path

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

# Configure module logger
logger = logging.getLogger(__name__)


def configure_mlflow_s3_backend() -> None:
    """
    Configure MLflow to use S3-compatible storage (MinIO) for artifacts.
    
    Sets up the necessary environment variables and configurations
    for MLflow to work with MinIO as an S3-compatible backend.
    """
    logger.info("Configuring MLflow S3 backend for MinIO")
    
    # Set S3 endpoint URL for MinIO
    s3_endpoint_url = os.getenv('MLFLOW_S3_ENDPOINT_URL', 'http://localhost:9000')
    os.environ['MLFLOW_S3_ENDPOINT_URL'] = s3_endpoint_url
    
    # Set AWS credentials for MinIO
    aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
    aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
    
    if aws_access_key_id and aws_secret_access_key:
        os.environ['AWS_ACCESS_KEY_ID'] = aws_access_key_id
        os.environ['AWS_SECRET_ACCESS_KEY'] = aws_secret_access_key
        logger.info("AWS credentials configured for MinIO")
    else:
        logger.warning("AWS credentials not found in environment variables")
    
    # Configure MLflow tracking URI if not already set
    mlflow_tracking_uri = os.getenv('MLFLOW_TRACKING_URI')
    if not mlflow_tracking_uri:
        mlflow_server_host = os.getenv('MLFLOW_SERVER_HOST', 'localhost')
        mlflow_server_port = os.getenv('MLFLOW_SERVER_PORT', '5011')
        tracking_uri = f"http://{mlflow_server_host}:{mlflow_server_port}"
        mlflow.set_tracking_uri(tracking_uri)
        logger.info(f"Set MLflow tracking URI: {tracking_uri}")
    
    logger.info("MLflow S3 backend configuration completed")


def log_experiment_params(params: Dict[str, Any]) -> None:
    """Log hyperparameters to MLflow (flattens nested dicts)."""
    logger.debug(f"Logging {len(params)} parameters to MLflow")
    
    # Flatten nested dictionaries
    flat_params = _flatten_dict(params)
    
    try:
        mlflow.log_params(flat_params)
        logger.info(f"Logged {len(flat_params)} parameters to MLflow")
    except MlflowException as e:
        logger.error(f"Failed to log parameters: {e}")
        raise


def log_experiment_metrics(metrics: Dict[str, float], step: Optional[int] = None) -> None:
    """Log evaluation metrics to MLflow."""
    logger.debug(f"Logging {len(metrics)} metrics to MLflow")
    
    try:
        if step is not None:
            for name, value in metrics.items():
                mlflow.log_metric(name, value, step=step)
        else:
            mlflow.log_metrics(metrics)
        
        logger.info(f"Logged metrics: {', '.join([f'{k}={v:.4f}' for k, v in metrics.items()])}")
    except MlflowException as e:
        logger.error(f"Failed to log metrics: {e}")
        raise


def log_model_artifact(
    model: Any,
    artifact_path: str,
    registered_model_name: Optional[str] = None
) -> str:
    """
    Log model artifact to MLflow.
    
    Saves the model to MLflow's artifact store and optionally registers it
    in the Model Registry.
    
    Args:
        model: Trained model object (should have save_model method or be picklable)
        artifact_path: Path within MLflow run to save model (e.g., "models/horizon_1")
        registered_model_name: If provided, register model with this name
    
    Returns:
        Model URI in MLflow
    """
    logger.info(f"Logging model artifact to path: {artifact_path}")
    
    try:
        # Log model using MLflow's pyfunc flavor for maximum compatibility
        model_info = mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=model,
            registered_model_name=registered_model_name
        )
        
        model_uri = model_info.model_uri
        logger.info(f"Model logged successfully: {model_uri}")
        
        if registered_model_name:
            logger.info(f"Model registered as: {registered_model_name}")
        
        return model_uri
        
    except Exception as e:
        logger.error(f"Failed to log model artifact: {e}")
        raise


def register_model(
    model_uri: str,
    model_name: str,
    tags: Optional[Dict[str, str]] = None
) -> str:
    """
    Register model in MLflow Model Registry.
    
    Creates a new model version in the Model Registry. If the model name
    doesn't exist, it creates it.
    
    Args:
        model_uri: URI of the model in MLflow (e.g., runs:/<run_id>/model)
        model_name: Name to register model under
        tags: Optional tags to add to model version
    
    Returns:
        Model version number as string
    
    Raises:
        MlflowException: If registration fails
    """
    logger.info(f"Registering model: {model_name}")
    logger.debug(f"Model URI: {model_uri}")
    
    try:
        client = MlflowClient()
        
        # Register the model
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )
        
        version_number = model_version.version
        logger.info(f"Model registered as version {version_number}")
        
        # Add tags if provided
        if tags:
            for key, value in tags.items():
                client.set_model_version_tag(
                    name=model_name,
                    version=version_number,
                    key=key,
                    value=str(value)
                )
            logger.debug(f"Added {len(tags)} tags to model version")
        
        return str(version_number)
        
    except MlflowException as e:
        logger.error(f"Failed to register model: {e}")
        raise


def get_latest_production_model(model_name: str) -> Any:
    """
    Load the latest production model from MLflow Model Registry.
    
    Fetches the model currently in the "Production" stage. If no production
    model exists, raises an exception.
    
    Args:
        model_name: Name of registered model
    
    Returns:
        Loaded model object
    
    Raises:
        MlflowException: If no production model found
    """
    logger.info(f"Loading production model: {model_name}")
    
    try:
        client = MlflowClient()
        
        # Get latest production version
        versions = client.get_latest_versions(
            name=model_name,
            stages=["Production"]
        )
        
        if not versions:
            raise MlflowException(
                f"No production model found for '{model_name}'. "
                "Please promote a model to Production stage first."
            )
        
        latest_version = versions[0]
        model_uri = f"models:/{model_name}/Production"
        
        logger.info(f"Loading model version {latest_version.version} from Production stage")
        
        # Load model as sklearn model (since we logged them as sklearn models)
        model_dict = mlflow.sklearn.load_model(model_uri)
        
        # Reconstruct LightGBMForecaster object from dictionary
        from src.models.lightgbm_model import LightGBMForecaster
        
        model = LightGBMForecaster(horizon=model_dict['horizon'])
        model.point_model = model_dict['point_model']
        model.lower_model = model_dict['lower_model']
        model.upper_model = model_dict['upper_model']
        model.feature_names = model_dict['feature_names']
        
        logger.info(f"Model loaded successfully: {model_name} v{latest_version.version}")
        
        return model
        
    except MlflowException as e:
        logger.error(f"Failed to load production model: {e}")
        raise


def get_model_by_version(model_name: str, version: str) -> Any:
    """
    Load a specific model version from MLflow Model Registry.
    
    Args:
        model_name: Name of registered model
        version: Version number as string
    
    Returns:
        Loaded model object
    """
    logger.info(f"Loading model: {model_name} version {version}")
    
    try:
        model_uri = f"models:/{model_name}/{version}"
        model = mlflow.pyfunc.load_model(model_uri)
        
        logger.info(f"Model loaded successfully: {model_name} v{version}")
        
        return model
        
    except MlflowException as e:
        logger.error(f"Failed to load model version: {e}")
        raise


def log_artifact_file(filepath: str, artifact_path: Optional[str] = None) -> None:
    """
    Log a file as an artifact in MLflow.
    
    Useful for logging plots, reports, or any other files.
    
    Args:
        filepath: Path to file to log
        artifact_path: Optional subdirectory in artifact store
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    logger.debug(f"Logging artifact: {filepath}")
    
    try:
        mlflow.log_artifact(str(filepath), artifact_path)
        logger.info(f"Artifact logged: {filepath.name}")
    except MlflowException as e:
        logger.error(f"Failed to log artifact: {e}")
        raise


def _flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten nested dictionary with dot notation.
    
    Internal helper function to handle nested parameter dictionaries.
    
    Args:
        d: Dictionary to flatten
        parent_key: Prefix for keys
        sep: Separator between nested keys
    
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    
    return dict(items)


def set_experiment(experiment_name: str) -> str:
    """
    Set or create MLflow experiment.
    
    If experiment doesn't exist, creates it.
    
    Args:
        experiment_name: Name of experiment
    
    Returns:
        Experiment ID
    """
    logger.info(f"Setting MLflow experiment: {experiment_name}")
    
    try:
        experiment = mlflow.set_experiment(experiment_name)
        experiment_id = experiment.experiment_id
        
        logger.info(f"Using experiment: {experiment_name} (ID: {experiment_id})")
        
        return experiment_id
        
    except MlflowException as e:
        logger.error(f"Failed to set experiment: {e}")
        raise
