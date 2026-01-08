"""
Model promotion logic for MLflow Model Registry.
"""

import logging
from typing import List, Optional

from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

# Configure module logger
logger = logging.getLogger(__name__)


def should_promote_model(
    test_mae: float,
    threshold: float,
    baseline_mae: Optional[float] = None
) -> bool:
    """
    Determine if a model should be promoted to production.
    Promotes if test_mae < threshold OR test_mae < baseline_mae.
    """
    logger.info("Evaluating model for promotion")
    logger.info(f"Test MAE: {test_mae:.4f}, Threshold: {threshold:.4f}")
    
    # Check absolute threshold
    passes_threshold = test_mae < threshold
    
    if baseline_mae is not None:
        logger.info(f"Baseline MAE: {baseline_mae:.4f}")
        better_than_baseline = test_mae < baseline_mae
        
        if better_than_baseline:
            logger.info(f"Model is better than baseline (improvement: {baseline_mae - test_mae:.4f})")
            return True
        else:
            logger.warning(f"Model is worse than baseline (degradation: {test_mae - baseline_mae:.4f})")
    
    if passes_threshold:
        logger.info(f"Model meets quality threshold ({test_mae:.4f} < {threshold:.4f})")
        return True
    else:
        logger.warning(f"Model does not meet quality threshold ({test_mae:.4f} >= {threshold:.4f})")
        return False


def promote_to_production(
    model_name: str,
    version: str,
    archive_existing: bool = True
) -> None:
    """
    Promote a model version to Production stage.
    
    Transitions the specified model version to "Production" stage.
    Optionally archives any existing production models.
    
    Args:
        model_name: Name of registered model
        version: Version number to promote
        archive_existing: If True, archive existing production models before promoting
    
    Raises:
        MlflowException: If promotion fails
    """
    logger.info(f"Promoting model to production: {model_name} version {version}")
    
    try:
        client = MlflowClient()
        
        # Archive existing production models if requested
        if archive_existing:
            existing_production = client.get_latest_versions(
                name=model_name,
                stages=["Production"]
            )
            
            for model_version in existing_production:
                old_version = model_version.version
                logger.info(f"Archiving existing production model: version {old_version}")
                
                client.transition_model_version_stage(
                    name=model_name,
                    version=old_version,
                    stage="Archived",
                    archive_existing_versions=False
                )
                
                logger.info(f"Version {old_version} archived")
        
        # Promote new model to production
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage="Production",
            archive_existing_versions=False
        )
        
        logger.info(f"Model promoted to Production: {model_name} v{version}")
        
    except MlflowException as e:
        logger.error(f"Failed to promote model: {e}")
        raise


def archive_old_models(
    model_name: str,
    keep_n: int = 52,
    stages_to_archive: Optional[List[str]] = None
) -> int:
    """
    Archive old model versions, keeping only the most recent N versions.
    
    Useful for maintaining a clean model registry and controlling storage costs.
    Default keeps 52 versions (1 year of weekly training).
    
    Args:
        model_name: Name of registered model
        keep_n: Number of recent versions to keep (default: 52)
        stages_to_archive: List of stages to archive (default: ["None", "Staging"])
    
    Returns:
        Number of versions archived
    """
    if stages_to_archive is None:
        stages_to_archive = ["None", "Staging"]
    
    logger.info(f"Archiving old model versions for: {model_name}")
    logger.info(f"Keeping most recent {keep_n} versions")
    
    try:
        client = MlflowClient()
        
        # Get all versions of the model
        all_versions = client.search_model_versions(f"name='{model_name}'")
        
        # Sort by version number (descending) to get newest first
        sorted_versions = sorted(
            all_versions,
            key=lambda v: int(v.version),
            reverse=True
        )
        
        logger.info(f"Found {len(sorted_versions)} total versions")
        
        archived_count = 0
        
        for i, model_version in enumerate(sorted_versions):
            version_num = model_version.version
            current_stage = model_version.current_stage
            
            # Skip if not in archivable stages
            if current_stage not in stages_to_archive:
                logger.debug(f"Version {version_num}: {current_stage} - skipping (protected stage)")
                continue
            
            # Archive if beyond keep_n threshold
            if i >= keep_n:
                logger.info(f"Archiving version {version_num} (rank {i+1}, stage: {current_stage})")
                
                client.transition_model_version_stage(
                    name=model_name,
                    version=version_num,
                    stage="Archived"
                )
                
                archived_count += 1
            else:
                logger.debug(f"Version {version_num}: {current_stage} - keeping (rank {i+1})")
        
        if archived_count > 0:
            logger.info(f"Archived {archived_count} old model versions")
        else:
            logger.info("No models to archive")
        
        return archived_count
        
    except MlflowException as e:
        logger.error(f"Failed to archive old models: {e}")
        raise


def get_production_model_mae(model_name: str) -> Optional[float]:
    """
    Get MAE of current production model.
    
    Retrieves the MAE tag from the production model version.
    Useful for comparing new models against the current baseline.
    
    Args:
        model_name: Name of registered model
    
    Returns:
        MAE of production model, or None if not found
    """
    logger.debug(f"Retrieving production model MAE for: {model_name}")
    
    try:
        client = MlflowClient()
        
        # Get production version
        production_versions = client.get_latest_versions(
            name=model_name,
            stages=["Production"]
        )
        
        if not production_versions:
            logger.info("No production model found")
            return None
        
        production_version = production_versions[0]
        
        # Try to get MAE from tags
        mae_tag = production_version.tags.get('mae') or production_version.tags.get('test_mae')
        
        if mae_tag:
            mae = float(mae_tag)
            logger.debug(f"Production model MAE: {mae:.4f}")
            return mae
        else:
            logger.warning(f"MAE tag not found for production model (version {production_version.version})")
            return None
            
    except (MlflowException, ValueError) as e:
        logger.warning(f"Could not retrieve production model MAE: {e}")
        return None


def delete_model_version(model_name: str, version: str) -> None:
    """
    Permanently delete a model version.
    
    WARNING: This is irreversible. Use with caution.
    Typically, archiving is preferred over deletion.
    
    Args:
        model_name: Name of registered model
        version: Version number to delete
    
    Raises:
        MlflowException: If deletion fails
    """
    logger.warning(f"Deleting model version: {model_name} v{version}")
    
    try:
        client = MlflowClient()
        client.delete_model_version(name=model_name, version=version)
        
        logger.info(f"Model version deleted: {model_name} v{version}")
        
    except MlflowException as e:
        logger.error(f"Failed to delete model version: {e}")
        raise


def add_model_description(
    model_name: str,
    version: str,
    description: str
) -> None:
    """
    Add or update description for a model version.
    
    Args:
        model_name: Name of registered model
        version: Version number
        description: Description text
    """
    logger.debug(f"Adding description to {model_name} v{version}")
    
    try:
        client = MlflowClient()
        client.update_model_version(
            name=model_name,
            version=version,
            description=description
        )
        
        logger.info(f"Description added to {model_name} v{version}")
        
    except MlflowException as e:
        logger.error(f"Failed to add description: {e}")
        raise
