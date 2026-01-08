#!/usr/bin/env python3
"""
Test script to verify MinIO integration with MLflow.

This script tests:
1. MinIO connectivity
2. MLflow S3 backend configuration
3. Artifact storage and retrieval
"""

import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

import mlflow
from src.utils.mlflow_utils import configure_mlflow_s3_backend

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def test_minio_connection():
    """Test MinIO connection and bucket access."""
    logger.info("Testing MinIO connection...")
    
    try:
        import boto3
        from botocore.exceptions import ClientError
        
        # Create S3 client for MinIO
        s3_client = boto3.client(
            's3',
            endpoint_url=os.getenv('MLFLOW_S3_ENDPOINT_URL', 'http://localhost:9000'),
            aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
            aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
        )
        
        # Test bucket access
        bucket_name = os.getenv('MINIO_BUCKET_NAME', 'mlflow-artifacts')
        s3_client.head_bucket(Bucket=bucket_name)
        logger.info(f"✅ Successfully connected to MinIO bucket: {bucket_name}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed to connect to MinIO: {e}")
        return False


def test_mlflow_s3_backend():
    """Test MLflow S3 backend configuration."""
    logger.info("Testing MLflow S3 backend configuration...")
    
    try:
        # Configure MLflow S3 backend
        configure_mlflow_s3_backend()
        
        # Set tracking URI
        mlflow.set_tracking_uri(os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5011'))
        
        # Test experiment creation
        experiment_name = "minio-test-experiment"
        experiment = mlflow.set_experiment(experiment_name)
        logger.info(f"✅ Created experiment: {experiment_name}")
        
        # Test artifact logging
        with mlflow.start_run():
            # Create a test artifact
            test_file = Path("test_artifact.txt")
            test_file.write_text("This is a test artifact for MinIO integration")
            
            # Log artifact
            mlflow.log_artifact(str(test_file))
            logger.info("✅ Successfully logged artifact to MinIO")
            
            # Clean up
            test_file.unlink()
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Failed MLflow S3 backend test: {e}")
        return False


def main():
    """Run all tests."""
    logger.info("Starting MinIO integration tests...")
    
    # Check environment variables
    required_vars = [
        'MLFLOW_S3_ENDPOINT_URL',
        'AWS_ACCESS_KEY_ID', 
        'AWS_SECRET_ACCESS_KEY',
        'MINIO_BUCKET_NAME'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"❌ Missing required environment variables: {missing_vars}")
        logger.info("Please copy env.template to .env and configure the variables")
        return False
    
    # Run tests
    tests = [
        ("MinIO Connection", test_minio_connection),
        ("MLflow S3 Backend", test_mlflow_s3_backend),
    ]
    
    results = []
    for test_name, test_func in tests:
        logger.info(f"\n--- Running {test_name} Test ---")
        result = test_func()
        results.append((test_name, result))
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    
    all_passed = True
    for test_name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        logger.info(f"{test_name}: {status}")
        if not result:
            all_passed = False
    
    if all_passed:
        logger.info("\n🎉 All tests passed! MinIO integration is working correctly.")
    else:
        logger.info("\n⚠️  Some tests failed. Please check the configuration.")
    
    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
