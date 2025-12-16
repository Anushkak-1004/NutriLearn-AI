"""
MLflow Configuration for NutriLearn AI
Handles MLflow setup, experiment management, and tracking URI configuration.

MLOps Concepts:
- Experiment Tracking: Record model training runs, parameters, and metrics
- Model Registry: Version control for ML models
- Artifact Storage: Store model files, plots, and other outputs
"""

import os
import logging
from pathlib import Path
import mlflow
from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)

# MLflow Configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
EXPERIMENT_NAME = "nutrilearn-food-recognition"
ARTIFACT_LOCATION = os.getenv("MLFLOW_ARTIFACT_LOCATION", "./mlartifacts")

# Initialize MLflow client
client = None


def initialize_mlflow():
    """
    Initialize MLflow tracking and create experiment if it doesn't exist.
    
    This function:
    1. Sets the tracking URI (where MLflow stores data)
    2. Creates or retrieves the experiment
    3. Initializes the MLflow client
    
    Returns:
        str: Experiment ID
        
    MLOps Note:
    The tracking URI can be:
    - Local file system: file:./mlruns
    - Remote server: http://mlflow-server:5000
    - Database: postgresql://user:pass@host/db
    """
    global client
    
    try:
        # Set tracking URI
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        logger.info(f"MLflow tracking URI set to: {MLFLOW_TRACKING_URI}")
        
        # Initialize client
        client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
        
        # Create or get experiment
        experiment_id = get_or_create_experiment(EXPERIMENT_NAME)
        
        # Set default experiment
        mlflow.set_experiment(EXPERIMENT_NAME)
        
        logger.info(f"MLflow initialized with experiment: {EXPERIMENT_NAME} (ID: {experiment_id})")
        
        return experiment_id
        
    except Exception as e:
        logger.error(f"Failed to initialize MLflow: {str(e)}")
        raise


def get_or_create_experiment(experiment_name: str) -> str:
    """
    Get existing experiment or create new one.
    
    Args:
        experiment_name: Name of the experiment
        
    Returns:
        str: Experiment ID
        
    MLOps Note:
    Experiments organize related runs. For example:
    - "nutrilearn-food-recognition" for production model
    - "nutrilearn-experiments" for research/testing
    """
    try:
        experiment = client.get_experiment_by_name(experiment_name)
        
        if experiment is None:
            # Create new experiment
            experiment_id = client.create_experiment(
                name=experiment_name,
                artifact_location=ARTIFACT_LOCATION
            )
            logger.info(f"Created new experiment: {experiment_name} (ID: {experiment_id})")
        else:
            experiment_id = experiment.experiment_id
            logger.info(f"Using existing experiment: {experiment_name} (ID: {experiment_id})")
        
        return experiment_id
        
    except Exception as e:
        logger.error(f"Error getting/creating experiment: {str(e)}")
        raise


def get_tracking_uri() -> str:
    """
    Get the current MLflow tracking URI.
    
    Returns:
        str: Tracking URI
    """
    return MLFLOW_TRACKING_URI


def get_mlflow_client() -> MlflowClient:
    """
    Get the MLflow client instance.
    
    Returns:
        MlflowClient: MLflow client for API operations
    """
    global client
    if client is None:
        initialize_mlflow()
    return client


def get_experiment_id() -> str:
    """
    Get the current experiment ID.
    
    Returns:
        str: Experiment ID
    """
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    return experiment.experiment_id if experiment else None


def cleanup_old_runs(days: int = 30):
    """
    Clean up old experiment runs (optional maintenance function).
    
    Args:
        days: Delete runs older than this many days
        
    MLOps Note:
    In production, implement retention policies to manage storage costs.
    Keep important runs (production models) and delete experimental runs.
    """
    # TODO: Implement cleanup logic
    logger.info(f"Cleanup of runs older than {days} days not yet implemented")
    pass
