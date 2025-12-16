"""
MLflow Experiment Tracker for NutriLearn AI
Tracks predictions, model metrics, and experiments.

MLOps Concepts:
- Run: A single execution of ML code (e.g., one prediction or training session)
- Metrics: Quantitative measurements (accuracy, confidence, latency)
- Parameters: Configuration values (model version, hyperparameters)
- Artifacts: Output files (models, plots, data)
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
import mlflow
from mlflow.entities import Run

from .mlflow_config import get_mlflow_client, get_experiment_id

logger = logging.getLogger(__name__)


def log_prediction(
    food_name: str,
    confidence: float,
    user_id: str,
    processing_time: float,
    image_size: tuple = None,
    timestamp: datetime = None
) -> Optional[str]:
    """
    Log a single prediction to MLflow.
    
    Args:
        food_name: Predicted food item
        confidence: Prediction confidence (0-1)
        user_id: User who made the prediction
        processing_time: Time taken to process (seconds)
        image_size: Image dimensions (width, height)
        timestamp: When prediction was made
        
    Returns:
        str: Run ID if successful, None otherwise
        
    MLOps Note:
    Logging predictions helps with:
    - Monitoring model performance in production
    - Detecting data drift
    - Understanding user behavior
    - Debugging issues
    """
    try:
        timestamp = timestamp or datetime.utcnow()
        
        with mlflow.start_run(run_name=f"prediction_{timestamp.strftime('%Y%m%d_%H%M%S')}"):
            # Log parameters
            mlflow.log_param("food_name", food_name)
            mlflow.log_param("user_id", user_id)
            if image_size:
                mlflow.log_param("image_width", image_size[0])
                mlflow.log_param("image_height", image_size[1])
            
            # Log metrics
            mlflow.log_metric("confidence", confidence)
            mlflow.log_metric("processing_time_seconds", processing_time)
            
            # Log tags for filtering
            mlflow.set_tag("type", "prediction")
            mlflow.set_tag("food_category", food_name)
            mlflow.set_tag("timestamp", timestamp.isoformat())
            
            run = mlflow.active_run()
            run_id = run.info.run_id
            
            logger.info(f"Logged prediction: {food_name} (confidence: {confidence:.2%}, run_id: {run_id})")
            
            return run_id
            
    except Exception as e:
        logger.error(f"Failed to log prediction: {str(e)}")
        return None


def log_model_metrics(
    accuracy: float,
    precision: float,
    recall: float,
    f1_score: float,
    model_version: str = "v1.0"
) -> Optional[str]:
    """
    Log model performance metrics.
    
    Args:
        accuracy: Overall accuracy (0-1)
        precision: Precision score (0-1)
        recall: Recall score (0-1)
        f1_score: F1 score (0-1)
        model_version: Model version identifier
        
    Returns:
        str: Run ID if successful
        
    MLOps Note:
    These metrics help track model quality over time and compare versions.
    - Accuracy: % of correct predictions
    - Precision: % of positive predictions that were correct
    - Recall: % of actual positives that were identified
    - F1: Harmonic mean of precision and recall
    """
    try:
        with mlflow.start_run(run_name=f"model_evaluation_{model_version}"):
            # Log metrics
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("precision", precision)
            mlflow.log_metric("recall", recall)
            mlflow.log_metric("f1_score", f1_score)
            
            # Log model version
            mlflow.log_param("model_version", model_version)
            
            # Tag as model evaluation
            mlflow.set_tag("type", "model_evaluation")
            mlflow.set_tag("model_version", model_version)
            
            run = mlflow.active_run()
            run_id = run.info.run_id
            
            logger.info(f"Logged model metrics for {model_version} (run_id: {run_id})")
            
            return run_id
            
    except Exception as e:
        logger.error(f"Failed to log model metrics: {str(e)}")
        return None


def log_parameters(model_config: Dict[str, Any]) -> Optional[str]:
    """
    Log model hyperparameters and configuration.
    
    Args:
        model_config: Dictionary of configuration parameters
        
    Returns:
        str: Run ID if successful
        
    MLOps Note:
    Logging parameters enables:
    - Reproducibility: Recreate exact model configuration
    - Comparison: Compare different hyperparameter settings
    - Debugging: Understand what configuration produced results
    """
    try:
        with mlflow.start_run(run_name="model_configuration"):
            # Log all parameters
            for key, value in model_config.items():
                mlflow.log_param(key, value)
            
            mlflow.set_tag("type", "configuration")
            
            run = mlflow.active_run()
            run_id = run.info.run_id
            
            logger.info(f"Logged {len(model_config)} parameters (run_id: {run_id})")
            
            return run_id
            
    except Exception as e:
        logger.error(f"Failed to log parameters: {str(e)}")
        return None


def compare_models(model_v1_run_id: str, model_v2_run_id: str) -> Dict[str, Any]:
    """
    Compare metrics between two model versions.
    
    Args:
        model_v1_run_id: Run ID of first model
        model_v2_run_id: Run ID of second model
        
    Returns:
        dict: Comparison results with metrics from both models
        
    MLOps Note:
    Model comparison helps decide:
    - Which model to deploy to production
    - Whether new model is better than current
    - Trade-offs between different metrics
    """
    try:
        client = get_mlflow_client()
        
        # Get runs
        run1 = client.get_run(model_v1_run_id)
        run2 = client.get_run(model_v2_run_id)
        
        # Extract metrics
        metrics1 = run1.data.metrics
        metrics2 = run2.data.metrics
        
        # Calculate improvements
        comparison = {
            "model_v1": {
                "run_id": model_v1_run_id,
                "metrics": metrics1
            },
            "model_v2": {
                "run_id": model_v2_run_id,
                "metrics": metrics2
            },
            "improvements": {}
        }
        
        # Calculate percentage improvements
        for metric in metrics1.keys():
            if metric in metrics2:
                improvement = ((metrics2[metric] - metrics1[metric]) / metrics1[metric]) * 100
                comparison["improvements"][metric] = f"{improvement:+.2f}%"
        
        logger.info(f"Compared models: {model_v1_run_id} vs {model_v2_run_id}")
        
        return comparison
        
    except Exception as e:
        logger.error(f"Failed to compare models: {str(e)}")
        return {}


def get_experiment_runs(limit: int = 10, run_type: str = None) -> List[Run]:
    """
    Retrieve recent experiment runs.
    
    Args:
        limit: Maximum number of runs to return
        run_type: Filter by run type (prediction, model_evaluation, etc.)
        
    Returns:
        list: List of MLflow Run objects
        
    MLOps Note:
    Querying runs enables:
    - Dashboard visualization
    - Performance monitoring
    - Trend analysis
    """
    try:
        client = get_mlflow_client()
        experiment_id = get_experiment_id()
        
        # Build filter string
        filter_string = ""
        if run_type:
            filter_string = f"tags.type = '{run_type}'"
        
        # Search runs
        runs = client.search_runs(
            experiment_ids=[experiment_id],
            filter_string=filter_string,
            max_results=limit,
            order_by=["start_time DESC"]
        )
        
        logger.info(f"Retrieved {len(runs)} runs from experiment")
        
        return runs
        
    except Exception as e:
        logger.error(f"Failed to get experiment runs: {str(e)}")
        return []


def get_metrics_history(metric_name: str, limit: int = 100) -> List[Dict[str, Any]]:
    """
    Get history of a specific metric over time.
    
    Args:
        metric_name: Name of the metric (e.g., 'confidence', 'accuracy')
        limit: Maximum number of data points
        
    Returns:
        list: List of dicts with timestamp and value
        
    MLOps Note:
    Metric history helps identify:
    - Performance trends over time
    - Model degradation
    - Impact of changes
    """
    try:
        runs = get_experiment_runs(limit=limit)
        
        history = []
        for run in runs:
            if metric_name in run.data.metrics:
                history.append({
                    "timestamp": datetime.fromtimestamp(run.info.start_time / 1000),
                    "value": run.data.metrics[metric_name],
                    "run_id": run.info.run_id
                })
        
        # Sort by timestamp
        history.sort(key=lambda x: x["timestamp"])
        
        logger.info(f"Retrieved {len(history)} data points for metric: {metric_name}")
        
        return history
        
    except Exception as e:
        logger.error(f"Failed to get metrics history: {str(e)}")
        return []


def get_prediction_statistics() -> Dict[str, Any]:
    """
    Get aggregated statistics from prediction runs.
    
    Returns:
        dict: Statistics including count, avg confidence, etc.
        
    MLOps Note:
    Aggregated statistics provide high-level insights:
    - Total predictions made
    - Average confidence scores
    - Most predicted foods
    """
    try:
        runs = get_experiment_runs(limit=1000, run_type="prediction")
        
        if not runs:
            return {
                "total_predictions": 0,
                "avg_confidence": 0,
                "food_distribution": {}
            }
        
        confidences = []
        foods = {}
        
        for run in runs:
            # Get confidence
            if "confidence" in run.data.metrics:
                confidences.append(run.data.metrics["confidence"])
            
            # Count foods
            if "food_name" in run.data.params:
                food = run.data.params["food_name"]
                foods[food] = foods.get(food, 0) + 1
        
        stats = {
            "total_predictions": len(runs),
            "avg_confidence": sum(confidences) / len(confidences) if confidences else 0,
            "min_confidence": min(confidences) if confidences else 0,
            "max_confidence": max(confidences) if confidences else 0,
            "food_distribution": foods,
            "unique_foods": len(foods)
        }
        
        logger.info(f"Calculated prediction statistics: {stats['total_predictions']} predictions")
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get prediction statistics: {str(e)}")
        return {}
