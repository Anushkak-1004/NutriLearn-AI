"""
MLOps API Routes for NutriLearn AI
Endpoints for experiment tracking, monitoring, and model management.
"""

import logging
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import JSONResponse

from ..mlops.tracker import (
    get_experiment_runs, get_metrics_history, get_prediction_statistics,
    log_model_metrics
)
from ..mlops.monitoring import (
    get_prediction_stats, get_confidence_distribution,
    get_model_performance, detect_drift, get_system_health
)
from ..mlops.mlflow_config import get_tracking_uri, get_experiment_id

logger = logging.getLogger(__name__)

# Create MLOps router
router = APIRouter(prefix="/api/v1/mlops", tags=["mlops"])


@router.get("/experiments")
async def list_experiments():
    """
    List all MLflow experiments.
    
    Returns:
        JSON with experiment information
        
    MLOps Note:
    Experiments organize related runs. This endpoint helps navigate
    the experiment tracking system.
    """
    try:
        from ..mlops.mlflow_config import get_mlflow_client
        
        client = get_mlflow_client()
        experiments = client.search_experiments()
        
        result = []
        for exp in experiments:
            result.append({
                "experiment_id": exp.experiment_id,
                "name": exp.name,
                "artifact_location": exp.artifact_location,
                "lifecycle_stage": exp.lifecycle_stage,
                "tags": exp.tags
            })
        
        return JSONResponse(content={
            "status": "success",
            "experiments": result,
            "tracking_uri": get_tracking_uri()
        })
        
    except Exception as e:
        logger.error(f"Error listing experiments: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/runs")
async def get_runs(
    limit: int = Query(default=50, ge=1, le=500),
    run_type: Optional[str] = Query(default=None, description="Filter by run type")
):
    """
    Get recent experiment runs with pagination.
    
    Args:
        limit: Maximum number of runs to return
        run_type: Filter by type (prediction, model_evaluation, etc.)
        
    Returns:
        JSON with run information
        
    MLOps Note:
    Runs represent individual executions. This endpoint provides
    access to historical data for analysis and debugging.
    """
    try:
        runs = get_experiment_runs(limit=limit, run_type=run_type)
        
        result = []
        for run in runs:
            result.append({
                "run_id": run.info.run_id,
                "experiment_id": run.info.experiment_id,
                "status": run.info.status,
                "start_time": datetime.fromtimestamp(run.info.start_time / 1000).isoformat(),
                "end_time": datetime.fromtimestamp(run.info.end_time / 1000).isoformat() if run.info.end_time else None,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags
            })
        
        return JSONResponse(content={
            "status": "success",
            "runs": result,
            "total": len(result),
            "limit": limit
        })
        
    except Exception as e:
        logger.error(f"Error getting runs: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics")
async def get_aggregated_metrics():
    """
    Get aggregated metrics across all predictions.
    
    Returns:
        JSON with aggregated statistics
        
    MLOps Note:
    Aggregated metrics provide high-level insights:
    - Total predictions made
    - Average confidence scores
    - Predictions per day
    - Most common foods
    """
    try:
        # Get prediction statistics
        pred_stats = get_prediction_statistics()
        
        # Get prediction stats by day/food/user
        detailed_stats = get_prediction_stats()
        
        # Combine results
        metrics = {
            "overview": {
                "total_predictions": pred_stats.get("total_predictions", 0),
                "avg_confidence": round(pred_stats.get("avg_confidence", 0), 3),
                "unique_foods": pred_stats.get("unique_foods", 0),
                "unique_users": detailed_stats.get("unique_users", 0)
            },
            "confidence": {
                "avg": round(pred_stats.get("avg_confidence", 0), 3),
                "min": round(pred_stats.get("min_confidence", 0), 3),
                "max": round(pred_stats.get("max_confidence", 0), 3)
            },
            "top_foods": pred_stats.get("food_distribution", {}),
            "predictions_by_day": detailed_stats.get("by_day", {}),
            "timestamp": datetime.utcnow().isoformat()
        }
        
        return JSONResponse(content={
            "status": "success",
            "metrics": metrics
        })
        
    except Exception as e:
        logger.error(f"Error getting metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model-versions")
async def list_model_versions():
    """
    List model versions and their performance.
    
    Returns:
        JSON with model version information
        
    MLOps Note:
    Model versioning enables:
    - Tracking improvements over time
    - Rolling back to previous versions
    - A/B testing different models
    """
    try:
        # Get model evaluation runs
        eval_runs = get_experiment_runs(limit=20, run_type="model_evaluation")
        
        versions = []
        for run in eval_runs:
            version_info = {
                "run_id": run.info.run_id,
                "model_version": run.data.params.get("model_version", "unknown"),
                "timestamp": datetime.fromtimestamp(run.info.start_time / 1000).isoformat(),
                "metrics": {
                    "accuracy": run.data.metrics.get("accuracy", 0),
                    "precision": run.data.metrics.get("precision", 0),
                    "recall": run.data.metrics.get("recall", 0),
                    "f1_score": run.data.metrics.get("f1_score", 0)
                },
                "status": run.info.status
            }
            versions.append(version_info)
        
        return JSONResponse(content={
            "status": "success",
            "model_versions": versions,
            "total": len(versions)
        })
        
    except Exception as e:
        logger.error(f"Error listing model versions: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/register-model")
async def register_model(
    model_version: str,
    accuracy: float,
    precision: float,
    recall: float,
    f1_score: float
):
    """
    Register a new model version with performance metrics.
    
    Args:
        model_version: Version identifier (e.g., "v2.0")
        accuracy: Model accuracy (0-1)
        precision: Precision score (0-1)
        recall: Recall score (0-1)
        f1_score: F1 score (0-1)
        
    Returns:
        JSON with registration confirmation
        
    MLOps Note:
    Model registration creates a record of model performance
    for comparison and deployment decisions.
    """
    try:
        # Log model metrics
        run_id = log_model_metrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            model_version=model_version
        )
        
        if run_id:
            return JSONResponse(content={
                "status": "success",
                "message": f"Model {model_version} registered successfully",
                "run_id": run_id,
                "metrics": {
                    "accuracy": accuracy,
                    "precision": precision,
                    "recall": recall,
                    "f1_score": f1_score
                }
            })
        else:
            raise HTTPException(status_code=500, detail="Failed to register model")
        
    except Exception as e:
        logger.error(f"Error registering model: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitoring/predictions")
async def get_prediction_monitoring():
    """
    Get prediction monitoring data.
    
    Returns:
        JSON with prediction statistics and trends
    """
    try:
        stats = get_prediction_stats()
        
        return JSONResponse(content={
            "status": "success",
            "data": stats
        })
        
    except Exception as e:
        logger.error(f"Error getting prediction monitoring: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitoring/confidence")
async def get_confidence_monitoring(bins: int = Query(default=10, ge=5, le=50)):
    """
    Get confidence score distribution.
    
    Args:
        bins: Number of histogram bins
        
    Returns:
        JSON with confidence distribution data
    """
    try:
        distribution = get_confidence_distribution(bins=bins)
        
        return JSONResponse(content={
            "status": "success",
            "data": distribution
        })
        
    except Exception as e:
        logger.error(f"Error getting confidence monitoring: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitoring/performance")
async def get_performance_monitoring():
    """
    Get model performance trends.
    
    Returns:
        JSON with performance metrics over time
    """
    try:
        performance = get_model_performance()
        
        return JSONResponse(content={
            "status": "success",
            "data": performance
        })
        
    except Exception as e:
        logger.error(f"Error getting performance monitoring: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitoring/drift")
async def get_drift_detection(
    window_days: int = Query(default=7, ge=1, le=90),
    threshold: float = Query(default=0.15, ge=0.01, le=1.0)
):
    """
    Detect data drift in predictions.
    
    Args:
        window_days: Number of days to analyze
        threshold: Drift detection threshold
        
    Returns:
        JSON with drift detection results
        
    MLOps Note:
    Data drift detection helps identify when model retraining is needed.
    """
    try:
        drift_result = detect_drift(window_days=window_days, threshold=threshold)
        
        return JSONResponse(content={
            "status": "success",
            "data": drift_result
        })
        
    except Exception as e:
        logger.error(f"Error detecting drift: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/monitoring/health")
async def get_health_monitoring():
    """
    Get system health metrics.
    
    Returns:
        JSON with system health status
    """
    try:
        health = get_system_health()
        
        return JSONResponse(content={
            "status": "success",
            "data": health
        })
        
    except Exception as e:
        logger.error(f"Error getting system health: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/metrics/history")
async def get_metric_history(
    metric_name: str = Query(..., description="Metric name to retrieve"),
    limit: int = Query(default=100, ge=1, le=1000)
):
    """
    Get historical data for a specific metric.
    
    Args:
        metric_name: Name of the metric (e.g., 'confidence', 'accuracy')
        limit: Maximum number of data points
        
    Returns:
        JSON with metric history
    """
    try:
        history = get_metrics_history(metric_name=metric_name, limit=limit)
        
        return JSONResponse(content={
            "status": "success",
            "metric_name": metric_name,
            "data": history,
            "total_points": len(history)
        })
        
    except Exception as e:
        logger.error(f"Error getting metric history: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
