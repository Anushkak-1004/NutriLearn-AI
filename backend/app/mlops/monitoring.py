"""
MLOps Monitoring for NutriLearn AI
Provides monitoring, drift detection, and system health checks.

MLOps Concepts:
- Model Monitoring: Track model performance in production
- Data Drift: Detect when input data distribution changes
- System Health: Monitor API performance and errors
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any
from collections import defaultdict, Counter

from .tracker import get_experiment_runs
from ..database import meal_logs

logger = logging.getLogger(__name__)


def get_prediction_stats() -> Dict[str, Any]:
    """
    Get comprehensive prediction statistics.
    
    Returns:
        dict: Statistics including counts by food, day, and user
        
    MLOps Note:
    Understanding prediction patterns helps:
    - Identify popular foods (may need better training data)
    - Detect unusual usage patterns
    - Plan infrastructure scaling
    """
    try:
        runs = get_experiment_runs(limit=1000, run_type="prediction")
        
        # Initialize counters
        by_food = Counter()
        by_day = defaultdict(int)
        by_user = Counter()
        
        for run in runs:
            # Count by food
            if "food_name" in run.data.params:
                food = run.data.params["food_name"]
                by_food[food] += 1
            
            # Count by day
            day = datetime.fromtimestamp(run.info.start_time / 1000).date()
            by_day[str(day)] += 1
            
            # Count by user
            if "user_id" in run.data.params:
                user = run.data.params["user_id"]
                by_user[user] += 1
        
        stats = {
            "total_predictions": len(runs),
            "by_food": dict(by_food.most_common(10)),  # Top 10 foods
            "by_day": dict(sorted(by_day.items())),
            "unique_users": len(by_user),
            "predictions_per_user": {
                "min": min(by_user.values()) if by_user else 0,
                "max": max(by_user.values()) if by_user else 0,
                "avg": sum(by_user.values()) / len(by_user) if by_user else 0
            }
        }
        
        logger.info(f"Generated prediction stats: {stats['total_predictions']} total predictions")
        
        return stats
        
    except Exception as e:
        logger.error(f"Failed to get prediction stats: {str(e)}")
        return {}


def get_confidence_distribution(bins: int = 10) -> Dict[str, Any]:
    """
    Get distribution of confidence scores.
    
    Args:
        bins: Number of bins for histogram
        
    Returns:
        dict: Histogram data with bins and counts
        
    MLOps Note:
    Confidence distribution reveals:
    - Model certainty levels
    - Potential issues (too many low confidence predictions)
    - Need for model retraining
    """
    try:
        runs = get_experiment_runs(limit=1000, run_type="prediction")
        
        confidences = []
        for run in runs:
            if "confidence" in run.data.metrics:
                confidences.append(run.data.metrics["confidence"])
        
        if not confidences:
            return {"bins": [], "counts": [], "total": 0}
        
        # Create histogram
        import numpy as np
        counts, bin_edges = np.histogram(confidences, bins=bins, range=(0, 1))
        
        # Format bins as ranges
        bin_labels = [f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}" for i in range(len(bin_edges)-1)]
        
        distribution = {
            "bins": bin_labels,
            "counts": counts.tolist(),
            "total": len(confidences),
            "statistics": {
                "mean": float(np.mean(confidences)),
                "median": float(np.median(confidences)),
                "std": float(np.std(confidences)),
                "min": float(np.min(confidences)),
                "max": float(np.max(confidences))
            }
        }
        
        logger.info(f"Generated confidence distribution with {len(confidences)} samples")
        
        return distribution
        
    except Exception as e:
        logger.error(f"Failed to get confidence distribution: {str(e)}")
        return {}


def get_model_performance() -> Dict[str, Any]:
    """
    Get model performance metrics over time.
    
    Returns:
        dict: Performance trends and current metrics
        
    MLOps Note:
    Tracking performance over time helps:
    - Detect model degradation
    - Validate improvements from retraining
    - Set performance baselines
    """
    try:
        # Get evaluation runs
        eval_runs = get_experiment_runs(limit=50, run_type="model_evaluation")
        
        if not eval_runs:
            return {
                "current": {},
                "history": [],
                "trend": "no_data"
            }
        
        # Extract metrics over time
        history = []
        for run in eval_runs:
            metrics = run.data.metrics
            timestamp = datetime.fromtimestamp(run.info.start_time / 1000)
            
            history.append({
                "timestamp": timestamp.isoformat(),
                "accuracy": metrics.get("accuracy", 0),
                "precision": metrics.get("precision", 0),
                "recall": metrics.get("recall", 0),
                "f1_score": metrics.get("f1_score", 0),
                "model_version": run.data.params.get("model_version", "unknown")
            })
        
        # Sort by timestamp
        history.sort(key=lambda x: x["timestamp"])
        
        # Get current (latest) metrics
        current = history[-1] if history else {}
        
        # Determine trend (simple: compare first and last)
        if len(history) >= 2:
            first_acc = history[0].get("accuracy", 0)
            last_acc = history[-1].get("accuracy", 0)
            
            if last_acc > first_acc * 1.05:  # 5% improvement
                trend = "improving"
            elif last_acc < first_acc * 0.95:  # 5% degradation
                trend = "degrading"
            else:
                trend = "stable"
        else:
            trend = "insufficient_data"
        
        performance = {
            "current": current,
            "history": history,
            "trend": trend,
            "total_evaluations": len(history)
        }
        
        logger.info(f"Generated model performance report: {trend} trend")
        
        return performance
        
    except Exception as e:
        logger.error(f"Failed to get model performance: {str(e)}")
        return {}


def detect_drift(window_days: int = 7, threshold: float = 0.15) -> Dict[str, Any]:
    """
    Detect data drift by comparing recent predictions to baseline.
    
    Args:
        window_days: Number of days to compare
        threshold: Drift threshold (0-1)
        
    Returns:
        dict: Drift detection results
        
    MLOps Note:
    Data drift occurs when:
    - Input data distribution changes
    - User behavior shifts
    - New food types appear
    
    Drift detection helps decide when to retrain the model.
    """
    try:
        # Get recent runs
        all_runs = get_experiment_runs(limit=1000, run_type="prediction")
        
        if len(all_runs) < 20:
            return {
                "drift_detected": False,
                "reason": "insufficient_data",
                "confidence": 0
            }
        
        # Split into recent and baseline
        cutoff_date = datetime.utcnow() - timedelta(days=window_days)
        
        recent_foods = Counter()
        baseline_foods = Counter()
        
        for run in all_runs:
            timestamp = datetime.fromtimestamp(run.info.start_time / 1000)
            food = run.data.params.get("food_name", "unknown")
            
            if timestamp >= cutoff_date:
                recent_foods[food] += 1
            else:
                baseline_foods[food] += 1
        
        # Calculate distribution difference (simple approach)
        # In production, use statistical tests like KS test or Chi-square
        
        if not baseline_foods:
            return {
                "drift_detected": False,
                "reason": "no_baseline_data",
                "confidence": 0
            }
        
        # Calculate percentage difference in top foods
        total_recent = sum(recent_foods.values())
        total_baseline = sum(baseline_foods.values())
        
        if total_recent == 0:
            return {
                "drift_detected": False,
                "reason": "no_recent_data",
                "confidence": 0
            }
        
        # Compare distributions
        max_diff = 0
        for food in set(list(recent_foods.keys()) + list(baseline_foods.keys())):
            recent_pct = recent_foods.get(food, 0) / total_recent
            baseline_pct = baseline_foods.get(food, 0) / total_baseline
            diff = abs(recent_pct - baseline_pct)
            max_diff = max(max_diff, diff)
        
        drift_detected = max_diff > threshold
        
        result = {
            "drift_detected": drift_detected,
            "drift_score": float(max_diff),
            "threshold": threshold,
            "window_days": window_days,
            "recent_predictions": total_recent,
            "baseline_predictions": total_baseline,
            "recommendation": "Consider retraining model" if drift_detected else "Model is stable",
            "top_recent_foods": dict(recent_foods.most_common(5)),
            "top_baseline_foods": dict(baseline_foods.most_common(5))
        }
        
        logger.info(f"Drift detection: {'DRIFT DETECTED' if drift_detected else 'No drift'} (score: {max_diff:.3f})")
        
        return result
        
    except Exception as e:
        logger.error(f"Failed to detect drift: {str(e)}")
        return {"drift_detected": False, "error": str(e)}


def get_system_health() -> Dict[str, Any]:
    """
    Get system health metrics.
    
    Returns:
        dict: Health metrics including response times and error rates
        
    MLOps Note:
    System health monitoring ensures:
    - API is responsive
    - Error rates are acceptable
    - Resources are adequate
    """
    try:
        # Get recent prediction runs
        runs = get_experiment_runs(limit=100, run_type="prediction")
        
        if not runs:
            return {
                "status": "unknown",
                "message": "No recent predictions"
            }
        
        # Calculate metrics
        processing_times = []
        error_count = 0
        
        for run in runs:
            # Get processing time
            if "processing_time_seconds" in run.data.metrics:
                processing_times.append(run.data.metrics["processing_time_seconds"])
            
            # Check for errors (runs with very low confidence might indicate errors)
            if "confidence" in run.data.metrics:
                if run.data.metrics["confidence"] < 0.5:
                    error_count += 1
        
        # Calculate statistics
        import numpy as np
        
        avg_response_time = float(np.mean(processing_times)) if processing_times else 0
        p95_response_time = float(np.percentile(processing_times, 95)) if processing_times else 0
        error_rate = (error_count / len(runs)) * 100 if runs else 0
        
        # Determine health status
        if error_rate > 10 or avg_response_time > 5:
            status = "unhealthy"
        elif error_rate > 5 or avg_response_time > 2:
            status = "degraded"
        else:
            status = "healthy"
        
        health = {
            "status": status,
            "metrics": {
                "avg_response_time_seconds": round(avg_response_time, 3),
                "p95_response_time_seconds": round(p95_response_time, 3),
                "error_rate_percent": round(error_rate, 2),
                "total_requests": len(runs),
                "error_count": error_count
            },
            "timestamp": datetime.utcnow().isoformat(),
            "recommendations": []
        }
        
        # Add recommendations
        if error_rate > 5:
            health["recommendations"].append("High error rate detected - investigate model quality")
        if avg_response_time > 2:
            health["recommendations"].append("Slow response times - consider model optimization")
        
        logger.info(f"System health: {status} (error rate: {error_rate:.1f}%, avg response: {avg_response_time:.3f}s)")
        
        return health
        
    except Exception as e:
        logger.error(f"Failed to get system health: {str(e)}")
        return {"status": "error", "message": str(e)}
