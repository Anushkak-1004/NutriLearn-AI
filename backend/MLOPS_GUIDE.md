# NutriLearn AI - MLOps Implementation Guide

## 🎯 Overview

This guide covers the complete MLOps implementation for NutriLearn AI, including experiment tracking, model monitoring, drift detection, and performance analysis using MLflow.

## 📚 What is MLOps?

**MLOps** (Machine Learning Operations) is a set of practices that combines ML, DevOps, and Data Engineering to:
- Deploy and maintain ML models in production reliably and efficiently
- Monitor model performance over time
- Detect and respond to model degradation
- Version control models and experiments
- Automate ML workflows

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     NutriLearn AI MLOps                      │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐      ┌──────────────┐      ┌───────────┐ │
│  │   Prediction │─────▶│    MLflow    │─────▶│ Dashboard │ │
│  │   Endpoint   │      │   Tracking   │      │    UI     │ │
│  └──────────────┘      └──────────────┘      └───────────┘ │
│         │                      │                     │       │
│         │                      │                     │       │
│         ▼                      ▼                     ▼       │
│  ┌──────────────┐      ┌──────────────┐      ┌───────────┐ │
│  │     Log      │      │   Monitor    │      │  Analyze  │ │
│  │ Predictions  │      │ Performance  │      │   Drift   │ │
│  └──────────────┘      └──────────────┘      └───────────┘ │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## 📁 File Structure

```
backend/app/mlops/
├── __init__.py
├── mlflow_config.py      # MLflow setup and configuration
├── tracker.py            # Experiment tracking functions
└── monitoring.py         # Monitoring and drift detection

backend/app/api/
└── mlops_routes.py       # MLOps API endpoints

frontend/src/pages/
└── MLOpsDashboard.jsx    # MLOps visualization dashboard
```

## 🚀 Getting Started

### 1. Install Dependencies

MLflow is already included in `requirements.txt`:

```bash
cd backend
pip install -r requirements.txt
```

### 2. Start the Backend

```bash
python -m uvicorn app.main:app --reload
```

The backend will automatically:
- Initialize MLflow
- Create the experiment "nutrilearn-food-recognition"
- Start logging predictions

### 3. View MLflow UI (Optional)

```bash
mlflow ui
```

Visit http://localhost:5000 to see the MLflow tracking UI.

### 4. Access MLOps Dashboard

Start the frontend and navigate to http://localhost:5173/mlops

## 🔧 Components

### 1. MLflow Configuration (`mlflow_config.py`)

**Purpose**: Initialize and configure MLflow tracking

**Key Functions**:
- `initialize_mlflow()` - Set up MLflow with tracking URI
- `get_or_create_experiment()` - Create/retrieve experiment
- `get_mlflow_client()` - Get MLflow client instance

**Configuration**:
```python
MLFLOW_TRACKING_URI = "file:./mlruns"  # Local file system
EXPERIMENT_NAME = "nutrilearn-food-recognition"
```

**Interview Tip**: Explain that MLflow can use different backends:
- Local: `file:./mlruns`
- Remote: `http://mlflow-server:5000`
- Database: `postgresql://user:pass@host/db`

### 2. Experiment Tracker (`tracker.py`)

**Purpose**: Log predictions, metrics, and parameters

**Key Functions**:

#### `log_prediction()`
Logs each food prediction with:
- Food name and confidence
- Processing time
- Image size
- User ID
- Timestamp

```python
log_prediction(
    food_name="Chicken Biryani",
    confidence=0.95,
    user_id="user_123",
    processing_time=0.234,
    image_size=(224, 224)
)
```

#### `log_model_metrics()`
Tracks model performance:
- Accuracy
- Precision
- Recall
- F1 Score

```python
log_model_metrics(
    accuracy=0.92,
    precision=0.91,
    recall=0.93,
    f1_score=0.92,
    model_version="v2.0"
)
```

#### `compare_models()`
Compares two model versions:
```python
comparison = compare_models(
    model_v1_run_id="abc123",
    model_v2_run_id="def456"
)
```

**Interview Tip**: Explain that tracking predictions helps with:
- Monitoring model performance in production
- Detecting data drift
- Understanding user behavior
- Debugging issues

### 3. Monitoring (`monitoring.py`)

**Purpose**: Monitor system health and detect issues

**Key Functions**:

#### `get_prediction_stats()`
Returns:
- Predictions by food type
- Predictions by day
- Predictions by user
- Unique users count

#### `get_confidence_distribution()`
Creates histogram of confidence scores to identify:
- Model certainty levels
- Potential issues (too many low confidence predictions)

#### `detect_drift()`
Detects data drift by comparing:
- Recent predictions vs. baseline
- Food distribution changes
- Confidence score changes

**Drift Detection Logic**:
```python
if distribution_difference > threshold:
    return "Drift detected - consider retraining"
```

#### `get_system_health()`
Monitors:
- Average response time
- P95 response time
- Error rate
- Total requests

**Interview Tip**: Explain that drift detection helps decide when to retrain models.

### 4. MLOps API Routes (`mlops_routes.py`)

**11 Endpoints for MLOps**:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/mlops/experiments` | GET | List all experiments |
| `/api/v1/mlops/runs` | GET | Get experiment runs |
| `/api/v1/mlops/metrics` | GET | Aggregated metrics |
| `/api/v1/mlops/model-versions` | GET | List model versions |
| `/api/v1/mlops/register-model` | POST | Register new model |
| `/api/v1/mlops/monitoring/predictions` | GET | Prediction statistics |
| `/api/v1/mlops/monitoring/confidence` | GET | Confidence distribution |
| `/api/v1/mlops/monitoring/performance` | GET | Performance trends |
| `/api/v1/mlops/monitoring/drift` | GET | Drift detection |
| `/api/v1/mlops/monitoring/health` | GET | System health |
| `/api/v1/mlops/metrics/history` | GET | Metric history |

**Example Usage**:
```python
import requests

# Get aggregated metrics
response = requests.get("http://localhost:8000/api/v1/mlops/metrics")
metrics = response.json()

print(f"Total predictions: {metrics['metrics']['overview']['total_predictions']}")
print(f"Avg confidence: {metrics['metrics']['overview']['avg_confidence']}")
```

### 5. MLOps Dashboard (`MLOpsDashboard.jsx`)

**Features**:
- Overview metrics cards
- System health status
- Drift detection alerts
- Recent predictions table
- Top predicted foods chart
- Confidence distribution histogram
- Model version comparison
- Real-time refresh

**Components**:
- Metrics Cards: Total predictions, avg confidence, unique users
- Health Monitor: Response times, error rates
- Drift Detector: Data distribution changes
- Predictions Table: Recent predictions with details
- Charts: Food distribution, confidence histogram

## 📊 Key Metrics Explained

### Model Performance Metrics

**Accuracy**: Percentage of correct predictions
```
Accuracy = (TP + TN) / (TP + TN + FP + FN)
```

**Precision**: Of all positive predictions, how many were correct?
```
Precision = TP / (TP + FP)
```

**Recall**: Of all actual positives, how many did we identify?
```
Recall = TP / (TP + FN)
```

**F1 Score**: Harmonic mean of precision and recall
```
F1 = 2 * (Precision * Recall) / (Precision + Recall)
```

### System Health Metrics

**Response Time**: Time to process a prediction
- Target: < 2 seconds average
- Alert: > 5 seconds

**Error Rate**: Percentage of failed predictions
- Target: < 5%
- Alert: > 10%

**Confidence Score**: Model's certainty in prediction
- Good: > 90%
- Acceptable: 80-90%
- Low: < 80%

## 🔍 Monitoring Best Practices

### 1. Track Everything
- Log every prediction
- Record processing times
- Store confidence scores
- Track user interactions

### 2. Set Alerts
- High error rates (> 10%)
- Slow response times (> 5s)
- Low confidence scores (< 70%)
- Drift detected

### 3. Regular Reviews
- Daily: Check system health
- Weekly: Review prediction patterns
- Monthly: Analyze model performance
- Quarterly: Evaluate for retraining

### 4. Version Control
- Tag model versions
- Document changes
- Compare performance
- Maintain rollback capability

## 🎯 Interview Talking Points

### Why MLOps Matters

1. **Production Reliability**
   - Models degrade over time
   - Need continuous monitoring
   - Quick issue detection

2. **Performance Tracking**
   - Measure improvements
   - Compare model versions
   - Justify decisions

3. **Data Drift**
   - User behavior changes
   - New food types appear
   - Distribution shifts

4. **Compliance & Auditing**
   - Track all predictions
   - Explain decisions
   - Regulatory requirements

### MLOps vs DevOps

| Aspect | DevOps | MLOps |
|--------|--------|-------|
| Code | Deterministic | Probabilistic |
| Testing | Unit tests | Model validation |
| Deployment | Binary | Model + data |
| Monitoring | Uptime, errors | Accuracy, drift |
| Updates | Code changes | Retraining |

### Production Challenges

1. **Model Degradation**
   - Solution: Continuous monitoring
   - Action: Retrain when drift detected

2. **Scalability**
   - Solution: Model optimization
   - Action: Batch predictions, caching

3. **Versioning**
   - Solution: MLflow model registry
   - Action: Tag and track versions

4. **A/B Testing**
   - Solution: Gradual rollout
   - Action: Compare metrics

## 🚀 Advanced Features (Future)

### 1. Automated Retraining
```python
if drift_detected and performance_degraded:
    trigger_retraining_pipeline()
```

### 2. A/B Testing
```python
if user_id % 2 == 0:
    prediction = model_v1.predict(image)
else:
    prediction = model_v2.predict(image)
```

### 3. Model Registry
```python
mlflow.register_model(
    model_uri=f"runs:/{run_id}/model",
    name="food-classifier"
)
```

### 4. Automated Alerts
```python
if error_rate > 0.10:
    send_alert("High error rate detected")
```

## 📈 Success Metrics

Track these KPIs:

1. **Model Performance**
   - Accuracy > 90%
   - Confidence > 85%
   - F1 Score > 0.88

2. **System Performance**
   - Response time < 2s
   - Error rate < 5%
   - Uptime > 99%

3. **User Engagement**
   - Daily active users
   - Predictions per user
   - Feature adoption

## 🔧 Troubleshooting

### MLflow Not Starting
```bash
# Check if port 5000 is in use
lsof -i :5000

# Use different port
mlflow ui --port 5001
```

### Predictions Not Logging
- Check MLflow initialization in startup
- Verify tracking URI is accessible
- Check logs for errors

### Dashboard Not Loading Data
- Ensure backend is running
- Check API endpoints are accessible
- Verify CORS configuration

## 📚 Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [MLOps Principles](https://ml-ops.org/)
- [Google MLOps Guide](https://cloud.google.com/architecture/mlops-continuous-delivery-and-automation-pipelines-in-machine-learning)

## ✅ Checklist for Production

- [ ] MLflow tracking configured
- [ ] All predictions logged
- [ ] Monitoring dashboard deployed
- [ ] Alerts configured
- [ ] Model versioning implemented
- [ ] Drift detection active
- [ ] Performance baselines set
- [ ] Documentation complete
- [ ] Team trained on MLOps tools

---

**Built with ❤️ for production-ready ML systems**
