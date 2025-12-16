# 🚀 MLOps Quick Start Guide

## Start Everything in 3 Steps

### 1. Start Backend (with MLOps)

```bash
cd backend
python -m uvicorn app.main:app --reload
```

**Expected Output**:
```
✓ ML model loaded successfully
✓ Using in-memory storage for development
✓ MLflow initialized successfully
  Experiment ID: 0
  Tracking URI: file:./mlruns
  MLflow UI: Run 'mlflow ui' and visit http://localhost:5000
```

### 2. Start Frontend

```bash
cd frontend
npm run dev
```

### 3. Access MLOps Dashboard

Open browser: **http://localhost:5173/mlops**

## Test the MLOps System

### Make Some Predictions

1. Go to http://localhost:5173/analyze
2. Upload 5-10 food images
3. Log the meals

### View MLOps Dashboard

1. Go to http://localhost:5173/mlops
2. Click "Refresh" button
3. See your predictions logged!

### View MLflow UI (Optional)

```bash
mlflow ui
```

Visit: http://localhost:5000

## What You'll See

### MLOps Dashboard Shows:

✅ **Total Predictions** - Count of all predictions  
✅ **Average Confidence** - Model certainty  
✅ **System Health** - Response times, error rates  
✅ **Drift Detection** - Data distribution changes  
✅ **Recent Predictions** - Last 10 predictions  
✅ **Top Foods** - Most predicted items  
✅ **Confidence Distribution** - Histogram of scores  

### MLflow UI Shows:

✅ **Experiments** - nutrilearn-food-recognition  
✅ **Runs** - Each prediction logged  
✅ **Metrics** - Confidence, processing time  
✅ **Parameters** - Food name, user ID  
✅ **Charts** - Metric trends  

## API Endpoints

Test with curl or Postman:

```bash
# Get aggregated metrics
curl http://localhost:8000/api/v1/mlops/metrics

# Get recent runs
curl http://localhost:8000/api/v1/mlops/runs?limit=10

# Get system health
curl http://localhost:8000/api/v1/mlops/monitoring/health

# Get drift detection
curl http://localhost:8000/api/v1/mlops/monitoring/drift
```

## Troubleshooting

### MLflow Not Initializing?

Check logs for errors. MLflow will continue without blocking if it fails.

### Dashboard Shows No Data?

1. Make some predictions first
2. Click "Refresh" button
3. Check browser console for errors

### Backend Not Starting?

```bash
# Install dependencies
pip install -r requirements.txt

# Check Python version (need 3.9+)
python --version
```

## Key Files

- **Backend MLOps**: `backend/app/mlops/`
- **MLOps API**: `backend/app/api/mlops_routes.py`
- **Dashboard**: `frontend/src/pages/MLOpsDashboard.jsx`
- **Documentation**: `backend/MLOPS_GUIDE.md`

## Interview Demo Script

1. **Show Dashboard**: "This is our MLOps monitoring dashboard"
2. **Make Prediction**: Upload food image, show it logs to MLflow
3. **Explain Metrics**: "We track confidence, response time, error rates"
4. **Show Drift**: "This detects when data distribution changes"
5. **Show MLflow UI**: "All data is stored in MLflow for analysis"

## Quick Commands

```bash
# Start backend
cd backend && python -m uvicorn app.main:app --reload

# Start frontend
cd frontend && npm run dev

# Start MLflow UI
mlflow ui

# Run API tests
cd backend && python test_api.py

# Check MLflow data
ls -la mlruns/
```

## Success Indicators

✅ Backend starts without errors  
✅ MLflow initialized message appears  
✅ Frontend loads MLOps dashboard  
✅ Predictions appear in dashboard  
✅ Metrics update on refresh  
✅ No console errors  

---

**That's it! Your MLOps system is running!** 🎉

For detailed information, see:
- `MLOPS_COMPLETE_SUMMARY.md` - Full overview
- `backend/MLOPS_GUIDE.md` - Comprehensive guide
- `backend/API_DOCUMENTATION.md` - API reference
