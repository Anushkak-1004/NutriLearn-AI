"""
NutriLearn AI - Backend API
FastAPI application for food recognition and nutrition education.
"""

import sys
import platform
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from datetime import datetime
import logging

from .api.routes import router
from .api.mlops_routes import router as mlops_router
from .api.auth_routes import router as auth_router
from .ml.predictor import load_model
from .database import init_supabase_client
from .mlops.mlflow_config import initialize_mlflow, get_tracking_uri

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI application
app = FastAPI(
    title="NutriLearn AI API",
    description="Food Recognition & Nutrition Education Platform - A production-ready MLOps project",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    contact={
        "name": "NutriLearn AI Team",
        "email": "support@nutrilearn.ai"
    },
    license_info={
        "name": "MIT License"
    }
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",  # Frontend dev server
        "http://localhost:3000",  # Alternative frontend port
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Exception Handlers
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Handle validation errors with detailed error messages.
    
    Args:
        request: The incoming request
        exc: The validation exception
        
    Returns:
        JSONResponse with error details
    """
    logger.error(f"Validation error: {exc.errors()}")
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "status": "error",
            "message": "Validation error",
            "errors": exc.errors(),
            "timestamp": datetime.utcnow().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """
    Handle general exceptions with logging.
    
    Args:
        request: The incoming request
        exc: The exception
        
    Returns:
        JSONResponse with error message
    """
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "status": "error",
            "message": "Internal server error",
            "detail": str(exc),
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# Startup Event
@app.on_event("startup")
async def startup_event():
    """
    Initialize services on application startup.
    
    This function:
    - Loads the ML model for food recognition
    - Initializes database connections
    - Logs system information
    """
    logger.info("=" * 60)
    logger.info("Starting NutriLearn AI Backend")
    logger.info("=" * 60)
    
    # Log system information
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Platform: {platform.platform()}")
    
    # Initialize ML model
    logger.info("Loading ML model...")
    try:
        global model
        model = load_model()
        logger.info("✓ ML model loaded successfully (using mock predictions)")
    except Exception as e:
        logger.error(f"✗ Failed to load ML model: {str(e)}")
    
    # Initialize database
    logger.info("Initializing database connection...")
    try:
        db_client = init_supabase_client()
        if db_client:
            logger.info("✓ Database connected successfully")
        else:
            logger.info("✓ Using in-memory storage for development")
    except Exception as e:
        logger.error(f"✗ Database initialization failed: {str(e)}")
    
    # Initialize MLflow
    logger.info("Initializing MLflow experiment tracking...")
    try:
        experiment_id = initialize_mlflow()
        tracking_uri = get_tracking_uri()
        logger.info(f"✓ MLflow initialized successfully")
        logger.info(f"  Experiment ID: {experiment_id}")
        logger.info(f"  Tracking URI: {tracking_uri}")
        logger.info(f"  MLflow UI: Run 'mlflow ui' and visit http://localhost:5000")
    except Exception as e:
        logger.error(f"✗ MLflow initialization failed: {str(e)}")
        logger.warning("  Continuing without MLflow tracking")
    
    logger.info("=" * 60)
    logger.info("NutriLearn AI Backend is ready!")
    logger.info("API Documentation: http://localhost:8000/api/docs")
    logger.info("MLOps Dashboard: http://localhost:8000/api/docs#/mlops")
    logger.info("=" * 60)


# Shutdown Event
@app.on_event("shutdown")
async def shutdown_event():
    """
    Cleanup on application shutdown.
    """
    logger.info("Shutting down NutriLearn AI Backend...")
    # TODO: Add cleanup tasks (close DB connections, save state, etc.)
    logger.info("Shutdown complete")


# Include API routes
app.include_router(auth_router)
app.include_router(router)
app.include_router(mlops_router)


@app.get("/health")
async def health_check():
    """
    Enhanced health check endpoint with system information.
    
    Returns:
        dict: Detailed status information including:
            - Service status
            - Version
            - Timestamp
            - System info
            - Component status
    """
    return {
        "status": "healthy",
        "service": "NutriLearn AI Backend",
        "version": "1.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "system": {
            "python_version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "platform": platform.system()
        },
        "components": {
            "api": "operational",
            "ml_model": "operational (mock)",
            "database": "operational (in-memory)"
        }
    }


@app.get("/")
async def root():
    """
    Root endpoint with API information and available endpoints.
    
    Returns:
        dict: Welcome message, documentation links, and endpoint summary
    """
    return {
        "message": "Welcome to NutriLearn AI API",
        "description": "Food Recognition & Nutrition Education Platform",
        "version": "1.0.0",
        "documentation": {
            "swagger": "/api/docs",
            "redoc": "/api/redoc"
        },
        "endpoints": {
            "health": "/health",
            "predict": "POST /api/v1/predict",
            "log_meal": "POST /api/v1/meals/log",
            "analysis": "GET /api/v1/users/{user_id}/analysis",
            "stats": "GET /api/v1/users/{user_id}/stats",
            "meals": "GET /api/v1/users/{user_id}/meals",
            "complete_module": "POST /api/v1/modules/{module_id}/complete"
        },
        "status": "operational"
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("Starting NutriLearn AI Backend...")
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
