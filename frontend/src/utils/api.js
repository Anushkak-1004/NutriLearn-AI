/**
 * API Client for NutriLearn AI Backend
 * Handles all HTTP requests to the FastAPI backend
 */

import axios from 'axios';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 30000, // 30 seconds
});

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method.toUpperCase()} ${config.url}`);
    return config;
  },
  (error) => {
    console.error('API Request Error:', error);
    return Promise.reject(error);
  }
);

// Response interceptor for error handling
api.interceptors.response.use(
  (response) => {
    console.log(`API Response: ${response.status} ${response.config.url}`);
    return response;
  },
  (error) => {
    console.error('API Response Error:', error.response?.data || error.message);
    return Promise.reject(error);
  }
);

/**
 * Predict food from image
 * @param {FormData} formData - Form data with image file
 * @returns {Promise<Object>} Prediction result
 */
export async function predictFood(formData) {
  const response = await api.post('/api/v1/predict', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
  return response.data;
}

/**
 * Log a meal
 * @param {string} userId - User identifier
 * @param {Object} mealData - Meal data (food_name, nutrition, meal_type)
 * @returns {Promise<Object>} Log response
 */
export async function logMeal(userId, mealData) {
  const response = await api.post('/api/v1/meals/log', {
    user_id: userId,
    ...mealData,
  });
  return response.data;
}

/**
 * Get user statistics
 * @param {string} userId - User identifier
 * @returns {Promise<Object>} User stats
 */
export async function getUserStats(userId) {
  const response = await api.get(`/api/v1/users/${userId}/stats`);
  return response.data;
}

/**
 * Get meal history
 * @param {string} userId - User identifier
 * @param {number} limit - Number of meals to fetch
 * @returns {Promise<Object>} Meal history with pagination
 */
export async function getMealHistory(userId, limit = 10) {
  const response = await api.get(`/api/v1/users/${userId}/meals`, {
    params: { limit },
  });
  return response.data;
}

/**
 * Get dietary analysis
 * @param {string} userId - User identifier
 * @param {number} days - Number of days to analyze
 * @returns {Promise<Object>} Analysis with patterns and recommendations
 */
export async function getDietaryAnalysis(userId, days = 7) {
  const response = await api.get(`/api/v1/users/${userId}/analysis`, {
    params: { days },
  });
  return response.data;
}

/**
 * Complete a learning module
 * @param {string} userId - User identifier
 * @param {string} moduleId - Module identifier
 * @param {number} quizScore - Quiz score (0-100)
 * @returns {Promise<Object>} Completion response with points
 */
export async function completeModule(userId, moduleId, quizScore) {
  const response = await api.post(`/api/v1/modules/${moduleId}/complete`, {
    user_id: userId,
    quiz_score: quizScore,
  });
  return response.data;
}

export default api;


/**
 * MLOps API Functions
 */

/**
 * Get MLflow experiments
 * @returns {Promise<Object>} Experiments data
 */
export async function getExperiments() {
  const response = await api.get('/api/v1/mlops/experiments');
  return response.data;
}

/**
 * Get experiment runs
 * @param {number} limit - Number of runs to fetch
 * @param {string} runType - Filter by run type
 * @returns {Promise<Object>} Runs data
 */
export async function getExperimentRuns(limit = 50, runType = null) {
  const response = await api.get('/api/v1/mlops/runs', {
    params: { limit, run_type: runType },
  });
  return response.data;
}

/**
 * Get aggregated metrics
 * @returns {Promise<Object>} Metrics data
 */
export async function getMLOpsMetrics() {
  const response = await api.get('/api/v1/mlops/metrics');
  return response.data;
}

/**
 * Get model versions
 * @returns {Promise<Object>} Model versions data
 */
export async function getModelVersions() {
  const response = await api.get('/api/v1/mlops/model-versions');
  return response.data;
}

/**
 * Get prediction monitoring data
 * @returns {Promise<Object>} Prediction statistics
 */
export async function getPredictionMonitoring() {
  const response = await api.get('/api/v1/mlops/monitoring/predictions');
  return response.data;
}

/**
 * Get confidence distribution
 * @param {number} bins - Number of histogram bins
 * @returns {Promise<Object>} Confidence distribution data
 */
export async function getConfidenceDistribution(bins = 10) {
  const response = await api.get('/api/v1/mlops/monitoring/confidence', {
    params: { bins },
  });
  return response.data;
}

/**
 * Get model performance trends
 * @returns {Promise<Object>} Performance data
 */
export async function getModelPerformance() {
  const response = await api.get('/api/v1/mlops/monitoring/performance');
  return response.data;
}

/**
 * Get drift detection results
 * @param {number} windowDays - Analysis window in days
 * @param {number} threshold - Drift threshold
 * @returns {Promise<Object>} Drift detection data
 */
export async function getDriftDetection(windowDays = 7, threshold = 0.15) {
  const response = await api.get('/api/v1/mlops/monitoring/drift', {
    params: { window_days: windowDays, threshold },
  });
  return response.data;
}

/**
 * Get system health
 * @returns {Promise<Object>} System health data
 */
export async function getSystemHealth() {
  const response = await api.get('/api/v1/mlops/monitoring/health');
  return response.data;
}
