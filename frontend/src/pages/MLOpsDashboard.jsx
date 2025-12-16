import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  ArrowLeft, Activity, TrendingUp, Users, AlertTriangle, 
  RefreshCw, BarChart3, PieChart, Clock, CheckCircle, XCircle
} from 'lucide-react';
import {
  getMLOpsMetrics, getExperimentRuns, getPredictionMonitoring,
  getConfidenceDistribution, getModelPerformance, getDriftDetection,
  getSystemHealth, getModelVersions
} from '../utils/api';

function MLOpsDashboard() {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [refreshing, setRefreshing] = useState(false);
  
  // State for different data sections
  const [metrics, setMetrics] = useState(null);
  const [recentRuns, setRecentRuns] = useState([]);
  const [predictionStats, setPredictionStats] = useState(null);
  const [confidenceData, setConfidenceData] = useState(null);
  const [performance, setPerformance] = useState(null);
  const [drift, setDrift] = useState(null);
  const [health, setHealth] = useState(null);
  const [modelVersions, setModelVersions] = useState([]);

  const fetchAllData = async () => {
    try {
      const [
        metricsData,
        runsData,
        predStatsData,
        confData,
        perfData,
        driftData,
        healthData,
        versionsData
      ] = await Promise.all([
        getMLOpsMetrics(),
        getExperimentRuns(20, 'prediction'),
        getPredictionMonitoring(),
        getConfidenceDistribution(10),
        getModelPerformance(),
        getDriftDetection(),
        getSystemHealth(),
        getModelVersions()
      ]);

      setMetrics(metricsData.metrics);
      setRecentRuns(runsData.runs || []);
      setPredictionStats(predStatsData.data);
      setConfidenceData(confData.data);
      setPerformance(perfData.data);
      setDrift(driftData.data);
      setHealth(healthData.data);
      setModelVersions(versionsData.model_versions || []);
    } catch (error) {
      console.error('Error fetching MLOps data:', error);
    } finally {
      setLoading(false);
      setRefreshing(false);
    }
  };

  useEffect(() => {
    fetchAllData();
  }, []);

  const handleRefresh = () => {
    setRefreshing(true);
    fetchAllData();
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-purple-50 to-blue-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-purple-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading MLOps Dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-purple-50 to-blue-50 py-8 px-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="flex items-center justify-between mb-8">
          <div className="flex items-center space-x-4">
            <button
              onClick={() => navigate('/')}
              className="flex items-center space-x-2 text-gray-600 hover:text-gray-800 transition-colors"
            >
              <ArrowLeft className="w-5 h-5" />
              <span className="font-medium">Back</span>
            </button>
            <h1 className="text-4xl font-bold text-gray-800">MLOps Dashboard</h1>
          </div>
          
          <button
            onClick={handleRefresh}
            disabled={refreshing}
            className="flex items-center space-x-2 bg-purple-600 hover:bg-purple-700 text-white font-semibold py-2 px-6 rounded-xl transition-colors disabled:opacity-50"
          >
            <RefreshCw className={`w-5 h-5 ${refreshing ? 'animate-spin' : ''}`} />
            <span>Refresh</span>
          </button>
        </div>

        {/* Overview Metrics */}
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8">
          <div className="bg-white rounded-2xl shadow-lg p-6">
            <div className="flex items-center justify-between mb-2">
              <p className="text-gray-600 text-sm">Total Predictions</p>
              <Activity className="w-6 h-6 text-purple-600" />
            </div>
            <p className="text-3xl font-bold text-gray-800">
              {metrics?.overview?.total_predictions || 0}
            </p>
          </div>

          <div className="bg-white rounded-2xl shadow-lg p-6">
            <div className="flex items-center justify-between mb-2">
              <p className="text-gray-600 text-sm">Avg Confidence</p>
              <TrendingUp className="w-6 h-6 text-green-600" />
            </div>
            <p className="text-3xl font-bold text-gray-800">
              {((metrics?.overview?.avg_confidence || 0) * 100).toFixed(1)}%
            </p>
          </div>

          <div className="bg-white rounded-2xl shadow-lg p-6">
            <div className="flex items-center justify-between mb-2">
              <p className="text-gray-600 text-sm">Unique Users</p>
              <Users className="w-6 h-6 text-blue-600" />
            </div>
            <p className="text-3xl font-bold text-gray-800">
              {metrics?.overview?.unique_users || 0}
            </p>
          </div>

          <div className="bg-white rounded-2xl shadow-lg p-6">
            <div className="flex items-center justify-between mb-2">
              <p className="text-gray-600 text-sm">Unique Foods</p>
              <PieChart className="w-6 h-6 text-orange-600" />
            </div>
            <p className="text-3xl font-bold text-gray-800">
              {metrics?.overview?.unique_foods || 0}
            </p>
          </div>
        </div>

        {/* System Health & Drift Detection */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* System Health */}
          <div className="bg-white rounded-2xl shadow-lg p-6">
            <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center">
              <Activity className="w-6 h-6 mr-2 text-purple-600" />
              System Health
            </h2>
            
            {health && (
              <div>
                <div className={`inline-flex items-center space-x-2 px-4 py-2 rounded-full mb-4 ${
                  health.status === 'healthy' ? 'bg-green-100 text-green-700' :
                  health.status === 'degraded' ? 'bg-yellow-100 text-yellow-700' :
                  'bg-red-100 text-red-700'
                }`}>
                  {health.status === 'healthy' ? <CheckCircle className="w-5 h-5" /> : <AlertTriangle className="w-5 h-5" />}
                  <span className="font-bold capitalize">{health.status}</span>
                </div>

                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600">Avg Response Time</span>
                    <span className="font-bold text-gray-800">
                      {health.metrics?.avg_response_time_seconds?.toFixed(3)}s
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600">P95 Response Time</span>
                    <span className="font-bold text-gray-800">
                      {health.metrics?.p95_response_time_seconds?.toFixed(3)}s
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600">Error Rate</span>
                    <span className={`font-bold ${
                      health.metrics?.error_rate_percent > 5 ? 'text-red-600' : 'text-green-600'
                    }`}>
                      {health.metrics?.error_rate_percent?.toFixed(2)}%
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600">Total Requests</span>
                    <span className="font-bold text-gray-800">
                      {health.metrics?.total_requests || 0}
                    </span>
                  </div>
                </div>

                {health.recommendations && health.recommendations.length > 0 && (
                  <div className="mt-4 p-3 bg-yellow-50 border border-yellow-200 rounded-lg">
                    <p className="text-sm font-medium text-yellow-800 mb-1">Recommendations:</p>
                    {health.recommendations.map((rec, idx) => (
                      <p key={idx} className="text-sm text-yellow-700">• {rec}</p>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Drift Detection */}
          <div className="bg-white rounded-2xl shadow-lg p-6">
            <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center">
              <AlertTriangle className="w-6 h-6 mr-2 text-orange-600" />
              Data Drift Detection
            </h2>
            
            {drift && (
              <div>
                <div className={`inline-flex items-center space-x-2 px-4 py-2 rounded-full mb-4 ${
                  drift.drift_detected ? 'bg-red-100 text-red-700' : 'bg-green-100 text-green-700'
                }`}>
                  {drift.drift_detected ? <XCircle className="w-5 h-5" /> : <CheckCircle className="w-5 h-5" />}
                  <span className="font-bold">
                    {drift.drift_detected ? 'Drift Detected' : 'No Drift'}
                  </span>
                </div>

                <div className="space-y-3">
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600">Drift Score</span>
                    <span className={`font-bold ${
                      drift.drift_detected ? 'text-red-600' : 'text-green-600'
                    }`}>
                      {(drift.drift_score * 100).toFixed(2)}%
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600">Threshold</span>
                    <span className="font-bold text-gray-800">
                      {(drift.threshold * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="flex justify-between items-center">
                    <span className="text-gray-600">Analysis Window</span>
                    <span className="font-bold text-gray-800">
                      {drift.window_days} days
                    </span>
                  </div>
                </div>

                {drift.recommendation && (
                  <div className={`mt-4 p-3 rounded-lg ${
                    drift.drift_detected ? 'bg-red-50 border border-red-200' : 'bg-green-50 border border-green-200'
                  }`}>
                    <p className={`text-sm font-medium ${
                      drift.drift_detected ? 'text-red-800' : 'text-green-800'
                    }`}>
                      {drift.recommendation}
                    </p>
                  </div>
                )}
              </div>
            )}
          </div>
        </div>

        {/* Recent Predictions Table */}
        <div className="bg-white rounded-2xl shadow-lg p-6 mb-8">
          <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center">
            <Clock className="w-6 h-6 mr-2 text-blue-600" />
            Recent Predictions
          </h2>
          
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="text-left py-3 px-4 text-gray-600 font-semibold">Time</th>
                  <th className="text-left py-3 px-4 text-gray-600 font-semibold">Food</th>
                  <th className="text-left py-3 px-4 text-gray-600 font-semibold">Confidence</th>
                  <th className="text-left py-3 px-4 text-gray-600 font-semibold">Processing Time</th>
                  <th className="text-left py-3 px-4 text-gray-600 font-semibold">User</th>
                </tr>
              </thead>
              <tbody>
                {recentRuns.slice(0, 10).map((run, idx) => (
                  <tr key={idx} className="border-b border-gray-100 hover:bg-gray-50">
                    <td className="py-3 px-4 text-sm text-gray-700">
                      {new Date(run.start_time).toLocaleString()}
                    </td>
                    <td className="py-3 px-4 text-sm font-medium text-gray-800">
                      {run.params?.food_name || 'N/A'}
                    </td>
                    <td className="py-3 px-4">
                      <span className={`inline-block px-2 py-1 rounded-full text-xs font-bold ${
                        run.metrics?.confidence > 0.9 ? 'bg-green-100 text-green-700' :
                        run.metrics?.confidence > 0.8 ? 'bg-yellow-100 text-yellow-700' :
                        'bg-orange-100 text-orange-700'
                      }`}>
                        {((run.metrics?.confidence || 0) * 100).toFixed(1)}%
                      </span>
                    </td>
                    <td className="py-3 px-4 text-sm text-gray-700">
                      {run.metrics?.processing_time_seconds?.toFixed(3)}s
                    </td>
                    <td className="py-3 px-4 text-sm text-gray-600">
                      {run.params?.user_id || 'anonymous'}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Top Foods & Confidence Distribution */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Top Foods */}
          <div className="bg-white rounded-2xl shadow-lg p-6">
            <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center">
              <BarChart3 className="w-6 h-6 mr-2 text-emerald-600" />
              Top Predicted Foods
            </h2>
            
            {metrics?.top_foods && Object.keys(metrics.top_foods).length > 0 ? (
              <div className="space-y-3">
                {Object.entries(metrics.top_foods)
                  .sort(([, a], [, b]) => b - a)
                  .slice(0, 8)
                  .map(([food, count], idx) => {
                    const maxCount = Math.max(...Object.values(metrics.top_foods));
                    const percentage = (count / maxCount) * 100;
                    
                    return (
                      <div key={idx}>
                        <div className="flex justify-between items-center mb-1">
                          <span className="text-sm font-medium text-gray-700">{food}</span>
                          <span className="text-sm font-bold text-gray-800">{count}</span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-gradient-to-r from-emerald-500 to-green-600 h-2 rounded-full transition-all"
                            style={{ width: `${percentage}%` }}
                          ></div>
                        </div>
                      </div>
                    );
                  })}
              </div>
            ) : (
              <p className="text-gray-500 text-center py-8">No prediction data available</p>
            )}
          </div>

          {/* Confidence Distribution */}
          <div className="bg-white rounded-2xl shadow-lg p-6">
            <h2 className="text-2xl font-bold text-gray-800 mb-4 flex items-center">
              <TrendingUp className="w-6 h-6 mr-2 text-blue-600" />
              Confidence Distribution
            </h2>
            
            {confidenceData && confidenceData.bins && confidenceData.bins.length > 0 ? (
              <div>
                <div className="space-y-2 mb-4">
                  {confidenceData.bins.map((bin, idx) => {
                    const maxCount = Math.max(...confidenceData.counts);
                    const percentage = (confidenceData.counts[idx] / maxCount) * 100;
                    
                    return (
                      <div key={idx}>
                        <div className="flex justify-between items-center mb-1">
                          <span className="text-xs font-medium text-gray-600">{bin}</span>
                          <span className="text-xs font-bold text-gray-800">
                            {confidenceData.counts[idx]}
                          </span>
                        </div>
                        <div className="w-full bg-gray-200 rounded-full h-2">
                          <div
                            className="bg-gradient-to-r from-blue-500 to-purple-600 h-2 rounded-full"
                            style={{ width: `${percentage}%` }}
                          ></div>
                        </div>
                      </div>
                    );
                  })}
                </div>
                
                {confidenceData.statistics && (
                  <div className="grid grid-cols-2 gap-4 pt-4 border-t border-gray-200">
                    <div>
                      <p className="text-xs text-gray-600">Mean</p>
                      <p className="text-lg font-bold text-gray-800">
                        {(confidenceData.statistics.mean * 100).toFixed(1)}%
                      </p>
                    </div>
                    <div>
                      <p className="text-xs text-gray-600">Median</p>
                      <p className="text-lg font-bold text-gray-800">
                        {(confidenceData.statistics.median * 100).toFixed(1)}%
                      </p>
                    </div>
                  </div>
                )}
              </div>
            ) : (
              <p className="text-gray-500 text-center py-8">No confidence data available</p>
            )}
          </div>
        </div>

        {/* Model Versions */}
        {modelVersions.length > 0 && (
          <div className="bg-white rounded-2xl shadow-lg p-6">
            <h2 className="text-2xl font-bold text-gray-800 mb-4">Model Versions</h2>
            
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="border-b border-gray-200">
                    <th className="text-left py-3 px-4 text-gray-600 font-semibold">Version</th>
                    <th className="text-left py-3 px-4 text-gray-600 font-semibold">Timestamp</th>
                    <th className="text-left py-3 px-4 text-gray-600 font-semibold">Accuracy</th>
                    <th className="text-left py-3 px-4 text-gray-600 font-semibold">Precision</th>
                    <th className="text-left py-3 px-4 text-gray-600 font-semibold">Recall</th>
                    <th className="text-left py-3 px-4 text-gray-600 font-semibold">F1 Score</th>
                  </tr>
                </thead>
                <tbody>
                  {modelVersions.map((version, idx) => (
                    <tr key={idx} className="border-b border-gray-100 hover:bg-gray-50">
                      <td className="py-3 px-4 font-medium text-gray-800">
                        {version.model_version}
                      </td>
                      <td className="py-3 px-4 text-sm text-gray-600">
                        {new Date(version.timestamp).toLocaleDateString()}
                      </td>
                      <td className="py-3 px-4 text-sm text-gray-700">
                        {(version.metrics.accuracy * 100).toFixed(2)}%
                      </td>
                      <td className="py-3 px-4 text-sm text-gray-700">
                        {(version.metrics.precision * 100).toFixed(2)}%
                      </td>
                      <td className="py-3 px-4 text-sm text-gray-700">
                        {(version.metrics.recall * 100).toFixed(2)}%
                      </td>
                      <td className="py-3 px-4 text-sm text-gray-700">
                        {(version.metrics.f1_score * 100).toFixed(2)}%
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default MLOpsDashboard;
