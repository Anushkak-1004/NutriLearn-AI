import { useNavigate } from 'react-router-dom';
import { Brain, Camera, TrendingUp, Upload, BarChart3, BookOpen, Award } from 'lucide-react';
import { useState, useEffect } from 'react';
import { getUserStats } from '../utils/api';
import { getUserId } from '../utils/storage';

function HomePage() {
  const navigate = useNavigate();
  const [points, setPoints] = useState(0);
  const userId = getUserId();

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const stats = await getUserStats(userId);
        setPoints(stats.total_points);
      } catch (error) {
        console.error('Error fetching stats:', error);
      }
    };
    fetchStats();
  }, [userId]);

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50">
      {/* Hero Section */}
      <div className="flex flex-col items-center justify-center min-h-[70vh] px-4 text-center">
        <div className="mb-6 relative">
          <div className="absolute inset-0 bg-emerald-400 blur-3xl opacity-20 rounded-full"></div>
          <Brain className="w-24 h-24 text-emerald-600 relative z-10" />
        </div>
        
        <h1 className="text-5xl md:text-7xl font-bold text-gray-800 mb-4">
          NutriLearn AI
        </h1>
        
        <p className="text-xl md:text-2xl text-gray-600 mb-8 max-w-2xl">
          Learn nutrition while you track your diet
        </p>

        {/* Points Badge */}
        {points > 0 && (
          <div className="mb-8 flex items-center space-x-2 bg-gradient-to-r from-amber-400 to-orange-500 text-white px-6 py-3 rounded-full shadow-lg">
            <Award className="w-6 h-6" />
            <span className="text-2xl font-bold">{points}</span>
            <span className="text-lg">points earned!</span>
          </div>
        )}

        {/* Action Cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full max-w-4xl mt-8">
          {/* Analyze Food Card */}
          <button
            onClick={() => navigate('/analyze')}
            className="group bg-white rounded-2xl shadow-lg hover:shadow-2xl transition-all duration-300 p-8 text-left transform hover:-translate-y-1"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="bg-emerald-100 p-4 rounded-xl group-hover:bg-emerald-200 transition-colors">
                <Camera className="w-8 h-8 text-emerald-600" />
              </div>
              <div className="text-emerald-600 font-semibold">Start →</div>
            </div>
            <h3 className="text-2xl font-bold text-gray-800 mb-2">Analyze Food</h3>
            <p className="text-gray-600">
              Upload a photo and get instant nutrition insights powered by AI
            </p>
          </button>

          {/* My Progress Card */}
          <button
            onClick={() => navigate('/dashboard')}
            className="group bg-white rounded-2xl shadow-lg hover:shadow-2xl transition-all duration-300 p-8 text-left transform hover:-translate-y-1"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="bg-blue-100 p-4 rounded-xl group-hover:bg-blue-200 transition-colors">
                <TrendingUp className="w-8 h-8 text-blue-600" />
              </div>
              <div className="text-blue-600 font-semibold">View →</div>
            </div>
            <h3 className="text-2xl font-bold text-gray-800 mb-2">My Progress</h3>
            <p className="text-gray-600">
              Track your meals, analyze patterns, and unlock learning modules
            </p>
          </button>
        </div>
      </div>

      {/* How It Works Section */}
      <div className="bg-white py-16 px-4">
        <div className="max-w-6xl mx-auto">
          <h2 className="text-4xl font-bold text-center text-gray-800 mb-12">
            How It Works
          </h2>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-8">
            {/* Step 1 */}
            <div className="text-center">
              <div className="bg-emerald-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                <Upload className="w-8 h-8 text-emerald-600" />
              </div>
              <div className="bg-emerald-600 text-white w-8 h-8 rounded-full flex items-center justify-center mx-auto mb-4 font-bold">
                1
              </div>
              <h3 className="text-xl font-bold text-gray-800 mb-2">Upload Food Photo</h3>
              <p className="text-gray-600">
                Take a picture of your meal and let our AI identify it
              </p>
            </div>

            {/* Step 2 */}
            <div className="text-center">
              <div className="bg-blue-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                <BarChart3 className="w-8 h-8 text-blue-600" />
              </div>
              <div className="bg-blue-600 text-white w-8 h-8 rounded-full flex items-center justify-center mx-auto mb-4 font-bold">
                2
              </div>
              <h3 className="text-xl font-bold text-gray-800 mb-2">Get Insights</h3>
              <p className="text-gray-600">
                Receive detailed nutrition information and track your patterns
              </p>
            </div>

            {/* Step 3 */}
            <div className="text-center">
              <div className="bg-purple-100 w-16 h-16 rounded-full flex items-center justify-center mx-auto mb-4">
                <BookOpen className="w-8 h-8 text-purple-600" />
              </div>
              <div className="bg-purple-600 text-white w-8 h-8 rounded-full flex items-center justify-center mx-auto mb-4 font-bold">
                3
              </div>
              <h3 className="text-xl font-bold text-gray-800 mb-2">Learn & Improve</h3>
              <p className="text-gray-600">
                Complete personalized learning modules and earn points
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default HomePage;
