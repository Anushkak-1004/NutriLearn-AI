import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Flame, Beef, Wheat, Droplet, Apple, Award, BookOpen, AlertCircle, TrendingUp } from 'lucide-react';
import { getUserStats, getMealHistory, getDietaryAnalysis } from '../utils/api';
import { getUserId } from '../utils/storage';

function DashboardPage() {
  const navigate = useNavigate();
  const [stats, setStats] = useState(null);
  const [meals, setMeals] = useState([]);
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(true);
  const userId = getUserId();

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [statsData, mealsData] = await Promise.all([
          getUserStats(userId),
          getMealHistory(userId, 10),
        ]);

        setStats(statsData);
        setMeals(mealsData.meals);

        // Only fetch analysis if user has enough meals
        if (statsData.total_meals >= 3) {
          const analysisData = await getDietaryAnalysis(userId);
          setAnalysis(analysisData);
        }
      } catch (error) {
        console.error('Error fetching dashboard data:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchData();
  }, [userId]);

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-16 w-16 border-b-2 border-emerald-600 mx-auto mb-4"></div>
          <p className="text-gray-600">Loading your dashboard...</p>
        </div>
      </div>
    );
  }

  const totalMeals = stats?.total_meals || 0;
  const needsMoreMeals = totalMeals < 3;

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50 py-8 px-4">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <button
          onClick={() => navigate('/')}
          className="flex items-center space-x-2 text-gray-600 hover:text-gray-800 mb-6 transition-colors"
        >
          <ArrowLeft className="w-5 h-5" />
          <span className="font-medium">Back to Home</span>
        </button>

        <h1 className="text-4xl font-bold text-gray-800 mb-8">My Dashboard</h1>

        {/* Top Stats */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
          <div className="bg-white rounded-2xl shadow-lg p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-600 text-sm">Meals Logged</p>
                <p className="text-3xl font-bold text-gray-800">{totalMeals}</p>
              </div>
              <div className="bg-emerald-100 p-3 rounded-xl">
                <Flame className="w-8 h-8 text-emerald-600" />
              </div>
            </div>
          </div>

          <div className="bg-white rounded-2xl shadow-lg p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-600 text-sm">Modules Completed</p>
                <p className="text-3xl font-bold text-gray-800">{stats?.completed_modules?.length || 0}</p>
              </div>
              <div className="bg-blue-100 p-3 rounded-xl">
                <BookOpen className="w-8 h-8 text-blue-600" />
              </div>
            </div>
          </div>

          <div className="bg-white rounded-2xl shadow-lg p-6">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-gray-600 text-sm">Total Points</p>
                <p className="text-3xl font-bold text-gray-800">{stats?.total_points || 0}</p>
              </div>
              <div className="bg-amber-100 p-3 rounded-xl">
                <Award className="w-8 h-8 text-amber-600" />
              </div>
            </div>
          </div>
        </div>

        {/* Main Content Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Left Panel - Today's Nutrition */}
          <div className="space-y-6">
            <div className="bg-white rounded-2xl shadow-lg p-6">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">Today's Nutrition</h2>
              
              {meals.length > 0 ? (
                <>
                  {/* Nutrition Summary */}
                  <div className="grid grid-cols-2 gap-4 mb-6">
                    <div className="bg-gradient-to-br from-red-50 to-orange-50 rounded-xl p-4">
                      <Flame className="w-5 h-5 text-red-600 mb-2" />
                      <div className="text-2xl font-bold text-gray-800">
                        {meals.reduce((sum, meal) => sum + meal.nutrition.calories, 0)}
                      </div>
                      <div className="text-sm text-gray-600">Calories</div>
                    </div>

                    <div className="bg-gradient-to-br from-pink-50 to-rose-50 rounded-xl p-4">
                      <Beef className="w-5 h-5 text-pink-600 mb-2" />
                      <div className="text-2xl font-bold text-gray-800">
                        {meals.reduce((sum, meal) => sum + meal.nutrition.protein, 0).toFixed(1)}g
                      </div>
                      <div className="text-sm text-gray-600">Protein</div>
                    </div>

                    <div className="bg-gradient-to-br from-amber-50 to-yellow-50 rounded-xl p-4">
                      <Wheat className="w-5 h-5 text-amber-600 mb-2" />
                      <div className="text-2xl font-bold text-gray-800">
                        {meals.reduce((sum, meal) => sum + meal.nutrition.carbs, 0).toFixed(1)}g
                      </div>
                      <div className="text-sm text-gray-600">Carbs</div>
                    </div>

                    <div className="bg-gradient-to-br from-blue-50 to-cyan-50 rounded-xl p-4">
                      <Droplet className="w-5 h-5 text-blue-600 mb-2" />
                      <div className="text-2xl font-bold text-gray-800">
                        {meals.reduce((sum, meal) => sum + meal.nutrition.fat, 0).toFixed(1)}g
                      </div>
                      <div className="text-sm text-gray-600">Fat</div>
                    </div>
                  </div>

                  {/* Recent Meals */}
                  <h3 className="text-lg font-bold text-gray-800 mb-3">Recent Meals</h3>
                  <div className="space-y-3">
                    {meals.slice(0, 5).map((meal, index) => (
                      <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-xl">
                        <div>
                          <p className="font-medium text-gray-800">{meal.food_name}</p>
                          <p className="text-sm text-gray-500 capitalize">{meal.meal_type}</p>
                        </div>
                        <div className="text-right">
                          <p className="font-bold text-gray-800">{meal.nutrition.calories} cal</p>
                          <p className="text-xs text-gray-500">{meal.nutrition.protein}g protein</p>
                        </div>
                      </div>
                    ))}
                  </div>
                </>
              ) : (
                <div className="text-center py-8">
                  <p className="text-gray-500">No meals logged yet</p>
                  <button
                    onClick={() => navigate('/analyze')}
                    className="mt-4 bg-emerald-600 hover:bg-emerald-700 text-white font-semibold py-2 px-6 rounded-xl transition-colors"
                  >
                    Log Your First Meal
                  </button>
                </div>
              )}
            </div>
          </div>

          {/* Right Panel - Learning Path */}
          <div className="space-y-6">
            <div className="bg-white rounded-2xl shadow-lg p-6">
              <h2 className="text-2xl font-bold text-gray-800 mb-4">Your Learning Path</h2>

              {needsMoreMeals ? (
                <div className="bg-blue-50 border border-blue-200 rounded-xl p-6 text-center">
                  <AlertCircle className="w-12 h-12 text-blue-600 mx-auto mb-3" />
                  <p className="text-gray-700 font-medium mb-2">
                    Log at least 3 meals for personalized analysis
                  </p>
                  <p className="text-gray-600 text-sm mb-4">
                    {totalMeals}/3 meals logged
                  </p>
                  <div className="w-full bg-gray-200 rounded-full h-2 mb-4">
                    <div
                      className="bg-blue-600 h-2 rounded-full transition-all"
                      style={{ width: `${(totalMeals / 3) * 100}%` }}
                    ></div>
                  </div>
                  <button
                    onClick={() => navigate('/analyze')}
                    className="bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-6 rounded-xl transition-colors"
                  >
                    Log More Meals
                  </button>
                </div>
              ) : (
                <>
                  {/* Dietary Patterns */}
                  {analysis?.patterns && analysis.patterns.length > 0 && (
                    <div className="mb-6">
                      <h3 className="text-lg font-bold text-gray-800 mb-3 flex items-center">
                        <TrendingUp className="w-5 h-5 mr-2 text-orange-600" />
                        Identified Patterns
                      </h3>
                      <div className="space-y-3">
                        {analysis.patterns.map((pattern, index) => (
                          <div
                            key={index}
                            className={`p-4 rounded-xl border-l-4 ${
                              pattern.severity === 'high'
                                ? 'bg-red-50 border-red-500'
                                : pattern.severity === 'medium'
                                ? 'bg-orange-50 border-orange-500'
                                : 'bg-yellow-50 border-yellow-500'
                            }`}
                          >
                            <p className="font-medium text-gray-800 mb-1">{pattern.description}</p>
                            <p className="text-sm text-gray-600">{pattern.recommendation}</p>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Recommended Modules */}
                  {analysis?.recommended_modules && analysis.recommended_modules.length > 0 && (
                    <div>
                      <h3 className="text-lg font-bold text-gray-800 mb-3 flex items-center">
                        <BookOpen className="w-5 h-5 mr-2 text-emerald-600" />
                        Recommended Learning
                      </h3>
                      <div className="space-y-3">
                        {analysis.recommended_modules.map((module, index) => (
                          <button
                            key={index}
                            onClick={() => navigate(`/learning/${module.module_id}`, { state: { module } })}
                            className="w-full text-left p-4 bg-gradient-to-r from-emerald-50 to-blue-50 rounded-xl hover:shadow-lg transition-all"
                          >
                            <div className="flex items-center justify-between mb-2">
                              <h4 className="font-bold text-gray-800">{module.title}</h4>
                              <span className="bg-emerald-600 text-white text-xs font-bold px-2 py-1 rounded-full">
                                +{module.points} pts
                              </span>
                            </div>
                            <p className="text-sm text-gray-600 mb-2">{module.reason}</p>
                            <p className="text-xs text-gray-500">⏱ {module.estimated_time} min</p>
                          </button>
                        ))}
                      </div>
                    </div>
                  )}
                </>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default DashboardPage;
