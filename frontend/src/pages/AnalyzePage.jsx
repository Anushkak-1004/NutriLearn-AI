import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { ArrowLeft, Camera, Loader2, CheckCircle, Flame, Beef, Wheat, Droplet, Apple } from 'lucide-react';
import { predictFood, logMeal } from '../utils/api';
import { getUserId } from '../utils/storage';

function AnalyzePage() {
  const navigate = useNavigate();
  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [loading, setLoading] = useState(false);
  const [prediction, setPrediction] = useState(null);
  const [mealType, setMealType] = useState('lunch');
  const [logging, setLogging] = useState(false);
  const [success, setSuccess] = useState(false);

  const handleFileSelect = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setPrediction(null);
      setSuccess(false);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;

    setLoading(true);
    try {
      const formData = new FormData();
      formData.append('file', selectedFile);

      const result = await predictFood(formData);
      setPrediction(result);
    } catch (error) {
      console.error('Prediction error:', error);
      alert('Failed to analyze image. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleLogMeal = async () => {
    if (!prediction) return;

    setLogging(true);
    try {
      const userId = getUserId();
      await logMeal(userId, {
        food_name: prediction.food_name,
        nutrition: prediction.nutrition,
        meal_type: mealType,
      });

      setSuccess(true);
      setTimeout(() => {
        navigate('/dashboard');
      }, 2000);
    } catch (error) {
      console.error('Logging error:', error);
      alert('Failed to log meal. Please try again.');
    } finally {
      setLogging(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-green-50 to-blue-50 py-8 px-4">
      <div className="max-w-4xl mx-auto">
        {/* Header */}
        <button
          onClick={() => navigate('/')}
          className="flex items-center space-x-2 text-gray-600 hover:text-gray-800 mb-6 transition-colors"
        >
          <ArrowLeft className="w-5 h-5" />
          <span className="font-medium">Back to Home</span>
        </button>

        <h1 className="text-4xl font-bold text-gray-800 mb-8">Analyze Your Food</h1>

        {/* Upload Section */}
        <div className="bg-white rounded-2xl shadow-lg p-8 mb-6">
          <label className="block">
            <div className="border-2 border-dashed border-gray-300 rounded-xl p-8 text-center hover:border-emerald-500 transition-colors cursor-pointer">
              {preview ? (
                <img src={preview} alt="Preview" className="max-h-64 mx-auto rounded-lg" />
              ) : (
                <div>
                  <Camera className="w-16 h-16 text-gray-400 mx-auto mb-4" />
                  <p className="text-gray-600 font-medium">Click to upload food image</p>
                  <p className="text-gray-400 text-sm mt-2">JPEG, PNG supported</p>
                </div>
              )}
            </div>
            <input
              type="file"
              accept="image/*"
              onChange={handleFileSelect}
              className="hidden"
            />
          </label>

          {selectedFile && !prediction && (
            <button
              onClick={handleAnalyze}
              disabled={loading}
              className="w-full mt-6 bg-emerald-600 hover:bg-emerald-700 text-white font-semibold py-3 px-6 rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
            >
              {loading ? (
                <>
                  <Loader2 className="w-5 h-5 animate-spin" />
                  <span>Analyzing...</span>
                </>
              ) : (
                <span>Analyze Food</span>
              )}
            </button>
          )}
        </div>

        {/* Results Section */}
        {prediction && (
          <div className="space-y-6">
            {/* Food Name & Confidence */}
            <div className="bg-white rounded-2xl shadow-lg p-8">
              <h2 className="text-3xl font-bold text-gray-800 mb-2">{prediction.food_name}</h2>
              <div className="flex items-center space-x-2">
                <span className="text-gray-600">Confidence:</span>
                <span className={`font-bold px-3 py-1 rounded-full text-sm ${
                  prediction.confidence > 0.9 ? 'bg-green-100 text-green-700' :
                  prediction.confidence > 0.8 ? 'bg-yellow-100 text-yellow-700' :
                  'bg-orange-100 text-orange-700'
                }`}>
                  {(prediction.confidence * 100).toFixed(1)}%
                </span>
              </div>
              <div className="mt-2 text-sm text-gray-500">
                {prediction.category} • {prediction.cuisine} cuisine
              </div>
            </div>

            {/* Nutrition Grid */}
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              <div className="bg-gradient-to-br from-red-500 to-orange-500 text-white rounded-xl p-4 shadow-lg">
                <Flame className="w-6 h-6 mb-2" />
                <div className="text-2xl font-bold">{prediction.nutrition.calories}</div>
                <div className="text-sm opacity-90">Calories</div>
              </div>

              <div className="bg-gradient-to-br from-pink-500 to-rose-500 text-white rounded-xl p-4 shadow-lg">
                <Beef className="w-6 h-6 mb-2" />
                <div className="text-2xl font-bold">{prediction.nutrition.protein}g</div>
                <div className="text-sm opacity-90">Protein</div>
              </div>

              <div className="bg-gradient-to-br from-amber-500 to-yellow-500 text-white rounded-xl p-4 shadow-lg">
                <Wheat className="w-6 h-6 mb-2" />
                <div className="text-2xl font-bold">{prediction.nutrition.carbs}g</div>
                <div className="text-sm opacity-90">Carbs</div>
              </div>

              <div className="bg-gradient-to-br from-blue-500 to-cyan-500 text-white rounded-xl p-4 shadow-lg">
                <Droplet className="w-6 h-6 mb-2" />
                <div className="text-2xl font-bold">{prediction.nutrition.fat}g</div>
                <div className="text-sm opacity-90">Fat</div>
              </div>

              <div className="bg-gradient-to-br from-green-500 to-emerald-500 text-white rounded-xl p-4 shadow-lg">
                <Apple className="w-6 h-6 mb-2" />
                <div className="text-2xl font-bold">{prediction.nutrition.fiber}g</div>
                <div className="text-sm opacity-90">Fiber</div>
              </div>
            </div>

            {/* Meal Type Selection */}
            <div className="bg-white rounded-2xl shadow-lg p-8">
              <h3 className="text-xl font-bold text-gray-800 mb-4">Meal Type</h3>
              <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                {['breakfast', 'lunch', 'dinner', 'snack'].map((type) => (
                  <button
                    key={type}
                    onClick={() => setMealType(type)}
                    className={`py-3 px-4 rounded-xl font-medium transition-all ${
                      mealType === type
                        ? 'bg-emerald-600 text-white shadow-lg'
                        : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                    }`}
                  >
                    {type.charAt(0).toUpperCase() + type.slice(1)}
                  </button>
                ))}
              </div>

              <button
                onClick={handleLogMeal}
                disabled={logging || success}
                className="w-full mt-6 bg-blue-600 hover:bg-blue-700 text-white font-semibold py-3 px-6 rounded-xl transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
              >
                {logging ? (
                  <>
                    <Loader2 className="w-5 h-5 animate-spin" />
                    <span>Logging...</span>
                  </>
                ) : success ? (
                  <>
                    <CheckCircle className="w-5 h-5" />
                    <span>Meal Logged!</span>
                  </>
                ) : (
                  <span>Log This Meal</span>
                )}
              </button>

              {success && (
                <div className="mt-4 p-4 bg-green-50 border border-green-200 rounded-xl text-green-700 text-center">
                  Meal logged successfully! Redirecting to dashboard...
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}

export default AnalyzePage;
