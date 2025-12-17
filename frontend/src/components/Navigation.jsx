import { Link, useLocation, useNavigate } from 'react-router-dom';
import { Brain, Award } from 'lucide-react';
import { useState, useEffect } from 'react';
import { getUserStats } from '../utils/api';
import { getUserId } from '../utils/storage';
import { isAuthenticated, logout } from '../utils/auth';

function Navigation() {
  const location = useLocation();
  const navigate = useNavigate();
  const [points, setPoints] = useState(0);
  const authenticated = isAuthenticated();
  const userId = getUserId();

  useEffect(() => {
    // Only fetch user points if authenticated
    if (!authenticated) {
      return;
    }

    // Fetch user points
    const fetchPoints = async () => {
      try {
        const stats = await getUserStats(userId);
        setPoints(stats.total_points);
      } catch (error) {
        console.error('Error fetching points:', error);
      }
    };

    fetchPoints();
    // Refresh points every 30 seconds
    const interval = setInterval(fetchPoints, 30000);
    return () => clearInterval(interval);
  }, [userId, authenticated]);

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  const isActive = (path) => {
    return location.pathname === path;
  };

  return (
    <nav className="sticky top-0 z-50 bg-white shadow-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center space-x-2 hover:opacity-80 transition-opacity">
            <Brain className="w-8 h-8 text-emerald-600" />
            <span className="text-xl font-bold text-gray-800">NutriLearn AI</span>
          </Link>

          {/* Navigation Links */}
          <div className="hidden md:flex items-center space-x-8">
            <Link
              to="/"
              className={`font-medium transition-colors ${
                isActive('/') ? 'text-emerald-600' : 'text-gray-600 hover:text-emerald-600'
              }`}
            >
              Home
            </Link>
            {authenticated && (
              <>
                <Link
                  to="/analyze"
                  className={`font-medium transition-colors ${
                    isActive('/analyze') ? 'text-emerald-600' : 'text-gray-600 hover:text-emerald-600'
                  }`}
                >
                  Analyze
                </Link>
                <Link
                  to="/dashboard"
                  className={`font-medium transition-colors ${
                    isActive('/dashboard') ? 'text-emerald-600' : 'text-gray-600 hover:text-emerald-600'
                  }`}
                >
                  Dashboard
                </Link>
                <Link
                  to="/learning"
                  className={`font-medium transition-colors ${
                    isActive('/learning') ? 'text-emerald-600' : 'text-gray-600 hover:text-emerald-600'
                  }`}
                >
                  Learning
                </Link>
                <Link
                  to="/mlops"
                  className={`font-medium transition-colors ${
                    isActive('/mlops') ? 'text-purple-600' : 'text-gray-600 hover:text-purple-600'
                  }`}
                >
                  MLOps
                </Link>
              </>
            )}
          </div>

          {/* Auth Actions */}
          <div className="flex items-center space-x-4">
            {authenticated ? (
              <>
                {/* Points Badge */}
                <div className="flex items-center space-x-2 bg-gradient-to-r from-amber-400 to-orange-500 text-white px-4 py-2 rounded-full shadow-lg">
                  <Award className="w-5 h-5" />
                  <span className="font-bold">{points}</span>
                  <span className="text-sm">pts</span>
                </div>
                <button
                  onClick={handleLogout}
                  className="px-4 py-2 text-sm font-medium text-white bg-red-600 rounded-md hover:bg-red-700 focus:outline-none focus:ring-2 focus:ring-red-500 focus:ring-offset-2 transition-colors"
                >
                  Logout
                </button>
              </>
            ) : (
              <>
                <Link
                  to="/login"
                  className="px-4 py-2 text-sm font-medium text-gray-700 hover:text-emerald-600 transition-colors"
                >
                  Login
                </Link>
                <Link
                  to="/signup"
                  className="px-4 py-2 text-sm font-medium text-white bg-emerald-600 rounded-md hover:bg-emerald-700 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:ring-offset-2 transition-colors"
                >
                  Sign Up
                </Link>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Mobile Navigation */}
      <div className="md:hidden border-t border-gray-200">
        <div className="flex justify-around py-2">
          <Link
            to="/"
            className={`flex flex-col items-center py-2 px-3 ${
              isActive('/') ? 'text-emerald-600' : 'text-gray-600'
            }`}
          >
            <span className="text-xs font-medium">Home</span>
          </Link>
          {authenticated ? (
            <>
              <Link
                to="/analyze"
                className={`flex flex-col items-center py-2 px-3 ${
                  isActive('/analyze') ? 'text-emerald-600' : 'text-gray-600'
                }`}
              >
                <span className="text-xs font-medium">Analyze</span>
              </Link>
              <Link
                to="/dashboard"
                className={`flex flex-col items-center py-2 px-3 ${
                  isActive('/dashboard') ? 'text-emerald-600' : 'text-gray-600'
                }`}
              >
                <span className="text-xs font-medium">Dashboard</span>
              </Link>
              <Link
                to="/mlops"
                className={`flex flex-col items-center py-2 px-3 ${
                  isActive('/mlops') ? 'text-purple-600' : 'text-gray-600'
                }`}
              >
                <span className="text-xs font-medium">MLOps</span>
              </Link>
            </>
          ) : (
            <>
              <Link
                to="/login"
                className={`flex flex-col items-center py-2 px-3 ${
                  isActive('/login') ? 'text-emerald-600' : 'text-gray-600'
                }`}
              >
                <span className="text-xs font-medium">Login</span>
              </Link>
              <Link
                to="/signup"
                className={`flex flex-col items-center py-2 px-3 ${
                  isActive('/signup') ? 'text-emerald-600' : 'text-gray-600'
                }`}
              >
                <span className="text-xs font-medium">Sign Up</span>
              </Link>
            </>
          )}
        </div>
      </div>
    </nav>
  );
}

export default Navigation;
