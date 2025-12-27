import { Link, useLocation, useNavigate } from 'react-router-dom';
import { Brain, Award, LogOut, User } from 'lucide-react';
import { useState, useEffect } from 'react';
import { useAuth } from '../contexts/AuthContext';
import { getUserStats } from '../utils/api';
import { getUserId } from '../utils/storage';

function Navigation() {
  const location = useLocation();
  const navigate = useNavigate();
  const { user, logout } = useAuth();
  const [points, setPoints] = useState(0);
  const userId = getUserId();

  useEffect(() => {
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
  }, [userId]);

  const isActive = (path) => {
    return location.pathname === path;
  };

  const handleLogout = () => {
    logout();
    navigate('/login');
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
              to="/mlops"
              className={`font-medium transition-colors ${
                isActive('/mlops') ? 'text-purple-600' : 'text-gray-600 hover:text-purple-600'
              }`}
            >
              MLOps
            </Link>
          </div>

          {/* Right Side - Points, User, Logout */}
          <div className="flex items-center space-x-4">
            {/* Points Badge */}
            <div className="flex items-center space-x-2 bg-gradient-to-r from-amber-400 to-orange-500 text-white px-4 py-2 rounded-full shadow-lg">
              <Award className="w-5 h-5" />
              <span className="font-bold">{points}</span>
              <span className="text-sm">pts</span>
            </div>

            {/* User Info & Logout (Desktop) */}
            <div className="hidden md:flex items-center space-x-3">
              {user && (
                <div className="flex items-center space-x-2 text-sm text-gray-700">
                  <User className="w-4 h-4 text-gray-500" />
                  <span className="font-medium">{user.full_name || user.email}</span>
                </div>
              )}
              <button
                onClick={handleLogout}
                className="flex items-center space-x-2 px-4 py-2 text-sm font-medium text-gray-700 hover:text-red-600 hover:bg-red-50 rounded-lg transition-all"
                title="Logout"
              >
                <LogOut className="w-4 h-4" />
                <span>Logout</span>
              </button>
            </div>

            {/* Logout Icon (Mobile) */}
            <button
              onClick={handleLogout}
              className="md:hidden flex items-center justify-center p-2 text-gray-700 hover:text-red-600 hover:bg-red-50 rounded-lg transition-all"
              title="Logout"
            >
              <LogOut className="w-5 h-5" />
            </button>
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
        </div>
      </div>
    </nav>
  );
}

export default Navigation;
