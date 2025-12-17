import { Navigate } from 'react-router-dom';
import { isAuthenticated } from '../utils/auth';

/**
 * Protected route wrapper that requires authentication
 * Redirects to login page if user is not authenticated
 * 
 * @param {Object} props - Component props
 * @param {React.ReactNode} props.children - Child components to render if authenticated
 * @returns {React.ReactElement} Children if authenticated, Navigate to login otherwise
 */
function ProtectedRoute({ children }) {
  if (!isAuthenticated()) {
    return <Navigate to="/login" replace />;
  }

  return children;
}

export default ProtectedRoute;
