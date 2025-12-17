# Frontend Authentication System Design

## Overview

This document outlines the design for implementing authentication functionality in the NutriLearn AI React frontend application. The authentication system will provide a seamless user experience for registration, login, logout, and protected route access while integrating with the existing JWT-based backend API.

The design follows React best practices including functional components, custom hooks for state management, centralized utility modules for authentication logic, and Axios interceptors for automatic token management. The system ensures security by storing JWT tokens in localStorage and automatically including them in API requests.

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   React Application                      │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │              App Component (Router)                │ │
│  │  ┌──────────────┐  ┌──────────────────────────┐  │ │
│  │  │ Public Routes│  │   Protected Routes       │  │ │
│  │  │ - /          │  │   - /dashboard           │  │ │
│  │  │ - /login     │  │   - /analyze             │  │ │
│  │  │ - /signup    │  │   - /learning            │  │ │
│  │  └──────────────┘  │   - /mlops               │  │ │
│  │                    └──────────────────────────┘  │ │
│  └────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │           Authentication Components                │ │
│  │  - LoginPage.jsx                                   │ │
│  │  - SignupPage.jsx                                  │ │
│  │  - ProtectedRoute.jsx                              │ │
│  │  - Navigation.jsx (updated)                        │ │
│  └────────────────────────────────────────────────────┘ │
│                                                          │
│  ┌────────────────────────────────────────────────────┐ │
│  │              Utility Modules                       │ │
│  │  ┌──────────────────┐  ┌──────────────────────┐  │ │
│  │  │   auth.js        │  │      api.js          │  │ │
│  │  │  - login()       │  │  - axios instance    │  │ │
│  │  │  - signup()      │  │  - interceptors      │  │ │
│  │  │  - logout()      │  │  - setAuthToken()    │  │ │
│  │  │  - getToken()    │  │                      │  │ │
│  │  │  - isAuth()      │  │                      │  │ │
│  │  └──────────────────┘  └──────────────────────┘  │ │
│  └────────────────────────────────────────────────────┘ │
│                                                          │
└──────────────────┬───────────────────────────────────────┘
                   │ HTTP + JWT Token
                   ▼
┌─────────────────────────────────────────────────────────┐
│              Backend API (FastAPI)                       │
│  - POST /api/v1/auth/signup                             │
│  - POST /api/v1/auth/login                              │
│  - GET  /api/v1/auth/me                                 │
│  - Protected endpoints (require Bearer token)           │
└─────────────────────────────────────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────────────────┐
│              Browser localStorage                        │
│  - token: "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."    │
└─────────────────────────────────────────────────────────┘
```

### Authentication Flow Diagrams

**Signup Flow:**
```
User fills form → SignupPage validates → auth.signup() called →
POST /api/v1/auth/signup → Backend validates → Returns JWT →
Store in localStorage → Update Axios headers → Redirect to /dashboard
```

**Login Flow:**
```
User fills form → LoginPage validates → auth.login() called →
POST /api/v1/auth/login → Backend validates → Returns JWT →
Store in localStorage → Update Axios headers → Redirect to /dashboard
```

**Protected Route Access:**
```
User navigates to /analyze → ProtectedRoute checks isAuthenticated() →
If token exists → Render page → API calls include Bearer token →
If no token → Redirect to /login
```

**Logout Flow:**
```
User clicks logout → auth.logout() called → Remove from localStorage →
Clear Axios headers → Redirect to / or /login
```

**Token Expiration Handling:**
```
API request → Backend returns 401 → Axios interceptor catches →
Remove token from localStorage → Redirect to /login
```

## Components and Interfaces

### 1. Authentication Utility Module (`frontend/src/utils/auth.js`)

Centralized module for all authentication operations.

**Functions:**

```javascript
/**
 * Register a new user account
 * @param {string} email - User's email address
 * @param {string} password - User's password
 * @returns {Promise<Object>} Response data with access_token
 * @throws {Error} If signup fails (validation, duplicate email, network)
 */
export async function signup(email, password) {
  const response = await api.post('/api/v1/auth/signup', { email, password });
  const { access_token } = response.data;
  localStorage.setItem('token', access_token);
  setAuthToken(access_token);
  return response.data;
}

/**
 * Authenticate user and obtain access token
 * @param {string} email - User's email address
 * @param {string} password - User's password
 * @returns {Promise<Object>} Response data with access_token
 * @throws {Error} If login fails (invalid credentials, network)
 */
export async function login(email, password) {
  const response = await api.post('/api/v1/auth/login', { email, password });
  const { access_token } = response.data;
  localStorage.setItem('token', access_token);
  setAuthToken(access_token);
  return response.data;
}

/**
 * Log out current user and clear authentication state
 */
export function logout() {
  localStorage.removeItem('token');
  setAuthToken(null);
}

/**
 * Retrieve stored JWT token from localStorage
 * @returns {string|null} JWT token or null if not found
 */
export function getToken() {
  return localStorage.getItem('token');
}

/**
 * Check if user is currently authenticated
 * @returns {boolean} True if valid token exists, false otherwise
 */
export function isAuthenticated() {
  const token = getToken();
  return token !== null && token !== undefined && token !== '';
}
```

### 2. API Utility Module Updates (`frontend/src/utils/api.js`)

Enhanced Axios configuration with authentication support.

**Functions:**

```javascript
import axios from 'axios';
import { getToken } from './auth';

// Create axios instance with base configuration
const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Set or clear authentication token in Axios default headers
 * @param {string|null} token - JWT token to set, or null to clear
 */
export function setAuthToken(token) {
  if (token) {
    api.defaults.headers.common['Authorization'] = `Bearer ${token}`;
  } else {
    delete api.defaults.headers.common['Authorization'];
  }
}

/**
 * Initialize authentication on app load
 * Retrieves token from localStorage and configures Axios
 */
export function initializeAuth() {
  const token = getToken();
  if (token) {
    setAuthToken(token);
  }
}

// Request interceptor - ensures token is always included
api.interceptors.request.use(
  (config) => {
    const token = getToken();
    if (token && !config.headers['Authorization']) {
      config.headers['Authorization'] = `Bearer ${token}`;
    }
    return config;
  },
  (error) => {
    return Promise.reject(error);
  }
);

// Response interceptor - handles 401 errors globally
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Token expired or invalid
      localStorage.removeItem('token');
      setAuthToken(null);
      
      // Redirect to login if not already there
      if (window.location.pathname !== '/login') {
        window.location.href = '/login';
      }
    }
    return Promise.reject(error);
  }
);

export default api;
```

### 3. Login Page Component (`frontend/src/pages/LoginPage.jsx`)

User interface for authentication.

```javascript
import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { login } from '../utils/auth';

function LoginPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      await login(email, password);
      navigate('/dashboard');
    } catch (err) {
      const message = err.response?.data?.message || 
                     err.response?.data?.detail ||
                     'Login failed. Please check your credentials.';
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="max-w-md w-full space-y-8 p-8 bg-white rounded-lg shadow">
        <h2 className="text-3xl font-bold text-center">Login to NutriLearn</h2>
        
        {error && (
          <div className="bg-red-50 text-red-600 p-3 rounded">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700">
              Email
            </label>
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700">
              Password
            </label>
            <input
              type="password"
              required
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md"
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full py-2 px-4 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
          >
            {loading ? 'Logging in...' : 'Login'}
          </button>
        </form>

        <p className="text-center text-sm text-gray-600">
          Don't have an account?{' '}
          <Link to="/signup" className="text-blue-600 hover:underline">
            Sign up
          </Link>
        </p>
      </div>
    </div>
  );
}

export default LoginPage;
```

### 4. Signup Page Component (`frontend/src/pages/SignupPage.jsx`)

User interface for registration.

```javascript
import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { signup } from '../utils/auth';

function SignupPage() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');

    // Client-side validation
    if (password.length < 8) {
      setError('Password must be at least 8 characters long');
      return;
    }

    if (password !== confirmPassword) {
      setError('Passwords do not match');
      return;
    }

    setLoading(true);

    try {
      await signup(email, password);
      navigate('/dashboard');
    } catch (err) {
      const message = err.response?.data?.message || 
                     err.response?.data?.detail ||
                     'Signup failed. Please try again.';
      setError(message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50">
      <div className="max-w-md w-full space-y-8 p-8 bg-white rounded-lg shadow">
        <h2 className="text-3xl font-bold text-center">Create Account</h2>
        
        {error && (
          <div className="bg-red-50 text-red-600 p-3 rounded">
            {error}
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700">
              Email
            </label>
            <input
              type="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700">
              Password
            </label>
            <input
              type="password"
              required
              minLength={8}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md"
            />
            <p className="mt-1 text-sm text-gray-500">
              Must be at least 8 characters
            </p>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700">
              Confirm Password
            </label>
            <input
              type="password"
              required
              value={confirmPassword}
              onChange={(e) => setConfirmPassword(e.target.value)}
              className="mt-1 block w-full px-3 py-2 border border-gray-300 rounded-md"
            />
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full py-2 px-4 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
          >
            {loading ? 'Creating account...' : 'Sign Up'}
          </button>
        </form>

        <p className="text-center text-sm text-gray-600">
          Already have an account?{' '}
          <Link to="/login" className="text-blue-600 hover:underline">
            Login
          </Link>
        </p>
      </div>
    </div>
  );
}

export default SignupPage;
```

### 5. Protected Route Component (`frontend/src/components/ProtectedRoute.jsx`)

Wrapper component for routes requiring authentication.

```javascript
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
```

### 6. Updated Navigation Component (`frontend/src/components/Navigation.jsx`)

Navigation menu that reflects authentication state.

```javascript
import { Link, useNavigate } from 'react-router-dom';
import { isAuthenticated, logout } from '../utils/auth';

function Navigation() {
  const navigate = useNavigate();
  const authenticated = isAuthenticated();

  const handleLogout = () => {
    logout();
    navigate('/');
  };

  return (
    <nav className="bg-white shadow-lg">
      <div className="max-w-7xl mx-auto px-4">
        <div className="flex justify-between h-16">
          <div className="flex space-x-8">
            <Link to="/" className="flex items-center text-xl font-bold">
              NutriLearn AI
            </Link>
            
            {authenticated && (
              <>
                <Link to="/dashboard" className="flex items-center hover:text-blue-600">
                  Dashboard
                </Link>
                <Link to="/analyze" className="flex items-center hover:text-blue-600">
                  Analyze
                </Link>
                <Link to="/learning" className="flex items-center hover:text-blue-600">
                  Learning
                </Link>
              </>
            )}
          </div>

          <div className="flex items-center space-x-4">
            {authenticated ? (
              <button
                onClick={handleLogout}
                className="px-4 py-2 text-sm font-medium text-white bg-red-600 rounded-md hover:bg-red-700"
              >
                Logout
              </button>
            ) : (
              <>
                <Link
                  to="/login"
                  className="px-4 py-2 text-sm font-medium text-gray-700 hover:text-blue-600"
                >
                  Login
                </Link>
                <Link
                  to="/signup"
                  className="px-4 py-2 text-sm font-medium text-white bg-blue-600 rounded-md hover:bg-blue-700"
                >
                  Sign Up
                </Link>
              </>
            )}
          </div>
        </div>
      </div>
    </nav>
  );
}

export default Navigation;
```

### 7. Updated App Component (`frontend/src/App.jsx`)

Main application component with routing configuration.

```javascript
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { useEffect } from 'react';
import Navigation from './components/Navigation';
import ProtectedRoute from './components/ProtectedRoute';
import HomePage from './pages/HomePage';
import LoginPage from './pages/LoginPage';
import SignupPage from './pages/SignupPage';
import DashboardPage from './pages/DashboardPage';
import AnalyzePage from './pages/AnalyzePage';
import LearningPage from './pages/LearningPage';
import MLOpsDashboard from './pages/MLOpsDashboard';
import { initializeAuth } from './utils/api';

function App() {
  useEffect(() => {
    // Initialize authentication on app load
    initializeAuth();
  }, []);

  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        <Navigation />
        <Routes>
          {/* Public routes */}
          <Route path="/" element={<HomePage />} />
          <Route path="/login" element={<LoginPage />} />
          <Route path="/signup" element={<SignupPage />} />

          {/* Protected routes */}
          <Route
            path="/dashboard"
            element={
              <ProtectedRoute>
                <DashboardPage />
              </ProtectedRoute>
            }
          />
          <Route
            path="/analyze"
            element={
              <ProtectedRoute>
                <AnalyzePage />
              </ProtectedRoute>
            }
          />
          <Route
            path="/learning"
            element={
              <ProtectedRoute>
                <LearningPage />
              </ProtectedRoute>
            }
          />
          <Route
            path="/mlops"
            element={
              <ProtectedRoute>
                <MLOpsDashboard />
              </ProtectedRoute>
            }
          />
        </Routes>
      </div>
    </Router>
  );
}

export default App;
```

## Data Models

### localStorage Structure

```javascript
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiI1NTBlODQwMC1lMjliLTQxZDQtYTcxNi00NDY2NTU0NDAwMDAiLCJleHAiOjE3MzU2ODk2MDAsImlhdCI6MTczNTA4NDgwMH0.signature"
}
```

### API Request Headers

```javascript
{
  "Content-Type": "application/json",
  "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

### Signup/Login Request Payload

```javascript
{
  "email": "user@example.com",
  "password": "SecurePass123!"
}
```

### Authentication Response

```javascript
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

### Error Response

```javascript
{
  "detail": "Invalid email or password",
  "status": "error"
}
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*


### Property 1: Successful authentication stores token and redirects
*For any* successful authentication response (signup or login) containing an access_token, the frontend should store the token in localStorage and redirect the user to the dashboard page.
**Validates: Requirements 1.3, 1.4, 2.3, 2.4**

### Property 2: Authentication errors display without redirect
*For any* authentication error response (signup or login), the frontend should display the error message to the user and should NOT redirect to any other page.
**Validates: Requirements 1.5, 2.5**

### Property 3: API requests include authentication token
*For any* API request to any endpoint when a JWT token exists in localStorage, the request should automatically include the token in the Authorization header as "Bearer {token}".
**Validates: Requirements 3.1, 3.2**

### Property 4: Token removal clears authorization headers
*For any* API request made after the JWT token is removed from localStorage, the request should NOT include an Authorization header.
**Validates: Requirements 3.3**

### Property 5: 401 responses trigger logout and redirect
*For any* API response with status 401 Unauthorized, the frontend should remove the JWT token from localStorage and redirect the user to the login page.
**Validates: Requirements 3.4, 10.2**

### Property 6: Logout clears all authentication state
*For any* logout action, the frontend should remove the JWT token from localStorage, clear the Authorization header from Axios, and redirect the user to the home or login page.
**Validates: Requirements 4.1, 4.2, 4.3**

### Property 7: Unauthenticated users cannot access protected routes
*For any* protected route, when a user without a valid JWT token attempts to access it, the frontend should redirect the user to the login page instead of rendering the protected content.
**Validates: Requirements 5.1**

### Property 8: Authenticated users can access protected routes
*For any* protected route, when a user with a valid JWT token attempts to access it, the frontend should render the requested page content without redirecting.
**Validates: Requirements 5.2, 5.4**

### Property 9: Invalid tokens are treated as unauthenticated
*For any* invalid or expired JWT token in localStorage, the frontend should treat the user as unauthenticated and redirect to login when attempting to access protected routes or after the first failed API request.
**Validates: Requirements 5.3, 8.3**

### Property 10: Authentication persists across page refreshes
*For any* page refresh when a valid JWT token exists in localStorage, the frontend should retrieve the token, configure Axios with the Authorization header, and maintain the authenticated state without requiring re-login.
**Validates: Requirements 8.1, 8.2**

### Property 11: Error messages are displayed appropriately
*For any* authentication error (validation error, network error, invalid credentials, duplicate email), the frontend should display a user-friendly error message that corresponds to the type of error received.
**Validates: Requirements 9.1, 9.2, 9.3, 9.4**

### Property 12: Token changes update request headers
*For any* change to the authentication token (login, signup, logout), the frontend should immediately update the Axios default headers to reflect the new token state for all subsequent requests.
**Validates: Requirements 10.3**

## Error Handling

### Client-Side Validation Errors

**Password Mismatch:**
- Occurs when password and confirm password fields don't match during signup
- Display: "Passwords do not match"
- Action: Prevent form submission, show error message

**Short Password:**
- Occurs when password is less than 8 characters
- Display: "Password must be at least 8 characters long"
- Action: Prevent form submission, show error message

**Invalid Email Format:**
- Occurs when email doesn't match email pattern
- Display: Browser's built-in validation message
- Action: Prevent form submission

### API Error Responses

**401 Unauthorized:**
- Invalid credentials during login
- Expired or invalid token on protected endpoints
- Display: "Invalid email or password" or "Session expired, please login again"
- Action: Show error message, redirect to login (for token errors)

**400 Bad Request:**
- Duplicate email during signup
- Display: "Email already registered"
- Action: Show error message, keep user on signup page

**422 Unprocessable Entity:**
- Backend validation errors (invalid email format, short password)
- Display: Specific validation error from backend
- Action: Show error message, keep user on current page

**Network Errors:**
- Connection timeout, server unreachable
- Display: "Connection error. Please check your internet connection and try again."
- Action: Show error message, allow retry

**500 Internal Server Error:**
- Unexpected backend errors
- Display: "Something went wrong. Please try again later."
- Action: Show error message, allow retry

### Error Display Strategy

All errors are displayed using a consistent error banner component:

```javascript
{error && (
  <div className="bg-red-50 text-red-600 p-3 rounded border border-red-200">
    {error}
  </div>
)}
```

Errors are cleared when:
- User starts typing in any form field
- User navigates to a different page
- User successfully completes an action

### Loading States

During async operations (login, signup), the UI should:
- Disable the submit button
- Show loading text ("Logging in...", "Creating account...")
- Prevent multiple submissions
- Maintain form data if operation fails

## Testing Strategy

### Unit Testing

Unit tests will verify specific examples and edge cases using **Jest** and **React Testing Library**:

**Authentication Utility Functions:**
- Test `signup()` stores token in localStorage and calls API
- Test `login()` stores token in localStorage and calls API
- Test `logout()` removes token from localStorage
- Test `getToken()` retrieves token from localStorage
- Test `isAuthenticated()` returns true when token exists
- Test `isAuthenticated()` returns false when token is null/undefined/empty

**API Utility Functions:**
- Test `setAuthToken()` sets Authorization header when token provided
- Test `setAuthToken(null)` removes Authorization header
- Test `initializeAuth()` loads token from localStorage on app start
- Test request interceptor adds token to requests
- Test response interceptor handles 401 by clearing token and redirecting

**Component Rendering:**
- Test LoginPage renders email and password inputs
- Test SignupPage renders email, password, and confirm password inputs
- Test ProtectedRoute redirects when not authenticated
- Test ProtectedRoute renders children when authenticated
- Test Navigation shows Login/Signup when not authenticated
- Test Navigation shows Logout when authenticated

**Form Submission:**
- Test LoginPage calls login() with form data on submit
- Test SignupPage calls signup() with form data on submit
- Test SignupPage validates password match before submission
- Test error messages display when API calls fail
- Test loading states during form submission

**Edge Cases:**
- Empty email or password fields
- Password exactly 8 characters (boundary)
- Password with 7 characters (should show error)
- Mismatched passwords in signup
- Network timeout errors
- Malformed API responses

### Property-Based Testing

Property-based tests will verify universal properties across many random inputs using **fast-check** library for JavaScript.

**Configuration:**
- Minimum 100 iterations per property test
- Use fast-check strategies for generating test data
- Tag each test with the property number from this design document

**Test Strategies:**

```javascript
import fc from 'fast-check';

// Email strategy: valid email addresses
const validEmail = fc.emailAddress();

// Password strategy: strings of 8-100 characters
const validPassword = fc.string({ minLength: 8, maxLength: 100 });

// Short password strategy: strings under 8 characters
const shortPassword = fc.string({ minLength: 0, maxLength: 7 });

// JWT token strategy: base64-like strings
const jwtToken = fc.string({ minLength: 20, maxLength: 500 });

// API error response strategy
const apiError = fc.record({
  response: fc.record({
    status: fc.integer({ min: 400, max: 599 }),
    data: fc.record({
      detail: fc.string(),
      message: fc.string()
    })
  })
});
```

**Property Tests:**

1. **Property 1 Test:** Generate random successful auth responses, verify token stored and redirect occurs
2. **Property 2 Test:** Generate random error responses, verify error displayed and no redirect
3. **Property 3 Test:** Generate random API requests with token present, verify Authorization header included
4. **Property 4 Test:** Remove token, make random requests, verify no Authorization header
5. **Property 5 Test:** Generate random 401 responses, verify token cleared and redirect occurs
6. **Property 6 Test:** Perform logout, verify token cleared, headers cleared, and redirect occurs
7. **Property 7 Test:** Generate random protected routes without token, verify redirect to login
8. **Property 8 Test:** Generate random protected routes with token, verify content renders
9. **Property 9 Test:** Generate random invalid tokens, verify treated as unauthenticated
10. **Property 10 Test:** Refresh page with token, verify auth state maintained
11. **Property 11 Test:** Generate random error types, verify appropriate messages displayed
12. **Property 12 Test:** Change token state, verify headers updated immediately

### Integration Testing

Integration tests will verify the complete authentication flow:

- **Full Signup Flow:** Fill form → Submit → API call → Token stored → Redirect to dashboard
- **Full Login Flow:** Fill form → Submit → API call → Token stored → Redirect to dashboard
- **Protected Route Flow:** Navigate to protected route → Check auth → Redirect if needed
- **Logout Flow:** Click logout → Token cleared → Headers cleared → Redirect to home
- **Token Expiration Flow:** Make API call with expired token → 401 response → Logout → Redirect to login
- **Page Refresh Flow:** Login → Refresh page → Auth maintained → Protected routes accessible

### Manual Testing Checklist

- [ ] Signup with valid credentials creates account and logs in
- [ ] Signup with duplicate email shows error
- [ ] Signup with mismatched passwords shows error
- [ ] Login with valid credentials logs in successfully
- [ ] Login with invalid credentials shows error
- [ ] Logout clears session and redirects
- [ ] Protected routes redirect to login when not authenticated
- [ ] Protected routes accessible when authenticated
- [ ] Navigation menu updates based on auth state
- [ ] Page refresh maintains authentication
- [ ] Token expiration triggers logout
- [ ] Network errors show appropriate messages

## Implementation Notes

### Dependencies

The following dependencies are already included in the project:

```json
{
  "react": "^18.2.0",
  "react-dom": "^18.2.0",
  "react-router-dom": "^6.x",
  "axios": "^1.x"
}
```

No additional dependencies are required for authentication functionality.

### Environment Variables

Add to `frontend/.env` and `frontend/.env.example`:

```bash
# API Configuration
VITE_API_URL=http://localhost:8000
```

### localStorage Security Considerations

**Advantages of localStorage for JWT:**
- Persists across browser sessions
- Simple API for storage and retrieval
- Automatically scoped to origin

**Security Considerations:**
- Vulnerable to XSS attacks (mitigated by React's built-in XSS protection)
- Not accessible from other domains (same-origin policy)
- Should only be used over HTTPS in production
- Tokens should have reasonable expiration (7 days)

**Alternative Approaches (Future Enhancements):**
- HttpOnly cookies (more secure but requires backend changes)
- sessionStorage (doesn't persist across tabs)
- In-memory storage (lost on page refresh)

### Migration Strategy

1. **Phase 1:** Create authentication utilities and components
2. **Phase 2:** Add login and signup pages with routing
3. **Phase 3:** Implement ProtectedRoute wrapper
4. **Phase 4:** Update Navigation component
5. **Phase 5:** Update App.jsx with protected routes
6. **Phase 6:** Test complete authentication flow
7. **Phase 7:** Remove getUserId() from storage.js (deprecated)

### Backward Compatibility

The authentication system is additive and doesn't break existing functionality:

- Public routes (home page) continue to work
- Prediction endpoint remains accessible without auth
- Existing components work unchanged
- Only new protected routes require authentication

### User Experience Considerations

**Loading States:**
- Show loading indicators during API calls
- Disable buttons during submission
- Prevent double-submission

**Error Messages:**
- Clear, actionable error messages
- Consistent error styling
- Errors clear when user takes action

**Navigation:**
- Smooth redirects after authentication
- Preserve intended destination (future enhancement)
- Clear visual feedback for auth state

**Form Validation:**
- Client-side validation for immediate feedback
- Server-side validation for security
- Helpful validation messages

## Performance Considerations

1. **Token Storage:** localStorage operations are synchronous but very fast
2. **Axios Interceptors:** Minimal overhead, run on every request/response
3. **Component Re-renders:** Use React.memo for Navigation if needed
4. **Route Protection:** ProtectedRoute check is instant (localStorage read)

## Security Checklist

- [ ] JWT tokens stored in localStorage (acceptable for this use case)
- [ ] Tokens automatically included in API requests
- [ ] Tokens cleared on logout
- [ ] Tokens cleared on 401 responses
- [ ] No sensitive data stored in localStorage besides token
- [ ] HTTPS used in production (deployment concern)
- [ ] Password validation on client and server
- [ ] Error messages don't reveal sensitive information
- [ ] XSS protection via React (automatic)
- [ ] CORS properly configured on backend

## Future Enhancements

1. **Remember Me:** Optional longer token expiration
2. **Redirect After Login:** Remember intended destination before login redirect
3. **Token Refresh:** Automatic token refresh before expiration
4. **Session Timeout Warning:** Warn user before token expires
5. **Multi-Tab Sync:** Sync auth state across browser tabs
6. **Social Login:** OAuth integration (Google, Facebook)
7. **Password Strength Indicator:** Visual feedback on password strength
8. **Email Verification:** Require email verification before full access
9. **Two-Factor Authentication:** Optional 2FA for enhanced security
10. **Account Settings:** Allow users to change email/password
