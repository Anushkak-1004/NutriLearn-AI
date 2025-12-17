# Implementation Plan: Frontend Authentication

## Task List

- [x] 1. Create authentication utility module


  - Create frontend/src/utils/auth.js
  - Implement signup() function that calls API and stores token
  - Implement login() function that calls API and stores token
  - Implement logout() function that clears token
  - Implement getToken() function that retrieves from localStorage
  - Implement isAuthenticated() function that checks token existence
  - Import api instance and setAuthToken from api.js
  - _Requirements: 1.2, 1.3, 2.2, 2.3, 4.1, 7.1, 7.2, 7.3, 7.4, 7.5_

- [ ]* 1.1 Write property test for token storage after authentication
  - **Property 1: Successful authentication stores token and redirects**
  - **Validates: Requirements 1.3, 1.4, 2.3, 2.4**

- [ ]* 1.2 Write unit tests for authentication utility functions
  - Test signup() stores token in localStorage
  - Test login() stores token in localStorage
  - Test logout() removes token from localStorage
  - Test getToken() retrieves token correctly
  - Test isAuthenticated() returns true when token exists
  - Test isAuthenticated() returns false when no token

- [x] 2. Update API utility module with authentication support


  - Update frontend/src/utils/api.js
  - Implement setAuthToken() function to set/clear Authorization header
  - Implement initializeAuth() function to load token on app start
  - Add request interceptor to include token in all requests
  - Add response interceptor to handle 401 errors (clear token and redirect)
  - Export setAuthToken and initializeAuth functions
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 8.2, 10.1, 10.2, 10.3_

- [ ]* 2.1 Write property test for automatic token inclusion
  - **Property 3: API requests include authentication token**
  - **Validates: Requirements 3.1, 3.2**

- [ ]* 2.2 Write property test for token removal
  - **Property 4: Token removal clears authorization headers**
  - **Validates: Requirements 3.3**

- [ ]* 2.3 Write property test for 401 response handling
  - **Property 5: 401 responses trigger logout and redirect**
  - **Validates: Requirements 3.4, 10.2**

- [ ]* 2.4 Write property test for token changes
  - **Property 12: Token changes update request headers**
  - **Validates: Requirements 10.3**

- [ ]* 2.5 Write unit tests for API utility functions
  - Test setAuthToken() sets Authorization header
  - Test setAuthToken(null) removes Authorization header
  - Test initializeAuth() loads token from localStorage
  - Test request interceptor adds token to requests
  - Test response interceptor handles 401 responses

- [x] 3. Create LoginPage component


  - Create frontend/src/pages/LoginPage.jsx
  - Implement form with email and password inputs
  - Add form validation (required fields)
  - Implement handleSubmit that calls auth.login()
  - Add error state and error message display
  - Add loading state during API call
  - Implement redirect to /dashboard on success using useNavigate
  - Add link to signup page
  - Style with Tailwind CSS
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 9.1, 9.2, 9.3_

- [ ]* 3.1 Write property test for login error handling
  - **Property 2: Authentication errors display without redirect**
  - **Validates: Requirements 1.5, 2.5**

- [ ]* 3.2 Write property test for error message display
  - **Property 11: Error messages are displayed appropriately**
  - **Validates: Requirements 9.1, 9.2, 9.3, 9.4**

- [ ]* 3.3 Write unit tests for LoginPage component
  - Test form renders with email and password inputs
  - Test form submission calls login() with correct data
  - Test error message displays when login fails
  - Test loading state during submission
  - Test redirect to dashboard on success

- [x] 4. Create SignupPage component


  - Create frontend/src/pages/SignupPage.jsx
  - Implement form with email, password, and confirm password inputs
  - Add client-side validation (password length, password match)
  - Implement handleSubmit that calls auth.signup()
  - Add error state and error message display
  - Add loading state during API call
  - Implement redirect to /dashboard on success using useNavigate
  - Add link to login page
  - Style with Tailwind CSS
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 9.1, 9.2, 9.4_

- [ ]* 4.1 Write unit tests for SignupPage component
  - Test form renders with all required inputs
  - Test password mismatch validation
  - Test short password validation
  - Test form submission calls signup() with correct data
  - Test error message displays when signup fails
  - Test loading state during submission
  - Test redirect to dashboard on success

- [x] 5. Create ProtectedRoute component


  - Create frontend/src/components/ProtectedRoute.jsx
  - Import isAuthenticated from auth.js
  - Check authentication status using isAuthenticated()
  - Render children if authenticated
  - Redirect to /login using Navigate if not authenticated
  - Use replace prop to prevent back button issues
  - _Requirements: 5.1, 5.2, 5.3, 5.4_

- [ ]* 5.1 Write property test for unauthenticated route protection
  - **Property 7: Unauthenticated users cannot access protected routes**
  - **Validates: Requirements 5.1**

- [ ]* 5.2 Write property test for authenticated route access
  - **Property 8: Authenticated users can access protected routes**
  - **Validates: Requirements 5.2, 5.4**

- [ ]* 5.3 Write property test for invalid token handling
  - **Property 9: Invalid tokens are treated as unauthenticated**
  - **Validates: Requirements 5.3, 8.3**

- [ ]* 5.4 Write unit tests for ProtectedRoute component
  - Test redirects to /login when not authenticated
  - Test renders children when authenticated
  - Test uses replace prop for redirect

- [x] 6. Update Navigation component


  - Update frontend/src/components/Navigation.jsx
  - Import isAuthenticated and logout from auth.js
  - Add state to track authentication status
  - Conditionally render Login/Signup links when not authenticated
  - Conditionally render Logout button when authenticated
  - Conditionally show/hide protected route links based on auth status
  - Implement handleLogout that calls logout() and redirects
  - _Requirements: 4.2, 6.1, 6.2, 6.3, 6.4_

- [ ]* 6.1 Write property test for logout functionality
  - **Property 6: Logout clears all authentication state**
  - **Validates: Requirements 4.1, 4.2, 4.3**

- [ ]* 6.2 Write unit tests for Navigation component
  - Test shows Login/Signup when not authenticated
  - Test shows Logout when authenticated
  - Test hides protected links when not authenticated
  - Test shows protected links when authenticated
  - Test logout button calls logout() and redirects

- [x] 7. Update App component with authentication routes


  - Update frontend/src/App.jsx
  - Import LoginPage, SignupPage, and ProtectedRoute
  - Import initializeAuth from api.js
  - Add useEffect to call initializeAuth() on mount
  - Add /login route with LoginPage component
  - Add /signup route with SignupPage component
  - Wrap /dashboard route with ProtectedRoute
  - Wrap /analyze route with ProtectedRoute
  - Wrap /learning route with ProtectedRoute
  - Wrap /mlops route with ProtectedRoute (if exists)
  - _Requirements: 5.1, 5.2, 8.1, 8.2_

- [ ]* 7.1 Write property test for authentication persistence
  - **Property 10: Authentication persists across page refreshes**
  - **Validates: Requirements 8.1, 8.2**

- [ ]* 7.2 Write integration tests for complete authentication flows
  - Test full signup flow (form → API → token → redirect)
  - Test full login flow (form → API → token → redirect)
  - Test protected route flow (navigate → check auth → redirect)
  - Test logout flow (click → clear → redirect)
  - Test page refresh maintains auth state

- [x] 8. Remove deprecated getUserId function


  - Update frontend/src/utils/storage.js
  - Remove getUserId() function (now use token-based identification)
  - Update any components that use getUserId() to use token instead
  - _Requirements: All (cleanup task)_

- [x] 9. Update environment configuration


  - Update frontend/.env.example
  - Add VITE_API_URL with default value http://localhost:8000
  - Ensure frontend/.env has VITE_API_URL configured
  - _Requirements: All_

- [x] 10. Checkpoint - Ensure all tests pass


  - Ensure all tests pass, ask the user if questions arise.

- [x] 11. Manual testing and validation


  - Test signup with valid credentials
  - Test signup with duplicate email
  - Test signup with mismatched passwords
  - Test login with valid credentials
  - Test login with invalid credentials
  - Test logout functionality
  - Test protected route access without auth
  - Test protected route access with auth
  - Test navigation menu updates
  - Test page refresh maintains auth
  - Test token expiration handling
  - _Requirements: All_

- [x] 12. Update documentation



  - Update frontend/FRONTEND_GUIDE.md with authentication section
  - Document authentication flow
  - Document how to use protected routes
  - Document localStorage token storage
  - Add troubleshooting section for common auth issues
  - _Requirements: All_
