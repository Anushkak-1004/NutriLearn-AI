# Implementation Plan: JWT Authentication

## Task List

- [x] 1. Update dependencies and environment configuration





  - Add python-jose[cryptography]==3.3.0 to requirements.txt
  - Add passlib[bcrypt]==1.7.4 to requirements.txt
  - Update .env.example with JWT configuration variables (JWT_SECRET_KEY, JWT_ALGORITHM, ACCESS_TOKEN_EXPIRE_DAYS)
  - Generate and add JWT_SECRET_KEY to .env file
  - _Requirements: 5.3, 5.4_

- [x] 2. Create authentication utilities module





  - Create backend/app/auth.py with core authentication functions
  - Implement generate_token() function for JWT creation with 7-day expiration
  - Implement verify_token() function for JWT validation and decoding
  - Implement get_password_hash() function using bcrypt
  - Implement verify_password() function for password verification
  - Load SECRET_KEY, ALGORITHM, and expiration settings from environment variables
  - _Requirements: 1.3, 2.4, 2.5, 5.1, 5.2, 5.4_

- [ ]* 2.1 Write property test for password hashing
  - **Property 1: Valid signup creates user with hashed password and returns token**
  - **Validates: Requirements 1.1, 1.3**

- [ ]* 2.2 Write property test for password verification
  - **Property 8: Password verification correctly validates credentials**
  - **Validates: Requirements 5.2**

- [ ]* 2.3 Write property test for token generation
  - **Property 4: Generated tokens contain user_id in payload**
  - **Validates: Requirements 2.5**

- [ ]* 2.4 Write property test for JWT algorithm
  - **Property 9: JWT tokens use HS256 algorithm**
  - **Validates: Requirements 5.4**

- [x] 3. Update Pydantic models for authentication




  - Add UserCreate model to backend/app/models.py (email: EmailStr, password: str with min_length=8)
  - Add UserLogin model (email: EmailStr, password: str)
  - Add Token model (access_token: str, token_type: str)
  - Add UserResponse model (user_id: str, email: str, created_at: datetime)
  - _Requirements: 1.4, 1.5, 2.1_

- [x] 4. Update database schema for authentication





  - Create migration file backend/migrations/003_add_authentication.sql
  - Add email column (VARCHAR(255), UNIQUE, NOT NULL) to users table
  - Add password_hash column (TEXT, NOT NULL) to users table
  - Add created_at column (TIMESTAMP, DEFAULT NOW()) to users table
  - Create index on email column for fast lookups
  - _Requirements: 8.1, 8.2, 8.3_

- [x] 5. Implement database functions for user management





  - Add create_user() function to backend/app/database.py
  - Add get_user_by_email() function to backend/app/database.py
  - Add get_user_by_id() function to backend/app/database.py
  - Handle both Supabase and in-memory storage implementations
  - Ensure password_hash is excluded from get_user_by_id() response
  - _Requirements: 1.1, 1.2, 2.1, 4.2, 8.4_

- [ ]* 5.1 Write unit tests for database user functions
  - Test create_user() with valid data
  - Test create_user() with duplicate email raises error
  - Test get_user_by_email() returns correct user
  - Test get_user_by_email() returns None for non-existent email
  - Test get_user_by_id() excludes password_hash

- [x] 6. Create authentication dependency for route protection





  - Create backend/app/dependencies.py
  - Implement HTTPBearer security scheme
  - Implement get_current_user() dependency that extracts and validates JWT tokens
  - Implement get_current_user_optional() dependency for optional authentication
  - Handle token extraction from Authorization header
  - Handle token validation errors (expired, invalid, malformed, missing)
  - Return user_id from validated token
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5, 6.1, 6.2, 6.3, 6.4_

- [ ]* 6.1 Write property test for token validation
  - **Property 5: Valid tokens grant access to protected endpoints**
  - **Validates: Requirements 3.1**

- [ ]* 6.2 Write property test for invalid token rejection
  - **Property 6: Invalid or malformed tokens are rejected**
  - **Validates: Requirements 3.3, 3.5**

- [ ]* 6.3 Write unit tests for authentication dependency
  - Test get_current_user() with valid token returns user_id
  - Test get_current_user() with expired token raises 401
  - Test get_current_user() with malformed token raises 401
  - Test get_current_user() without token raises 401

- [x] 7. Create authentication API routes





  - Create backend/app/api/auth_routes.py with APIRouter
  - Implement POST /api/v1/auth/signup endpoint
  - Implement POST /api/v1/auth/login endpoint
  - Implement GET /api/v1/auth/me endpoint (requires authentication)
  - Handle validation errors with appropriate HTTP status codes
  - Handle duplicate email errors during signup
  - Handle invalid credentials during login with generic error message
  - Return Token model from signup and login endpoints
  - Return UserResponse model from /me endpoint
  - _Requirements: 1.1, 1.2, 1.4, 1.5, 2.1, 2.2, 2.3, 4.1, 4.2, 4.3_

- [ ]* 7.1 Write property test for signup endpoint
  - **Property 1: Valid signup creates user with hashed password and returns token**
  - **Validates: Requirements 1.1, 1.3**

- [ ]* 7.2 Write property test for invalid email rejection
  - **Property 2: Invalid email formats are rejected during signup**
  - **Validates: Requirements 1.4**

- [ ]* 7.3 Write property test for login endpoint
  - **Property 3: Valid login returns token with correct expiration**
  - **Validates: Requirements 2.1, 2.4**

- [ ]* 7.4 Write property test for profile endpoint
  - **Property 7: Profile requests return user data without password**
  - **Validates: Requirements 4.1, 4.2**

- [ ]* 7.5 Write unit tests for authentication routes
  - Test signup with valid data returns token
  - Test signup with duplicate email returns 400
  - Test signup with invalid email returns 422
  - Test signup with short password returns 422
  - Test login with correct credentials returns token
  - Test login with wrong password returns 401
  - Test login with non-existent email returns 401
  - Test /me with valid token returns user info
  - Test /me without token returns 401

- [x] 8. Register authentication routes in main application


  - Import auth_routes router in backend/app/main.py
  - Include auth_routes router with app.include_router()
  - Ensure auth routes are registered before application startup
  - _Requirements: 1.1, 2.1, 4.1_

- [x] 9. Checkpoint - Ensure all tests pass


  - Ensure all tests pass, ask the user if questions arise.

- [x] 10. Update meal logging endpoint to use authentication


  - Add get_current_user dependency to POST /api/v1/meals/log endpoint in backend/app/api/routes.py
  - Extract user_id from authenticated token instead of request body
  - Remove user_id from MealLogRequest model (or make it optional and ignored)
  - Update endpoint to use authenticated user_id for meal logging
  - _Requirements: 9.1_

- [ ]* 10.1 Write property test for authenticated meal logging
  - **Property 10: Meal logging extracts user_id from token**
  - **Validates: Requirements 9.1**

- [ ]* 10.2 Write unit tests for authenticated meal logging
  - Test meal logging with valid token uses token user_id
  - Test meal logging without token returns 401
  - Test meal logging with invalid token returns 401

- [x] 11. Update user statistics endpoint to use authorization


  - Add get_current_user dependency to GET /api/v1/users/{user_id}/stats endpoint
  - Verify that authenticated user_id matches the user_id in URL path
  - Return 403 Forbidden if user attempts to access another user's stats
  - _Requirements: 9.2, 9.3, 9.4_

- [ ]* 11.1 Write property test for user authorization
  - **Property 11: Users cannot access other users' data**
  - **Validates: Requirements 9.2**

- [ ]* 11.2 Write unit tests for user authorization
  - Test user can access their own stats
  - Test user cannot access another user's stats (returns 403)
  - Test stats endpoint without token returns 401

- [x] 12. Update remaining user-specific endpoints with authentication


  - Add get_current_user dependency to GET /api/v1/users/{user_id}/analysis
  - Add get_current_user dependency to GET /api/v1/users/{user_id}/meals
  - Add authorization checks to verify user_id matches authenticated user
  - Add get_current_user dependency to POST /api/v1/modules/{module_id}/complete
  - Update ModuleCompletionRequest to extract user_id from token instead of body
  - _Requirements: 9.2, 9.4_

- [ ]* 12.1 Write unit tests for protected endpoints
  - Test analysis endpoint requires authentication
  - Test meals endpoint requires authentication
  - Test module completion requires authentication
  - Test authorization checks for user-specific endpoints

- [x] 13. Verify public endpoints remain accessible


  - Verify POST /api/v1/predict works without authentication
  - Verify GET /health works without authentication
  - Verify GET / works without authentication
  - Verify GET /api/docs works without authentication
  - _Requirements: 7.1, 7.2_

- [ ]* 13.1 Write unit tests for public endpoints
  - Test /predict endpoint works without token
  - Test /health endpoint works without token
  - Test root endpoint works without token

- [x] 14. Update API documentation


  - Add authentication section to backend/API_DOCUMENTATION.md
  - Document signup endpoint with request/response examples
  - Document login endpoint with request/response examples
  - Document /me endpoint with authentication requirement
  - Document how to include Bearer token in requests
  - Document error responses for authentication failures
  - Update protected endpoint documentation to indicate authentication requirement
  - _Requirements: All_

- [x] 15. Create authentication testing guide



  - Create backend/tests/test_authentication.py with comprehensive test suite
  - Include property-based tests using Hypothesis
  - Include unit tests for all authentication functions
  - Include integration tests for complete authentication flows
  - Document how to run authentication tests
  - _Requirements: All_

- [-] 16. Final checkpoint - Ensure all tests pass



  - Ensure all tests pass, ask the user if questions arise.
