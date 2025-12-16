# JWT Authentication System Design

## Overview

This document outlines the design for implementing JWT (JSON Web Token) based authentication in the NutriLearn AI backend API. The authentication system will provide secure user registration, login, and token-based access control for protected endpoints while maintaining public access to prediction and health check endpoints.

The design follows industry best practices for security, including bcrypt password hashing, environment-based secret management, and dependency injection for authentication middleware. The system integrates seamlessly with the existing FastAPI application and Supabase database infrastructure.

## Architecture

### High-Level Architecture

```
┌─────────────────┐
│   Frontend      │
│   (React)       │
└────────┬────────┘
         │ HTTP + JWT Token
         ▼
┌─────────────────────────────────────────┐
│         FastAPI Application             │
│  ┌───────────────────────────────────┐  │
│  │   Auth Routes                     │  │
│  │   - POST /auth/signup             │  │
│  │   - POST /auth/login              │  │
│  │   - GET  /auth/me                 │  │
│  └───────────────────────────────────┘  │
│  ┌───────────────────────────────────┐  │
│  │   Protected Routes                │  │
│  │   - POST /meals/log               │  │
│  │   - GET  /users/{id}/stats        │  │
│  │   (requires JWT token)            │  │
│  └───────────────────────────────────┘  │
│  ┌───────────────────────────────────┐  │
│  │   Public Routes                   │  │
│  │   - POST /predict                 │  │
│  │   - GET  /health                  │  │
│  │   (no authentication)             │  │
│  └───────────────────────────────────┘  │
│  ┌───────────────────────────────────┐  │
│  │   Auth Middleware                 │  │
│  │   - Token validation              │  │
│  │   - User extraction               │  │
│  └───────────────────────────────────┘  │
└────────────┬────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────┐
│      Supabase PostgreSQL Database       │
│  ┌───────────────────────────────────┐  │
│  │   users table                     │  │
│  │   - id (UUID, PK)                 │  │
│  │   - email (unique)                │  │
│  │   - password_hash                 │  │
│  │   - created_at                    │  │
│  │   - updated_at                    │  │
│  └───────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

### Authentication Flow

**Registration Flow:**
```
User → POST /auth/signup → Validate email/password → Hash password → 
Store in DB → Generate JWT → Return token
```

**Login Flow:**
```
User → POST /auth/login → Fetch user by email → Verify password hash → 
Generate JWT → Return token
```

**Protected Endpoint Access:**
```
User → Request with Authorization header → Extract token → Verify signature → 
Decode payload → Extract user_id → Execute endpoint logic
```

## Components and Interfaces

### 1. Authentication Module (`backend/app/auth.py`)

Core authentication utilities for token generation, validation, and password hashing.

**Functions:**

```python
def generate_token(user_id: str, expires_delta: Optional[timedelta] = None) -> str:
    """
    Generate a JWT access token for a user.
    
    Args:
        user_id: User identifier to encode in token
        expires_delta: Optional custom expiration time (default: 7 days)
        
    Returns:
        Encoded JWT token string
        
    Raises:
        Exception: If token generation fails
    """

def verify_token(token: str) -> Dict[str, Any]:
    """
    Verify and decode a JWT token.
    
    Args:
        token: JWT token string to verify
        
    Returns:
        Decoded token payload containing user_id and expiration
        
    Raises:
        JWTError: If token is invalid, expired, or malformed
    """

def get_password_hash(password: str) -> str:
    """
    Hash a password using bcrypt.
    
    Args:
        password: Plain text password
        
    Returns:
        Bcrypt hashed password string
    """

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.
    
    Args:
        plain_password: Plain text password to verify
        hashed_password: Stored bcrypt hash
        
    Returns:
        True if password matches, False otherwise
    """
```

**Configuration:**
- `SECRET_KEY`: Loaded from environment variable (generated with `openssl rand -hex 32`)
- `ALGORITHM`: HS256 (HMAC with SHA-256)
- `ACCESS_TOKEN_EXPIRE_DAYS`: 7 days default expiration

### 2. Pydantic Models (`backend/app/models.py`)

Data validation models for authentication requests and responses.

```python
class UserCreate(BaseModel):
    """Request model for user registration."""
    email: EmailStr
    password: str = Field(..., min_length=8, description="Password must be at least 8 characters")

class UserLogin(BaseModel):
    """Request model for user login."""
    email: EmailStr
    password: str

class Token(BaseModel):
    """Response model for authentication tokens."""
    access_token: str
    token_type: str = "bearer"

class UserResponse(BaseModel):
    """Response model for user information."""
    user_id: str
    email: str
    created_at: datetime
```

### 3. Authentication Routes (`backend/app/api/auth_routes.py`)

API endpoints for authentication operations.

**Endpoints:**

```python
@router.post("/api/v1/auth/signup", response_model=Token, status_code=201)
async def signup(user_data: UserCreate):
    """
    Register a new user account.
    
    Process:
    1. Validate email format and password length
    2. Check if email already exists
    3. Hash password with bcrypt
    4. Insert user into database
    5. Generate JWT token
    6. Return token
    
    Returns:
        Token with access_token and token_type
        
    Raises:
        HTTPException 400: If email already exists
        HTTPException 422: If validation fails
        HTTPException 500: If database operation fails
    """

@router.post("/api/v1/auth/login", response_model=Token)
async def login(credentials: UserLogin):
    """
    Authenticate user and return access token.
    
    Process:
    1. Fetch user by email
    2. Verify password against stored hash
    3. Generate JWT token
    4. Return token
    
    Returns:
        Token with access_token and token_type
        
    Raises:
        HTTPException 401: If credentials are invalid
        HTTPException 500: If database operation fails
    """

@router.get("/api/v1/auth/me", response_model=UserResponse)
async def get_current_user_info(current_user_id: str = Depends(get_current_user)):
    """
    Get authenticated user's profile information.
    
    Process:
    1. Extract user_id from validated JWT token
    2. Fetch user details from database
    3. Return user information (excluding password)
    
    Returns:
        UserResponse with user_id, email, and created_at
        
    Raises:
        HTTPException 401: If token is invalid
        HTTPException 404: If user not found
    """
```

### 4. Authentication Dependencies (`backend/app/dependencies.py`)

Dependency injection for protecting routes with authentication.

```python
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Optional

# OAuth2 scheme for extracting Bearer tokens
security = HTTPBearer()

async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    """
    Dependency to extract and validate JWT token from request.
    
    This dependency:
    1. Extracts Bearer token from Authorization header
    2. Verifies token signature and expiration
    3. Decodes token payload
    4. Returns authenticated user_id
    
    Args:
        credentials: HTTP Bearer credentials from request header
        
    Returns:
        user_id: Authenticated user's ID
        
    Raises:
        HTTPException 401: If token is missing, invalid, or expired
        
    Usage:
        @router.get("/protected")
        async def protected_route(user_id: str = Depends(get_current_user)):
            # user_id is automatically extracted and validated
            return {"user_id": user_id}
    """

async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False))
) -> Optional[str]:
    """
    Optional authentication dependency for endpoints that work with or without auth.
    
    Returns:
        user_id if token is valid, None if no token provided
    """
```

### 5. Database Schema Updates

**Users Table Enhancement:**

```sql
-- Add authentication columns to existing users table
ALTER TABLE users ADD COLUMN IF NOT EXISTS email VARCHAR(255) UNIQUE NOT NULL;
ALTER TABLE users ADD COLUMN IF NOT EXISTS password_hash TEXT NOT NULL;
ALTER TABLE users ADD COLUMN IF NOT EXISTS created_at TIMESTAMP DEFAULT NOW();

-- Create index on email for faster lookups
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);
```

**Database Functions (`backend/app/database.py`):**

```python
def create_user(email: str, password_hash: str) -> str:
    """
    Create a new user in the database.
    
    Args:
        email: User's email address
        password_hash: Bcrypt hashed password
        
    Returns:
        user_id: UUID of created user
        
    Raises:
        Exception: If email already exists or database operation fails
    """

def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """
    Fetch user by email address.
    
    Args:
        email: User's email address
        
    Returns:
        User dictionary with id, email, password_hash, created_at
        Returns None if user not found
    """

def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch user by ID.
    
    Args:
        user_id: User's UUID
        
    Returns:
        User dictionary with id, email, created_at (no password_hash)
        Returns None if user not found
    """
```

## Data Models

### JWT Token Payload

```json
{
  "sub": "user_uuid_here",
  "exp": 1735689600,
  "iat": 1735084800
}
```

- `sub` (subject): User ID (UUID)
- `exp` (expiration): Unix timestamp when token expires
- `iat` (issued at): Unix timestamp when token was created

### User Database Record

```json
{
  "id": "550e8400-e29b-41d4-a716-446655440000",
  "email": "user@example.com",
  "password_hash": "$2b$12$KIXxJ...",
  "total_meals": 0,
  "total_points": 0,
  "current_streak": 0,
  "created_at": "2025-12-17T10:30:00Z",
  "updated_at": "2025-12-17T10:30:00Z"
}
```

### Authentication Request/Response Examples

**Signup Request:**
```json
{
  "email": "user@example.com",
  "password": "SecurePass123!"
}
```

**Login Request:**
```json
{
  "email": "user@example.com",
  "password": "SecurePass123!"
}
```

**Token Response:**
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer"
}
```

**User Profile Response:**
```json
{
  "user_id": "550e8400-e29b-41d4-a716-446655440000",
  "email": "user@example.com",
  "created_at": "2025-12-17T10:30:00Z"
}
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*


### Property 1: Valid signup creates user with hashed password and returns token
*For any* valid email and password (8+ characters), submitting to the signup endpoint should create a user record with a bcrypt-hashed password (not plain text) and return a valid JWT token that can be decoded to extract the user_id.
**Validates: Requirements 1.1, 1.3**

### Property 2: Invalid email formats are rejected during signup
*For any* string that is not a valid email format (missing @, invalid domain, etc.), submitting to the signup endpoint should reject the request with a validation error.
**Validates: Requirements 1.4**

### Property 3: Valid login returns token with correct expiration
*For any* user with known credentials, logging in with correct email and password should return a JWT token with an expiration time set to 7 days from the current time.
**Validates: Requirements 2.1, 2.4**

### Property 4: Generated tokens contain user_id in payload
*For any* user authentication (signup or login), the returned JWT token should contain the user's ID in the "sub" field of the decoded payload.
**Validates: Requirements 2.5**

### Property 5: Valid tokens grant access to protected endpoints
*For any* protected endpoint and valid JWT token, including the token in the Authorization header should allow the request to proceed and correctly extract the user_id from the token payload.
**Validates: Requirements 3.1**

### Property 6: Invalid or malformed tokens are rejected
*For any* protected endpoint and malformed JWT token (wrong signature, corrupted payload, invalid format), the request should be rejected with an authentication error.
**Validates: Requirements 3.3, 3.5**

### Property 7: Profile requests return user data without password
*For any* authenticated user requesting their profile, the response should include user_id, email, and created_at, but should NOT include the password_hash field.
**Validates: Requirements 4.1, 4.2**

### Property 8: Password verification correctly validates credentials
*For any* user with a stored password hash, verifying the correct plain text password should return true, and verifying any incorrect password should return false.
**Validates: Requirements 5.2**

### Property 9: JWT tokens use HS256 algorithm
*For any* generated JWT token, decoding the token header should show the algorithm is "HS256".
**Validates: Requirements 5.4**

### Property 10: Meal logging extracts user_id from token
*For any* authenticated meal logging request, the user_id associated with the meal should come from the JWT token, not from the request body, ensuring users can only log meals for themselves.
**Validates: Requirements 9.1**

### Property 11: Users cannot access other users' data
*For any* two different users A and B, when user A attempts to access user B's statistics endpoint with user A's valid token, the request should be rejected with an authorization error.
**Validates: Requirements 9.2**

## Error Handling

### Authentication Errors

**Invalid Credentials (401 Unauthorized):**
- Wrong password during login
- Non-existent email during login
- Generic message: "Invalid email or password" (don't reveal which is wrong)

**Token Errors (401 Unauthorized):**
- Missing Authorization header
- Malformed token format
- Invalid token signature
- Expired token
- Token with invalid payload

**Validation Errors (422 Unprocessable Entity):**
- Invalid email format
- Password too short (< 8 characters)
- Missing required fields

**Conflict Errors (400 Bad Request):**
- Email already registered during signup

**Authorization Errors (403 Forbidden):**
- Attempting to access another user's resources
- Valid token but insufficient permissions

### Error Response Format

All authentication errors follow a consistent format:

```json
{
  "status": "error",
  "message": "Human-readable error message",
  "detail": "Additional context (optional)",
  "timestamp": "2025-12-17T10:30:00Z"
}
```

### Security Considerations

1. **Password Security:**
   - Never log passwords (plain or hashed)
   - Use bcrypt with appropriate cost factor (12 rounds)
   - Enforce minimum password length (8 characters)

2. **Token Security:**
   - Store SECRET_KEY in environment variables only
   - Use strong random secret (32+ bytes)
   - Set reasonable expiration (7 days)
   - Validate signature on every request

3. **Information Disclosure:**
   - Don't reveal if email exists during login failures
   - Don't reveal if email exists during signup failures (use generic "already registered")
   - Don't include password hashes in any API responses

4. **Rate Limiting (Future Enhancement):**
   - Limit login attempts per IP
   - Limit signup attempts per IP
   - Implement exponential backoff for failed attempts

## Testing Strategy

### Unit Testing

Unit tests will verify specific examples and edge cases:

**Authentication Functions:**
- Test `generate_token()` creates valid JWT with correct structure
- Test `verify_token()` correctly decodes valid tokens
- Test `verify_token()` raises error for expired tokens
- Test `get_password_hash()` produces bcrypt hashes
- Test `verify_password()` correctly validates passwords

**API Endpoints:**
- Test signup with valid data creates user
- Test signup with duplicate email returns 400
- Test login with correct credentials returns token
- Test login with wrong password returns 401
- Test /auth/me with valid token returns user info
- Test /auth/me without token returns 401
- Test public endpoints work without authentication
- Test protected endpoints require authentication

**Edge Cases:**
- Empty email or password
- Password exactly 8 characters (boundary)
- Password with 7 characters (should fail)
- Very long passwords (1000+ characters)
- Special characters in passwords
- Unicode characters in email

### Property-Based Testing

Property-based tests will verify universal properties across many random inputs using the **Hypothesis** library for Python.

**Configuration:**
- Minimum 100 iterations per property test
- Use Hypothesis strategies for generating test data
- Tag each test with the property number from this design document

**Test Strategies:**

```python
from hypothesis import given, strategies as st
from hypothesis.strategies import emails, text

# Email strategy: valid email addresses
valid_emails = emails()

# Password strategy: strings of 8-100 characters
valid_passwords = text(min_size=8, max_size=100, alphabet=st.characters(blacklist_categories=('Cs',)))

# Invalid email strategy: strings that aren't valid emails
invalid_emails = text(min_size=1, max_size=100).filter(lambda x: '@' not in x or '.' not in x)

# Short password strategy: strings under 8 characters
short_passwords = text(min_size=0, max_size=7)
```

**Property Tests:**

1. **Property 1 Test:** Generate random valid emails and passwords, signup, verify user created with bcrypt hash and valid token returned
2. **Property 2 Test:** Generate random invalid email formats, verify signup rejects them
3. **Property 3 Test:** Create users, login with correct credentials, verify token expiration is ~7 days
4. **Property 4 Test:** Generate random user signups/logins, decode tokens, verify user_id in payload
5. **Property 5 Test:** Generate random valid tokens, make requests to protected endpoints, verify access granted
6. **Property 6 Test:** Generate random malformed tokens, verify protected endpoints reject them
7. **Property 7 Test:** Create random users, request profiles, verify password_hash not in response
8. **Property 8 Test:** Create users with random passwords, verify correct passwords validate and wrong ones don't
9. **Property 9 Test:** Generate random tokens, decode headers, verify algorithm is HS256
10. **Property 10 Test:** Create users, log meals with tokens, verify user_id from token is used
11. **Property 11 Test:** Create two users, verify user A cannot access user B's stats

### Integration Testing

Integration tests will verify the complete authentication flow:

- **Full Registration Flow:** Signup → Receive token → Access protected endpoint
- **Full Login Flow:** Create user → Login → Receive token → Access protected endpoint
- **Token Expiration Flow:** Create expired token → Attempt access → Verify rejection
- **Authorization Flow:** User A creates data → User B attempts access → Verify rejection
- **Public Endpoint Flow:** Access /predict and /health without token → Verify success

### Test Database

- Use separate test database or in-memory storage for tests
- Clean up test data after each test
- Use fixtures for common test scenarios (authenticated user, multiple users, etc.)

## Implementation Notes

### Dependencies to Add

```txt
python-jose[cryptography]==3.3.0  # JWT token handling
passlib[bcrypt]==1.7.4            # Password hashing
python-multipart==0.0.6           # Already in requirements (form data)
```

### Environment Variables

Add to `.env` and `.env.example`:

```bash
# JWT Configuration
JWT_SECRET_KEY=your-secret-key-here  # Generate with: openssl rand -hex 32
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_DAYS=7
```

### Migration Strategy

1. **Database Migration:** Add email, password_hash, created_at columns to users table
2. **Backward Compatibility:** Existing endpoints continue to work during migration
3. **Gradual Rollout:** 
   - Phase 1: Add auth endpoints (signup, login, /me)
   - Phase 2: Add authentication to meal logging
   - Phase 3: Add authentication to user stats
   - Phase 4: Add authorization checks for user-specific data

### Protected vs Public Endpoints

**Protected (require authentication):**
- `POST /api/v1/meals/log`
- `GET /api/v1/users/{user_id}/stats`
- `GET /api/v1/users/{user_id}/analysis`
- `GET /api/v1/users/{user_id}/meals`
- `POST /api/v1/modules/{module_id}/complete`
- `GET /api/v1/auth/me`

**Public (no authentication required):**
- `POST /api/v1/predict`
- `GET /health`
- `GET /`
- `POST /api/v1/auth/signup`
- `POST /api/v1/auth/login`
- `GET /api/docs` (Swagger UI)
- `GET /api/redoc` (ReDoc)

### Frontend Integration

The frontend will need to:

1. **Store Token:** Save JWT token in localStorage after signup/login
2. **Include Token:** Add `Authorization: Bearer <token>` header to all protected requests
3. **Handle 401:** Redirect to login page when token expires
4. **Clear Token:** Remove token from localStorage on logout

Example frontend code:

```javascript
// Store token after login
localStorage.setItem('token', response.data.access_token);

// Include token in requests
const token = localStorage.getItem('token');
axios.post('/api/v1/meals/log', data, {
  headers: { Authorization: `Bearer ${token}` }
});

// Handle 401 errors
axios.interceptors.response.use(
  response => response,
  error => {
    if (error.response?.status === 401) {
      localStorage.removeItem('token');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);
```

## Performance Considerations

1. **Token Validation:** JWT validation is fast (cryptographic signature check)
2. **Password Hashing:** Bcrypt is intentionally slow (security feature) - acceptable for login/signup
3. **Database Queries:** Index on email column for fast user lookups
4. **Caching:** Consider caching user data for frequently accessed profiles (future enhancement)

## Security Audit Checklist

- [ ] SECRET_KEY is loaded from environment variables
- [ ] SECRET_KEY is not committed to version control
- [ ] Passwords are hashed with bcrypt before storage
- [ ] Password hashes are never returned in API responses
- [ ] Login errors don't reveal if email exists
- [ ] Tokens have reasonable expiration (7 days)
- [ ] Token signatures are validated on every request
- [ ] Protected endpoints require valid tokens
- [ ] Users cannot access other users' data
- [ ] HTTPS is used in production (handled by deployment)
- [ ] CORS is properly configured

## Future Enhancements

1. **Refresh Tokens:** Long-lived refresh tokens for better UX
2. **Password Reset:** Email-based password reset flow
3. **Email Verification:** Verify email addresses before activation
4. **OAuth Integration:** Social login (Google, Facebook)
5. **Two-Factor Authentication:** TOTP-based 2FA
6. **Session Management:** Track active sessions, allow logout from all devices
7. **Rate Limiting:** Prevent brute force attacks
8. **Audit Logging:** Log all authentication events
9. **Password Strength Meter:** Client-side password strength validation
10. **Account Deletion:** Allow users to delete their accounts
