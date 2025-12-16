"""
Authentication API Routes

This module provides authentication endpoints for:
- User registration (signup)
- User login
- User profile retrieval

All endpoints follow RESTful conventions and return consistent response formats.
"""

import logging
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, status, Depends

from app.models import UserCreate, UserLogin, Token, UserResponse
from app.auth import generate_token, get_password_hash, verify_password
from app.database import create_user, get_user_by_email, get_user_by_id
from app.dependencies import get_current_user

# Configure logging
logger = logging.getLogger(__name__)

# Create router with prefix and tags
router = APIRouter(
    prefix="/api/v1/auth",
    tags=["authentication"]
)


@router.post("/signup", response_model=Token, status_code=status.HTTP_201_CREATED)
async def signup(user_data: UserCreate) -> Token:
    """
    Register a new user account.
    
    Process:
    1. Validate email format and password length (handled by Pydantic)
    2. Check if email already exists
    3. Hash password with bcrypt
    4. Insert user into database
    5. Generate JWT token
    6. Return token
    
    Args:
        user_data: UserCreate model with email and password
    
    Returns:
        Token with access_token and token_type
        
    Raises:
        HTTPException 400: If email already exists
        HTTPException 422: If validation fails (handled by FastAPI)
        HTTPException 500: If database operation fails
        
    Example:
        POST /api/v1/auth/signup
        {
            "email": "user@example.com",
            "password": "SecurePass123!"
        }
        
        Response:
        {
            "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
            "token_type": "bearer"
        }
    """
    try:
        # Check if user already exists
        existing_user = get_user_by_email(user_data.email)
        if existing_user:
            logger.warning(f"Signup attempt with existing email: {user_data.email}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email is already registered"
            )
        
        # Hash the password
        password_hash = get_password_hash(user_data.password)
        logger.debug(f"Password hashed for user: {user_data.email}")
        
        # Create user in database
        user_id = create_user(email=user_data.email, password_hash=password_hash)
        logger.info(f"New user created: {user_data.email} (ID: {user_id})")
        
        # Generate JWT token
        access_token = generate_token(user_id=user_id)
        logger.debug(f"JWT token generated for user: {user_id}")
        
        # Return token
        return Token(access_token=access_token, token_type="bearer")
        
    except HTTPException:
        # Re-raise HTTP exceptions (like duplicate email)
        raise
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Signup failed for {user_data.email}: {str(e)}")
        
        # Check if it's a duplicate email error from database
        error_msg = str(e).lower()
        if "duplicate" in error_msg or "already registered" in error_msg:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email is already registered"
            )
        
        # Generic error for other failures
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create user account"
        )


@router.post("/login", response_model=Token)
async def login(credentials: UserLogin) -> Token:
    """
    Authenticate user and return access token.
    
    Process:
    1. Fetch user by email
    2. Verify password against stored hash
    3. Generate JWT token
    4. Return token
    
    Args:
        credentials: UserLogin model with email and password
    
    Returns:
        Token with access_token and token_type
        
    Raises:
        HTTPException 401: If credentials are invalid (generic message for security)
        HTTPException 500: If database operation fails
        
    Example:
        POST /api/v1/auth/login
        {
            "email": "user@example.com",
            "password": "SecurePass123!"
        }
        
        Response:
        {
            "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
            "token_type": "bearer"
        }
    """
    try:
        # Fetch user by email
        user = get_user_by_email(credentials.email)
        
        # Check if user exists
        if not user:
            logger.warning(f"Login attempt with non-existent email: {credentials.email}")
            # Use generic error message to not reveal if email exists
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Verify password
        is_valid = verify_password(credentials.password, user["password_hash"])
        
        if not is_valid:
            logger.warning(f"Login attempt with incorrect password for: {credentials.email}")
            # Use generic error message to not reveal if email exists
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid email or password"
            )
        
        # Generate JWT token
        access_token = generate_token(user_id=user["id"])
        logger.info(f"User logged in successfully: {credentials.email} (ID: {user['id']})")
        
        # Return token
        return Token(access_token=access_token, token_type="bearer")
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Login failed for {credentials.email}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Authentication failed"
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(current_user_id: str = Depends(get_current_user)) -> UserResponse:
    """
    Get authenticated user's profile information.
    
    Process:
    1. Extract user_id from validated JWT token (via dependency)
    2. Fetch user details from database
    3. Return user information (excluding password)
    
    Args:
        current_user_id: User ID extracted from JWT token (injected by dependency)
    
    Returns:
        UserResponse with user_id, email, and created_at
        
    Raises:
        HTTPException 401: If token is invalid (handled by dependency)
        HTTPException 404: If user not found in database
        HTTPException 500: If database operation fails
        
    Example:
        GET /api/v1/auth/me
        Headers: Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
        
        Response:
        {
            "user_id": "550e8400-e29b-41d4-a716-446655440000",
            "email": "user@example.com",
            "created_at": "2025-12-17T10:30:00Z"
        }
    """
    try:
        # Fetch user details from database
        user = get_user_by_id(current_user_id)
        
        # Check if user exists
        if not user:
            logger.error(f"User not found in database: {current_user_id}")
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        logger.debug(f"Profile retrieved for user: {current_user_id}")
        
        # Return user information (password_hash is excluded by get_user_by_id)
        return UserResponse(
            user_id=user["id"],
            email=user["email"],
            created_at=user["created_at"]
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Failed to retrieve profile for user {current_user_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user profile"
        )
