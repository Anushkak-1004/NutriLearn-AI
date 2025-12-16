"""
Authentication dependencies for FastAPI route protection.

This module provides dependency injection functions for:
- Extracting and validating JWT tokens from requests
- Protecting routes with authentication requirements
- Optional authentication for flexible endpoints
"""

from typing import Optional
import logging

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError

from app.auth import verify_token

# Configure logging
logger = logging.getLogger(__name__)

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
        user_id: Authenticated user's ID from token payload
        
    Raises:
        HTTPException 401: If token is missing, invalid, expired, or malformed
        
    Usage:
        @router.get("/protected")
        async def protected_route(user_id: str = Depends(get_current_user)):
            # user_id is automatically extracted and validated
            return {"user_id": user_id}
    """
    # Extract token from credentials
    token = credentials.credentials
    
    try:
        # Verify and decode token
        payload = verify_token(token)
        
        # Extract user_id from payload
        user_id: str = payload.get("sub")
        
        if user_id is None:
            logger.warning("Token payload missing 'sub' field")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        logger.debug(f"Successfully authenticated user: {user_id}")
        return user_id
        
    except JWTError as e:
        # Handle all JWT-related errors (expired, invalid signature, malformed)
        logger.warning(f"JWT validation failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        # Handle unexpected errors
        logger.error(f"Unexpected error during authentication: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(
        HTTPBearer(auto_error=False)
    )
) -> Optional[str]:
    """
    Optional authentication dependency for endpoints that work with or without auth.
    
    This dependency allows endpoints to provide different functionality based on
    whether the user is authenticated or not, without requiring authentication.
    
    Args:
        credentials: Optional HTTP Bearer credentials from request header
        
    Returns:
        user_id: Authenticated user's ID if token is valid, None if no token provided
        
    Raises:
        HTTPException 401: If token is provided but invalid
        
    Usage:
        @router.get("/flexible")
        async def flexible_route(user_id: Optional[str] = Depends(get_current_user_optional)):
            if user_id:
                return {"message": "Authenticated", "user_id": user_id}
            else:
                return {"message": "Anonymous access"}
    """
    # If no credentials provided, return None (anonymous access)
    if credentials is None:
        logger.debug("No authentication credentials provided (optional auth)")
        return None
    
    # If credentials provided, validate them
    try:
        token = credentials.credentials
        payload = verify_token(token)
        user_id: str = payload.get("sub")
        
        if user_id is None:
            logger.warning("Token payload missing 'sub' field (optional auth)")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        logger.debug(f"Successfully authenticated user (optional): {user_id}")
        return user_id
        
    except JWTError as e:
        # If token is provided but invalid, still raise error
        logger.warning(f"JWT validation failed (optional auth): {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"Unexpected error during optional authentication: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )
