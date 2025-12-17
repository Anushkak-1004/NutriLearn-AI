"""
Authentication utilities for JWT token generation and password hashing.

This module provides core authentication functions including:
- JWT token generation and verification
- Password hashing and verification using bcrypt
- Configuration loading from environment variables
"""

import os
from datetime import datetime, timedelta
from typing import Dict, Any, Optional

from jose import JWTError, jwt
import bcrypt
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# JWT Configuration
SECRET_KEY = os.getenv("JWT_SECRET_KEY")
ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_DAYS = int(os.getenv("ACCESS_TOKEN_EXPIRE_DAYS", "7"))

# Validate that SECRET_KEY is set
if not SECRET_KEY:
    raise ValueError(
        "JWT_SECRET_KEY environment variable is not set. "
        "Generate one with: openssl rand -hex 32"
    )


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
        
    Example:
        >>> token = generate_token("user-123")
        >>> print(token)
        'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...'
    """
    # Use custom expiration if provided, otherwise use default
    if expires_delta is None:
        expires_delta = timedelta(days=ACCESS_TOKEN_EXPIRE_DAYS)
    
    # Calculate expiration time using the provided or default expires_delta
    expire = datetime.utcnow() + expires_delta
    
    # Create token payload
    to_encode = {
        "sub": user_id,  # Subject (user ID)
        "exp": expire,    # Expiration time
        "iat": datetime.utcnow()  # Issued at time
    }
    
    # Encode and return JWT token
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Dict[str, Any]:
    """
    Verify and decode a JWT token.
    
    Args:
        token: JWT token string to verify
        
    Returns:
        Decoded token payload containing user_id and expiration
        
    Raises:
        JWTError: If token is invalid, expired, or malformed
        
    Example:
        >>> payload = verify_token(token)
        >>> user_id = payload["sub"]
    """
    try:
        # Decode and verify token
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError as e:
        # Re-raise JWT errors for proper handling by caller
        raise e


def get_password_hash(password: str) -> str:
    """
    Hash a password using bcrypt.
    
    Args:
        password: Plain text password
        
    Returns:
        Bcrypt hashed password string
        
    Example:
        >>> hashed = get_password_hash("SecurePass123!")
        >>> print(hashed)
        '$2b$12$KIXxJ...'
    """
    # Convert password to bytes and hash with bcrypt
    # Bcrypt has a 72-byte limit, so truncate if necessary
    password_bytes = password.encode('utf-8')[:72]
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    # Return as string for storage
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verify a password against its hash.
    
    Args:
        plain_password: Plain text password to verify
        hashed_password: Stored bcrypt hash
        
    Returns:
        True if password matches, False otherwise
        
    Example:
        >>> is_valid = verify_password("SecurePass123!", hashed)
        >>> print(is_valid)
        True
    """
    # Convert both to bytes for bcrypt verification
    # Bcrypt has a 72-byte limit, so truncate if necessary
    password_bytes = plain_password.encode('utf-8')[:72]
    hashed_bytes = hashed_password.encode('utf-8')
    return bcrypt.checkpw(password_bytes, hashed_bytes)
