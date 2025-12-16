"""
Quick validation test for authentication dependencies.
"""

import pytest
from fastapi import HTTPException
from fastapi.security import HTTPAuthorizationCredentials
from datetime import timedelta

from app.auth import generate_token
from app.dependencies import get_current_user, get_current_user_optional


@pytest.mark.asyncio
async def test_get_current_user_with_valid_token():
    """Test that get_current_user extracts user_id from valid token."""
    # Generate a valid token
    user_id = "test-user-123"
    token = generate_token(user_id)
    
    # Create credentials object
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    
    # Call dependency
    result = await get_current_user(credentials)
    
    # Verify user_id is extracted correctly
    assert result == user_id


@pytest.mark.asyncio
async def test_get_current_user_with_invalid_token():
    """Test that get_current_user rejects invalid tokens."""
    # Create invalid token
    invalid_token = "invalid.token.here"
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=invalid_token)
    
    # Should raise HTTPException with 401 status
    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(credentials)
    
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_get_current_user_with_expired_token():
    """Test that get_current_user rejects expired tokens."""
    # Generate an expired token (negative expiration)
    user_id = "test-user-456"
    token = generate_token(user_id, expires_delta=timedelta(seconds=-1))
    
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    
    # Should raise HTTPException with 401 status
    with pytest.raises(HTTPException) as exc_info:
        await get_current_user(credentials)
    
    assert exc_info.value.status_code == 401


@pytest.mark.asyncio
async def test_get_current_user_optional_with_valid_token():
    """Test that get_current_user_optional returns user_id with valid token."""
    user_id = "test-user-789"
    token = generate_token(user_id)
    
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=token)
    
    result = await get_current_user_optional(credentials)
    
    assert result == user_id


@pytest.mark.asyncio
async def test_get_current_user_optional_without_token():
    """Test that get_current_user_optional returns None when no token provided."""
    # No credentials provided
    result = await get_current_user_optional(None)
    
    assert result is None


@pytest.mark.asyncio
async def test_get_current_user_optional_with_invalid_token():
    """Test that get_current_user_optional rejects invalid tokens."""
    invalid_token = "invalid.token.here"
    credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=invalid_token)
    
    # Should raise HTTPException even for optional auth if token is provided but invalid
    with pytest.raises(HTTPException) as exc_info:
        await get_current_user_optional(credentials)
    
    assert exc_info.value.status_code == 401


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
