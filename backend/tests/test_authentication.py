"""
Comprehensive Authentication Test Suite for NutriLearn AI

This module contains unit tests, property-based tests, and integration tests
for the JWT authentication system.

Test Categories:
1. Unit Tests - Test individual authentication functions
2. Property-Based Tests - Test universal properties using Hypothesis
3. Integration Tests - Test complete authentication flows

Run tests:
    pytest backend/tests/test_authentication.py -v
    pytest backend/tests/test_authentication.py -v --hypothesis-show-statistics
"""

import pytest
from hypothesis import given, strategies as st, settings
from hypothesis.strategies import emails, text
from datetime import datetime, timedelta
from jose import jwt, JWTError
import os

# Import authentication functions
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from app.auth import (
    generate_token, verify_token, get_password_hash, verify_password,
    SECRET_KEY, ALGORITHM
)
from app.models import UserCreate, UserLogin, Token, UserResponse


# ============================================================================
# UNIT TESTS - Test Individual Functions
# ============================================================================

class TestPasswordHashing:
    """Unit tests for password hashing and verification."""
    
    def test_password_hash_is_not_plaintext(self):
        """Test that hashed password is not the same as plaintext."""
        password = "SecurePass123!"
        hashed = get_password_hash(password)
        
        assert hashed != password
        assert len(hashed) > len(password)
        assert hashed.startswith("$2b$")  # bcrypt prefix
    
    def test_same_password_different_hashes(self):
        """Test that same password produces different hashes (salt)."""
        password = "SecurePass123!"
        hash1 = get_password_hash(password)
        hash2 = get_password_hash(password)
        
        assert hash1 != hash2  # Different due to random salt
    
    def test_verify_correct_password(self):
        """Test that correct password verifies successfully."""
        password = "SecurePass123!"
        hashed = get_password_hash(password)
        
        assert verify_password(password, hashed) is True
    
    def test_verify_incorrect_password(self):
        """Test that incorrect password fails verification."""
        password = "SecurePass123!"
        wrong_password = "WrongPass456!"
        hashed = get_password_hash(password)
        
        assert verify_password(wrong_password, hashed) is False
    
    def test_verify_empty_password(self):
        """Test that empty password fails verification."""
        password = "SecurePass123!"
        hashed = get_password_hash(password)
        
        assert verify_password("", hashed) is False


class TestTokenGeneration:
    """Unit tests for JWT token generation."""
    
    def test_generate_token_creates_valid_jwt(self):
        """Test that generated token is a valid JWT."""
        user_id = "test_user_123"
        token = generate_token(user_id)
        
        assert isinstance(token, str)
        assert len(token) > 0
        assert token.count('.') == 2  # JWT has 3 parts separated by dots
    
    def test_token_contains_user_id(self):
        """Test that token payload contains user_id."""
        user_id = "test_user_123"
        token = generate_token(user_id)
        
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert payload["sub"] == user_id
    
    def test_token_has_expiration(self):
        """Test that token has expiration time."""
        user_id = "test_user_123"
        token = generate_token(user_id)
        
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        assert "exp" in payload
        assert isinstance(payload["exp"], int)
    
    def test_token_expiration_is_7_days(self):
        """Test that token expires in approximately 7 days."""
        user_id = "test_user_123"
        token = generate_token(user_id)
        
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        exp_time = datetime.fromtimestamp(payload["exp"])
        now = datetime.utcnow()
        
        # Should be approximately 7 days (allow 1 minute tolerance)
        delta = exp_time - now
        assert 6.99 <= delta.days <= 7.01
    
    def test_custom_expiration(self):
        """Test that custom expiration time is respected."""
        user_id = "test_user_123"
        custom_delta = timedelta(hours=1)
        token = generate_token(user_id, expires_delta=custom_delta)
        
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        exp_time = datetime.fromtimestamp(payload["exp"])
        now = datetime.utcnow()
        
        delta = exp_time - now
        # Should be approximately 1 hour (allow 5 minute tolerance)
        hours = delta.total_seconds() / 3600
        assert 0.9 <= hours <= 1.1


class TestTokenVerification:
    """Unit tests for JWT token verification."""
    
    def test_verify_valid_token(self):
        """Test that valid token is verified successfully."""
        user_id = "test_user_123"
        token = generate_token(user_id)
        
        payload = verify_token(token)
        assert payload["sub"] == user_id
    
    def test_verify_expired_token(self):
        """Test that expired token raises error."""
        user_id = "test_user_123"
        # Create token that expires immediately
        token = generate_token(user_id, expires_delta=timedelta(seconds=-1))
        
        with pytest.raises(JWTError):
            verify_token(token)
    
    def test_verify_invalid_signature(self):
        """Test that token with invalid signature raises error."""
        user_id = "test_user_123"
        token = generate_token(user_id)
        
        # Tamper with token by changing last character
        tampered_token = token[:-1] + ('a' if token[-1] != 'a' else 'b')
        
        with pytest.raises(JWTError):
            verify_token(tampered_token)
    
    def test_verify_malformed_token(self):
        """Test that malformed token raises error."""
        malformed_tokens = [
            "not.a.token",
            "invalid",
            "",
            "a.b",  # Only 2 parts
            "a.b.c.d"  # Too many parts
        ]
        
        for token in malformed_tokens:
            with pytest.raises(JWTError):
                verify_token(token)
    
    def test_token_algorithm_is_hs256(self):
        """Test that token uses HS256 algorithm."""
        user_id = "test_user_123"
        token = generate_token(user_id)
        
        # Decode header without verification
        header = jwt.get_unverified_header(token)
        assert header["alg"] == "HS256"


class TestPydanticModels:
    """Unit tests for authentication Pydantic models."""
    
    def test_user_create_valid(self):
        """Test UserCreate model with valid data."""
        user = UserCreate(
            email="user@example.com",
            password="SecurePass123!"
        )
        assert user.email == "user@example.com"
        assert user.password == "SecurePass123!"
    
    def test_user_create_invalid_email(self):
        """Test UserCreate rejects invalid email."""
        with pytest.raises(ValueError):
            UserCreate(
                email="not-an-email",
                password="SecurePass123!"
            )
    
    def test_user_create_short_password(self):
        """Test UserCreate rejects password shorter than 8 characters."""
        with pytest.raises(ValueError):
            UserCreate(
                email="user@example.com",
                password="short"
            )
    
    def test_user_login_valid(self):
        """Test UserLogin model with valid data."""
        login = UserLogin(
            email="user@example.com",
            password="SecurePass123!"
        )
        assert login.email == "user@example.com"
        assert login.password == "SecurePass123!"
    
    def test_token_model(self):
        """Test Token model."""
        token = Token(
            access_token="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
            token_type="bearer"
        )
        assert token.access_token.startswith("eyJ")
        assert token.token_type == "bearer"
    
    def test_user_response_model(self):
        """Test UserResponse model."""
        user = UserResponse(
            user_id="550e8400-e29b-41d4-a716-446655440000",
            email="user@example.com",
            created_at=datetime.utcnow()
        )
        assert user.user_id == "550e8400-e29b-41d4-a716-446655440000"
        assert user.email == "user@example.com"


# ============================================================================
# PROPERTY-BASED TESTS - Test Universal Properties
# ============================================================================

# Hypothesis strategies for generating test data
valid_emails_strategy = emails()
valid_passwords_strategy = text(
    min_size=8,
    max_size=100,
    alphabet=st.characters(blacklist_categories=('Cs',))
)
invalid_emails_strategy = text(min_size=1, max_size=100).filter(
    lambda x: '@' not in x or '.' not in x.split('@')[-1]
)
short_passwords_strategy = text(min_size=0, max_size=7)
user_ids_strategy = text(min_size=1, max_size=100, alphabet=st.characters(
    whitelist_categories=('Lu', 'Ll', 'Nd'),
    blacklist_characters='\x00'
))


class TestPasswordHashingProperties:
    """Property-based tests for password hashing."""
    
    @given(password=valid_passwords_strategy)
    @settings(max_examples=100)
    def test_property_password_hash_never_equals_plaintext(self, password):
        """
        **Feature: jwt-authentication, Property 8: Password verification correctly validates credentials**
        **Validates: Requirements 5.2**
        
        Property: For any valid password, the hashed version should never equal the plaintext.
        """
        hashed = get_password_hash(password)
        assert hashed != password
    
    @given(password=valid_passwords_strategy)
    @settings(max_examples=100)
    def test_property_correct_password_always_verifies(self, password):
        """
        **Feature: jwt-authentication, Property 8: Password verification correctly validates credentials**
        **Validates: Requirements 5.2**
        
        Property: For any password, verifying the correct password against its hash should return True.
        """
        hashed = get_password_hash(password)
        assert verify_password(password, hashed) is True
    
    @given(password=valid_passwords_strategy, wrong_password=valid_passwords_strategy)
    @settings(max_examples=100)
    def test_property_wrong_password_fails_verification(self, password, wrong_password):
        """
        **Feature: jwt-authentication, Property 8: Password verification correctly validates credentials**
        **Validates: Requirements 5.2**
        
        Property: For any two different passwords, verifying the wrong password should return False.
        """
        if password == wrong_password:
            return  # Skip if passwords happen to be the same
        
        hashed = get_password_hash(password)
        assert verify_password(wrong_password, hashed) is False


class TestTokenGenerationProperties:
    """Property-based tests for token generation."""
    
    @given(user_id=user_ids_strategy)
    @settings(max_examples=100)
    def test_property_token_contains_user_id(self, user_id):
        """
        **Feature: jwt-authentication, Property 4: Generated tokens contain user_id in payload**
        **Validates: Requirements 2.5**
        
        Property: For any user_id, the generated token should contain that user_id in the payload.
        """
        token = generate_token(user_id)
        payload = verify_token(token)
        assert payload["sub"] == user_id
    
    @given(user_id=user_ids_strategy)
    @settings(max_examples=100)
    def test_property_token_is_valid_jwt(self, user_id):
        """
        **Feature: jwt-authentication, Property 5: Valid tokens grant access to protected endpoints**
        **Validates: Requirements 3.1**
        
        Property: For any user_id, the generated token should be a valid JWT that can be verified.
        """
        token = generate_token(user_id)
        payload = verify_token(token)
        assert "sub" in payload
        assert "exp" in payload
    
    @given(user_id=user_ids_strategy)
    @settings(max_examples=100)
    def test_property_token_uses_hs256(self, user_id):
        """
        **Feature: jwt-authentication, Property 9: JWT tokens use HS256 algorithm**
        **Validates: Requirements 5.4**
        
        Property: For any user_id, the generated token should use HS256 algorithm.
        """
        token = generate_token(user_id)
        header = jwt.get_unverified_header(token)
        assert header["alg"] == "HS256"


class TestTokenVerificationProperties:
    """Property-based tests for token verification."""
    
    @given(user_id=user_ids_strategy)
    @settings(max_examples=100)
    def test_property_valid_token_grants_access(self, user_id):
        """
        **Feature: jwt-authentication, Property 5: Valid tokens grant access to protected endpoints**
        **Validates: Requirements 3.1**
        
        Property: For any valid token, verification should succeed and return the user_id.
        """
        token = generate_token(user_id)
        payload = verify_token(token)
        assert payload["sub"] == user_id
    
    @given(user_id=user_ids_strategy, random_string=text(min_size=10, max_size=50))
    @settings(max_examples=50)
    def test_property_malformed_tokens_rejected(self, user_id, random_string):
        """
        **Feature: jwt-authentication, Property 6: Invalid or malformed tokens are rejected**
        **Validates: Requirements 3.3, 3.5**
        
        Property: For any malformed token, verification should raise an error.
        """
        # Generate various malformed tokens
        malformed_tokens = [
            random_string,
            f"{random_string}.{random_string}",
            f"{random_string}.{random_string}.{random_string}.{random_string}"
        ]
        
        for token in malformed_tokens:
            with pytest.raises(JWTError):
                verify_token(token)


class TestEmailValidationProperties:
    """Property-based tests for email validation."""
    
    @given(email=valid_emails_strategy, password=valid_passwords_strategy)
    @settings(max_examples=100)
    def test_property_valid_email_accepted(self, email, password):
        """
        **Feature: jwt-authentication, Property 2: Invalid email formats are rejected during signup**
        **Validates: Requirements 1.4**
        
        Property: For any valid email format, UserCreate should accept it.
        """
        user = UserCreate(email=email, password=password)
        assert user.email == email
    
    @given(invalid_email=invalid_emails_strategy, password=valid_passwords_strategy)
    @settings(max_examples=100)
    def test_property_invalid_email_rejected(self, invalid_email, password):
        """
        **Feature: jwt-authentication, Property 2: Invalid email formats are rejected during signup**
        **Validates: Requirements 1.4**
        
        Property: For any invalid email format, UserCreate should reject it.
        """
        with pytest.raises(ValueError):
            UserCreate(email=invalid_email, password=password)


# ============================================================================
# INTEGRATION TESTS - Test Complete Flows
# ============================================================================

class TestAuthenticationFlows:
    """Integration tests for complete authentication flows."""
    
    def test_signup_login_flow(self):
        """Test complete signup and login flow."""
        # Simulate signup
        email = "newuser@example.com"
        password = "SecurePass123!"
        
        # Hash password (as done in signup)
        password_hash = get_password_hash(password)
        
        # Simulate user creation (would be stored in DB)
        user_id = "new_user_123"
        
        # Generate token for new user
        token = generate_token(user_id)
        
        # Verify token works
        payload = verify_token(token)
        assert payload["sub"] == user_id
        
        # Simulate login - verify password
        assert verify_password(password, password_hash) is True
        
        # Generate new token on login
        login_token = generate_token(user_id)
        login_payload = verify_token(login_token)
        assert login_payload["sub"] == user_id
    
    def test_token_expiration_flow(self):
        """Test that expired tokens are properly rejected."""
        user_id = "test_user_123"
        
        # Create token that expires in 1 second
        token = generate_token(user_id, expires_delta=timedelta(seconds=1))
        
        # Token should work immediately
        payload = verify_token(token)
        assert payload["sub"] == user_id
        
        # Wait for expiration
        import time
        time.sleep(2)
        
        # Token should now be expired
        with pytest.raises(JWTError):
            verify_token(token)
    
    def test_password_change_flow(self):
        """Test password change invalidates old password."""
        user_id = "test_user_123"
        old_password = "OldPass123!"
        new_password = "NewPass456!"
        
        # Hash old password
        old_hash = get_password_hash(old_password)
        
        # Verify old password works
        assert verify_password(old_password, old_hash) is True
        
        # Change password (hash new password)
        new_hash = get_password_hash(new_password)
        
        # Old password should not work with new hash
        assert verify_password(old_password, new_hash) is False
        
        # New password should work with new hash
        assert verify_password(new_password, new_hash) is True


# ============================================================================
# TEST CONFIGURATION
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--hypothesis-show-statistics"])
