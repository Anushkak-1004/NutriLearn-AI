# Authentication Testing Guide

## Overview

This guide explains how to run and understand the authentication test suite for NutriLearn AI.

## Test Suite Structure

The authentication test suite (`test_authentication.py`) contains three types of tests:

### 1. Unit Tests
Test individual authentication functions in isolation:
- Password hashing and verification
- JWT token generation
- JWT token verification
- Pydantic model validation

### 2. Property-Based Tests
Test universal properties that should hold across all inputs using Hypothesis:
- Password hashing properties
- Token generation properties
- Token verification properties
- Email validation properties

### 3. Integration Tests
Test complete authentication flows:
- Signup → Login flow
- Token expiration handling
- Password change flow

## Running Tests

### Run All Authentication Tests

```bash
cd backend
python -m pytest tests/test_authentication.py -v
```

### Run Specific Test Classes

```bash
# Run only password hashing tests
python -m pytest tests/test_authentication.py::TestPasswordHashing -v

# Run only token generation tests
python -m pytest tests/test_authentication.py::TestTokenGeneration -v

# Run only property-based tests
python -m pytest tests/test_authentication.py::TestPasswordHashingProperties -v
```

### Run with Hypothesis Statistics

To see detailed statistics about property-based tests:

```bash
python -m pytest tests/test_authentication.py -v --hypothesis-show-statistics
```

### Run with Coverage

```bash
python -m pytest tests/test_authentication.py --cov=app.auth --cov-report=html
```

## Test Categories

### Password Hashing Tests

**Unit Tests:**
- `test_password_hash_is_not_plaintext` - Verifies hashed password differs from plaintext
- `test_same_password_different_hashes` - Verifies salt randomization
- `test_verify_correct_password` - Verifies correct password validation
- `test_verify_incorrect_password` - Verifies incorrect password rejection
- `test_verify_empty_password` - Verifies empty password rejection

**Property Tests:**
- `test_property_password_hash_never_equals_plaintext` - Tests across 100 random passwords
- `test_property_correct_password_always_verifies` - Tests verification across 100 random passwords
- `test_property_wrong_password_fails_verification` - Tests rejection across 100 password pairs

### Token Generation Tests

**Unit Tests:**
- `test_generate_token_creates_valid_jwt` - Verifies JWT structure
- `test_token_contains_user_id` - Verifies user_id in payload
- `test_token_has_expiration` - Verifies expiration field exists
- `test_token_expiration_is_7_days` - Verifies default 7-day expiration
- `test_custom_expiration` - Verifies custom expiration times

**Property Tests:**
- `test_property_token_contains_user_id` - Tests across 100 random user IDs
- `test_property_token_is_valid_jwt` - Tests JWT validity across 100 tokens
- `test_property_token_uses_hs256` - Verifies HS256 algorithm across 100 tokens

### Token Verification Tests

**Unit Tests:**
- `test_verify_valid_token` - Verifies valid token acceptance
- `test_verify_expired_token` - Verifies expired token rejection
- `test_verify_invalid_signature` - Verifies tampered token rejection
- `test_verify_malformed_token` - Verifies malformed token rejection
- `test_token_algorithm_is_hs256` - Verifies algorithm in header

**Property Tests:**
- `test_property_valid_token_grants_access` - Tests across 100 valid tokens
- `test_property_malformed_tokens_rejected` - Tests across 50 malformed tokens

### Email Validation Tests

**Property Tests:**
- `test_property_valid_email_accepted` - Tests across 100 valid emails
- `test_property_invalid_email_rejected` - Tests across 100 invalid emails

### Integration Tests

- `test_signup_login_flow` - Complete user registration and login
- `test_token_expiration_flow` - Token lifecycle and expiration
- `test_password_change_flow` - Password update and validation

## Understanding Property-Based Tests

Property-based tests use Hypothesis to generate random test data and verify that certain properties hold true across all inputs.

### Example: Password Hashing Property

```python
@given(password=valid_passwords_strategy)
@settings(max_examples=100)
def test_property_correct_password_always_verifies(self, password):
    """For any password, verifying the correct password should return True."""
    hashed = get_password_hash(password)
    assert verify_password(password, hashed) is True
```

This test:
1. Generates 100 random valid passwords
2. Hashes each password
3. Verifies that each password validates against its hash
4. If any example fails, Hypothesis will try to find the simplest failing case

### Test Strategies

The test suite uses these Hypothesis strategies:

- `valid_emails_strategy` - Generates valid email addresses
- `valid_passwords_strategy` - Generates passwords 8-100 characters
- `invalid_emails_strategy` - Generates invalid email formats
- `short_passwords_strategy` - Generates passwords 0-7 characters
- `user_ids_strategy` - Generates valid user ID strings

## Correctness Properties Tested

Each property-based test validates a specific correctness property from the design document:

1. **Property 1**: Valid signup creates user with hashed password and returns token
2. **Property 2**: Invalid email formats are rejected during signup
3. **Property 4**: Generated tokens contain user_id in payload
4. **Property 5**: Valid tokens grant access to protected endpoints
5. **Property 6**: Invalid or malformed tokens are rejected
6. **Property 8**: Password verification correctly validates credentials
7. **Property 9**: JWT tokens use HS256 algorithm

## Expected Test Results

All tests should pass. Expected output:

```
tests/test_authentication.py::TestPasswordHashing::test_password_hash_is_not_plaintext PASSED
tests/test_authentication.py::TestPasswordHashing::test_same_password_different_hashes PASSED
tests/test_authentication.py::TestPasswordHashing::test_verify_correct_password PASSED
...
tests/test_authentication.py::TestAuthenticationFlows::test_password_change_flow PASSED

============================== 34 passed in 5.23s ==============================
```

## Troubleshooting

### Import Errors

If you see import errors, ensure you're running from the `backend` directory:

```bash
cd backend
python -m pytest tests/test_authentication.py -v
```

### Hypothesis Failures

If a property-based test fails, Hypothesis will show you the failing example:

```
Falsifying example: test_property_correct_password_always_verifies(
    self=<test_authentication.TestPasswordHashingProperties object at 0x...>,
    password='example_password'
)
```

This helps you identify edge cases that break your code.

### Slow Tests

Property-based tests run 100 examples by default. To run faster during development:

```python
@settings(max_examples=10)  # Reduce from 100 to 10
```

## Adding New Tests

### Adding a Unit Test

```python
def test_new_feature(self):
    """Test description."""
    # Arrange
    input_data = "test"
    
    # Act
    result = function_to_test(input_data)
    
    # Assert
    assert result == expected_output
```

### Adding a Property Test

```python
@given(input_data=st.text())
@settings(max_examples=100)
def test_property_new_feature(self, input_data):
    """
    **Feature: jwt-authentication, Property X: Description**
    **Validates: Requirements X.Y**
    
    Property: For any input, this property should hold.
    """
    result = function_to_test(input_data)
    assert some_property_holds(result)
```

## Continuous Integration

These tests should be run in CI/CD pipelines:

```yaml
# .github/workflows/test.yml
- name: Run authentication tests
  run: |
    cd backend
    python -m pytest tests/test_authentication.py -v --cov=app.auth
```

## Test Coverage Goals

Target coverage for authentication module:
- **Password functions**: 100%
- **Token functions**: 100%
- **Pydantic models**: 100%
- **Overall auth module**: 95%+

Check coverage:

```bash
python -m pytest tests/test_authentication.py --cov=app.auth --cov-report=term-missing
```

## Best Practices

1. **Run tests before committing**: Ensure all tests pass
2. **Add tests for new features**: Every new auth feature needs tests
3. **Use property tests for general behavior**: Use unit tests for specific cases
4. **Keep tests fast**: Property tests should complete in < 10 seconds
5. **Document test intent**: Clear docstrings explain what's being tested

## Security Testing Notes

These tests verify:
- ✅ Passwords are never stored in plaintext
- ✅ Password hashes use bcrypt with salt
- ✅ Tokens expire after 7 days
- ✅ Tokens use secure HS256 algorithm
- ✅ Invalid tokens are rejected
- ✅ Malformed tokens are rejected
- ✅ Email validation prevents injection

## Further Reading

- [Hypothesis Documentation](https://hypothesis.readthedocs.io/)
- [pytest Documentation](https://docs.pytest.org/)
- [JWT Best Practices](https://tools.ietf.org/html/rfc8725)
- [OWASP Authentication Cheat Sheet](https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html)

## Support

For issues with tests:
1. Check this guide first
2. Review test output carefully
3. Check the design document for property definitions
4. Consult the team if tests consistently fail
