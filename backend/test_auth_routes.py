"""
Quick test script to verify authentication routes.
Tests signup, login, and profile endpoints.
"""

import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from app.main import app
from app.database import clear_all_data

# Create test client
client = TestClient(app)

def test_auth_routes():
    """Test authentication routes."""
    print("=" * 60)
    print("Testing Authentication Routes")
    print("=" * 60)
    
    # Clear any existing data
    clear_all_data()
    print("\n✓ Cleared existing data")
    
    # Test 1: Signup with valid data
    print("\n[Test 1] Testing signup with valid data...")
    signup_data = {
        "email": "testuser@example.com",
        "password": "SecurePass123!"
    }
    
    response = client.post("/api/v1/auth/signup", json=signup_data)
    print(f"  Status: {response.status_code}")
    
    if response.status_code == 201:
        data = response.json()
        print(f"  ✓ Signup successful")
        print(f"    - Token type: {data['token_type']}")
        print(f"    - Token length: {len(data['access_token'])} chars")
        signup_token = data['access_token']
    else:
        print(f"  ✗ Signup failed: {response.json()}")
        return False
    
    # Test 2: Signup with duplicate email
    print("\n[Test 2] Testing signup with duplicate email...")
    response = client.post("/api/v1/auth/signup", json=signup_data)
    print(f"  Status: {response.status_code}")
    
    if response.status_code == 400:
        print(f"  ✓ Duplicate email correctly rejected")
        print(f"    - Error: {response.json()['detail']}")
    else:
        print(f"  ✗ Should have rejected duplicate email")
        return False
    
    # Test 3: Signup with invalid email
    print("\n[Test 3] Testing signup with invalid email...")
    invalid_email_data = {
        "email": "not-an-email",
        "password": "SecurePass123!"
    }
    
    response = client.post("/api/v1/auth/signup", json=invalid_email_data)
    print(f"  Status: {response.status_code}")
    
    if response.status_code == 422:
        print(f"  ✓ Invalid email correctly rejected")
    else:
        print(f"  ✗ Should have rejected invalid email")
        return False
    
    # Test 4: Signup with short password
    print("\n[Test 4] Testing signup with short password...")
    short_password_data = {
        "email": "another@example.com",
        "password": "short"
    }
    
    response = client.post("/api/v1/auth/signup", json=short_password_data)
    print(f"  Status: {response.status_code}")
    
    if response.status_code == 422:
        print(f"  ✓ Short password correctly rejected")
    else:
        print(f"  ✗ Should have rejected short password")
        return False
    
    # Test 5: Login with correct credentials
    print("\n[Test 5] Testing login with correct credentials...")
    login_data = {
        "email": "testuser@example.com",
        "password": "SecurePass123!"
    }
    
    response = client.post("/api/v1/auth/login", json=login_data)
    print(f"  Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"  ✓ Login successful")
        print(f"    - Token type: {data['token_type']}")
        print(f"    - Token length: {len(data['access_token'])} chars")
        login_token = data['access_token']
    else:
        print(f"  ✗ Login failed: {response.json()}")
        return False
    
    # Test 6: Login with wrong password
    print("\n[Test 6] Testing login with wrong password...")
    wrong_password_data = {
        "email": "testuser@example.com",
        "password": "WrongPassword123!"
    }
    
    response = client.post("/api/v1/auth/login", json=wrong_password_data)
    print(f"  Status: {response.status_code}")
    
    if response.status_code == 401:
        print(f"  ✓ Wrong password correctly rejected")
        print(f"    - Error: {response.json()['detail']}")
    else:
        print(f"  ✗ Should have rejected wrong password")
        return False
    
    # Test 7: Login with non-existent email
    print("\n[Test 7] Testing login with non-existent email...")
    nonexistent_data = {
        "email": "nonexistent@example.com",
        "password": "SomePassword123!"
    }
    
    response = client.post("/api/v1/auth/login", json=nonexistent_data)
    print(f"  Status: {response.status_code}")
    
    if response.status_code == 401:
        print(f"  ✓ Non-existent email correctly rejected")
        print(f"    - Error: {response.json()['detail']}")
    else:
        print(f"  ✗ Should have rejected non-existent email")
        return False
    
    # Test 8: Get profile with valid token
    print("\n[Test 8] Testing profile retrieval with valid token...")
    headers = {"Authorization": f"Bearer {login_token}"}
    response = client.get("/api/v1/auth/me", headers=headers)
    print(f"  Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"  ✓ Profile retrieved successfully")
        print(f"    - Email: {data['email']}")
        print(f"    - User ID: {data['user_id']}")
        print(f"    - Created at: {data['created_at']}")
        print(f"    - Password hash excluded: {'password_hash' not in data}")
    else:
        print(f"  ✗ Profile retrieval failed: {response.json()}")
        return False
    
    # Test 9: Get profile without token
    print("\n[Test 9] Testing profile retrieval without token...")
    response = client.get("/api/v1/auth/me")
    print(f"  Status: {response.status_code}")
    
    if response.status_code == 403:  # HTTPBearer returns 403 when no credentials
        print(f"  ✓ Request without token correctly rejected")
    else:
        print(f"  ✗ Should have rejected request without token")
        return False
    
    # Test 10: Get profile with invalid token
    print("\n[Test 10] Testing profile retrieval with invalid token...")
    headers = {"Authorization": "Bearer invalid_token_here"}
    response = client.get("/api/v1/auth/me", headers=headers)
    print(f"  Status: {response.status_code}")
    
    if response.status_code == 401:
        print(f"  ✓ Invalid token correctly rejected")
        print(f"    - Error: {response.json()['detail']}")
    else:
        print(f"  ✗ Should have rejected invalid token")
        return False
    
    # Test 11: Verify token contains user_id
    print("\n[Test 11] Verifying token contains user_id...")
    from app.auth import verify_token
    try:
        payload = verify_token(login_token)
        if "sub" in payload:
            print(f"  ✓ Token contains user_id in 'sub' field")
            print(f"    - User ID: {payload['sub']}")
        else:
            print(f"  ✗ Token missing 'sub' field")
            return False
    except Exception as e:
        print(f"  ✗ Token verification failed: {e}")
        return False
    
    # Test 12: Verify password is hashed
    print("\n[Test 12] Verifying password is hashed...")
    from app.database import get_user_by_email
    user = get_user_by_email("testuser@example.com")
    if user and user['password_hash'] != "SecurePass123!":
        print(f"  ✓ Password is hashed (not stored as plain text)")
        print(f"    - Hash starts with: {user['password_hash'][:10]}...")
    else:
        print(f"  ✗ Password should be hashed")
        return False
    
    print("\n" + "=" * 60)
    print("✅ All authentication tests passed!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_auth_routes()
    sys.exit(0 if success else 1)
