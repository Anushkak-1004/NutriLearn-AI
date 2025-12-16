"""
Quick test to verify user management database functions.
"""

import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from app.database import create_user, get_user_by_email, get_user_by_id, clear_all_data
from app.auth import get_password_hash

def test_user_functions():
    """Test the user management functions."""
    print("=" * 60)
    print("Testing User Management Functions")
    print("=" * 60)
    
    # Clear any existing data
    clear_all_data()
    print("\n✓ Cleared existing data")
    
    # Test 1: Create a new user
    print("\n[Test 1] Creating a new user...")
    email = "test@example.com"
    password = "SecurePass123!"
    password_hash = get_password_hash(password)
    
    try:
        user_id = create_user(email, password_hash)
        print(f"✓ User created successfully with ID: {user_id}")
    except Exception as e:
        print(f"✗ Failed to create user: {e}")
        return False
    
    # Test 2: Get user by email
    print("\n[Test 2] Fetching user by email...")
    user = get_user_by_email(email)
    if user:
        print(f"✓ User found: {user['email']}")
        print(f"  - ID: {user['id']}")
        print(f"  - Has password_hash: {bool(user.get('password_hash'))}")
        print(f"  - Created at: {user.get('created_at')}")
    else:
        print("✗ User not found by email")
        return False
    
    # Test 3: Get user by ID
    print("\n[Test 3] Fetching user by ID...")
    user_by_id = get_user_by_id(user_id)
    if user_by_id:
        print(f"✓ User found: {user_by_id['email']}")
        print(f"  - ID: {user_by_id['id']}")
        print(f"  - Password hash excluded: {'password_hash' not in user_by_id}")
        print(f"  - Created at: {user_by_id.get('created_at')}")
    else:
        print("✗ User not found by ID")
        return False
    
    # Test 4: Verify password_hash is excluded from get_user_by_id
    print("\n[Test 4] Verifying password_hash exclusion...")
    if 'password_hash' not in user_by_id:
        print("✓ Password hash correctly excluded from get_user_by_id()")
    else:
        print("✗ Password hash should not be in get_user_by_id() response")
        return False
    
    # Test 5: Try to create duplicate user
    print("\n[Test 5] Testing duplicate email rejection...")
    try:
        duplicate_id = create_user(email, password_hash)
        print("✗ Duplicate email should have been rejected")
        return False
    except Exception as e:
        if "already registered" in str(e).lower():
            print(f"✓ Duplicate email correctly rejected: {e}")
        else:
            print(f"✗ Unexpected error: {e}")
            return False
    
    # Test 6: Get non-existent user by email
    print("\n[Test 6] Testing non-existent user by email...")
    non_existent = get_user_by_email("nonexistent@example.com")
    if non_existent is None:
        print("✓ Non-existent user correctly returns None")
    else:
        print("✗ Non-existent user should return None")
        return False
    
    # Test 7: Get non-existent user by ID
    print("\n[Test 7] Testing non-existent user by ID...")
    non_existent_id = get_user_by_id("00000000-0000-0000-0000-000000000000")
    if non_existent_id is None:
        print("✓ Non-existent user ID correctly returns None")
    else:
        print("✗ Non-existent user ID should return None")
        return False
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = test_user_functions()
    sys.exit(0 if success else 1)
