"""
Validation script for authentication routes module.
Checks that the module is correctly structured without running the server.
"""

import sys
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

def validate_auth_routes():
    """Validate the auth routes module structure."""
    print("=" * 60)
    print("Validating Authentication Routes Module")
    print("=" * 60)
    
    # Test 1: Import the module
    print("\n[Test 1] Importing auth_routes module...")
    try:
        from app.api import auth_routes
        print("  ✓ Module imported successfully")
    except Exception as e:
        print(f"  ✗ Failed to import module: {e}")
        return False
    
    # Test 2: Check router exists
    print("\n[Test 2] Checking router exists...")
    if hasattr(auth_routes, 'router'):
        print("  ✓ Router found")
    else:
        print("  ✗ Router not found")
        return False
    
    # Test 3: Check router configuration
    print("\n[Test 3] Checking router configuration...")
    router = auth_routes.router
    print(f"  - Prefix: {router.prefix}")
    print(f"  - Tags: {router.tags}")
    
    if router.prefix == "/api/v1/auth":
        print("  ✓ Correct prefix")
    else:
        print(f"  ✗ Wrong prefix: {router.prefix}")
        return False
    
    # Test 4: Check endpoints exist
    print("\n[Test 4] Checking endpoints...")
    routes = router.routes
    print(f"  - Total routes: {len(routes)}")
    
    route_info = []
    for route in routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            methods = list(route.methods)
            path = route.path
            name = route.name if hasattr(route, 'name') else 'unknown'
            route_info.append((methods, path, name))
            print(f"    • {methods[0]} {path} ({name})")
    
    # Check for required endpoints
    required_endpoints = [
        ('POST', '/api/v1/auth/signup', 'signup'),
        ('POST', '/api/v1/auth/login', 'login'),
        ('GET', '/api/v1/auth/me', 'get_current_user_info')
    ]
    
    for method, path, name in required_endpoints:
        found = any(
            method in methods and path == route_path
            for methods, route_path, route_name in route_info
        )
        if found:
            print(f"  ✓ {method} {path} endpoint found")
        else:
            print(f"  ✗ {method} {path} endpoint missing")
            return False
    
    # Test 5: Check endpoint functions exist
    print("\n[Test 5] Checking endpoint functions...")
    functions = ['signup', 'login', 'get_current_user_info']
    for func_name in functions:
        if hasattr(auth_routes, func_name):
            func = getattr(auth_routes, func_name)
            print(f"  ✓ {func_name}() function exists")
            
            # Check if it's async
            import inspect
            if inspect.iscoroutinefunction(func):
                print(f"    - Is async: Yes")
            
            # Check docstring
            if func.__doc__:
                first_line = func.__doc__.strip().split('\n')[0]
                print(f"    - Docstring: {first_line}")
        else:
            print(f"  ✗ {func_name}() function not found")
            return False
    
    # Test 6: Check imports
    print("\n[Test 6] Checking required imports...")
    required_imports = [
        'UserCreate', 'UserLogin', 'Token', 'UserResponse',
        'generate_token', 'get_password_hash', 'verify_password',
        'create_user', 'get_user_by_email', 'get_user_by_id',
        'get_current_user'
    ]
    
    for import_name in required_imports:
        if hasattr(auth_routes, import_name):
            print(f"  ✓ {import_name} imported")
        else:
            print(f"  ✗ {import_name} not imported")
            return False
    
    # Test 7: Check response models
    print("\n[Test 7] Checking response models...")
    for route in routes:
        if hasattr(route, 'response_model') and route.response_model:
            print(f"  ✓ {route.name} has response model: {route.response_model.__name__}")
    
    # Test 8: Check status codes
    print("\n[Test 8] Checking status codes...")
    for route in routes:
        if hasattr(route, 'status_code'):
            print(f"  - {route.name}: {route.status_code}")
    
    print("\n" + "=" * 60)
    print("✅ All validation checks passed!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Register the router in backend/app/main.py (Task 8)")
    print("2. Run the test suite: python backend/test_auth_routes.py")
    print("3. Start the server and test with: python backend/test_api.py")
    return True

if __name__ == "__main__":
    success = validate_auth_routes()
    sys.exit(0 if success else 1)
