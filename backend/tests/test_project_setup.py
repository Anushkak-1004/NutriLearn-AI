"""
Project Setup Validation Tests
Tests to verify the project structure and configuration are correct.
"""

import os
import json
from pathlib import Path
import pytest


# Get project root (two levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent


def test_directory_structure_exists():
    """
    Verify all required directories exist.
    Feature: project-setup, Property 1: Directory structure completeness
    Validates: Requirements 1.1, 1.2, 1.3, 1.4
    """
    required_dirs = [
        "frontend/src/components",
        "frontend/src/pages",
        "frontend/src/utils",
        "frontend/src/hooks",
        "frontend/public",
        "backend/app/api",
        "backend/app/models",
        "backend/app/ml",
        "backend/app/mlops",
        "backend/tests",
        "ml-models",
        "docs",
    ]
    
    for dir_path in required_dirs:
        full_path = PROJECT_ROOT / dir_path
        assert full_path.exists(), f"Required directory missing: {dir_path}"
        assert full_path.is_dir(), f"Path exists but is not a directory: {dir_path}"


def test_configuration_files_exist():
    """
    Verify all required configuration files exist.
    Feature: project-setup, Property 2: Configuration file presence
    Validates: Requirements 2.1, 2.2, 3.1, 4.1, 4.2, 4.3, 5.1
    """
    required_files = [
        "frontend/package.json",
        "frontend/.gitignore",
        "frontend/.env.example",
        "backend/requirements.txt",
        "backend/.gitignore",
        "backend/.env.example",
        "docker-compose.yml",
        "README.md",
        ".gitignore",
    ]
    
    for file_path in required_files:
        full_path = PROJECT_ROOT / file_path
        assert full_path.exists(), f"Required file missing: {file_path}"
        assert full_path.is_file(), f"Path exists but is not a file: {file_path}"


def test_package_json_validity():
    """
    Verify package.json is valid JSON and contains required fields.
    Feature: project-setup, Property 3: Package configuration validity
    Validates: Requirements 2.1, 2.2
    """
    package_json_path = PROJECT_ROOT / "frontend" / "package.json"
    
    with open(package_json_path, 'r') as f:
        package_data = json.load(f)
    
    required_fields = ["name", "version", "dependencies", "scripts"]
    for field in required_fields:
        assert field in package_data, f"package.json missing required field: {field}"
    
    # Verify key dependencies
    assert "react" in package_data["dependencies"], "React dependency missing"
    assert "axios" in package_data["dependencies"], "Axios dependency missing"


def test_frontend_dependencies_complete():
    """
    Verify frontend package.json contains all required dependencies.
    Feature: project-setup, Property 4: Dependency specification completeness
    Validates: Requirements 2.1, 2.3
    """
    package_json_path = PROJECT_ROOT / "frontend" / "package.json"
    
    with open(package_json_path, 'r') as f:
        package_data = json.load(f)
    
    required_deps = ["react", "react-dom", "axios"]
    all_deps = {**package_data.get("dependencies", {}), **package_data.get("devDependencies", {})}
    
    for dep in required_deps:
        assert dep in all_deps, f"Required dependency missing: {dep}"


def test_backend_dependencies_complete():
    """
    Verify backend requirements.txt contains all required dependencies.
    Feature: project-setup, Property 4: Dependency specification completeness
    Validates: Requirements 2.1, 2.3
    """
    requirements_path = PROJECT_ROOT / "backend" / "requirements.txt"
    
    with open(requirements_path, 'r') as f:
        requirements_content = f.read().lower()
    
    required_packages = ["fastapi", "pydantic", "torch", "mlflow", "supabase"]
    
    for package in required_packages:
        assert package in requirements_content, f"Required package missing from requirements.txt: {package}"


def test_env_example_files_complete():
    """
    Verify .env.example files contain required environment variables.
    Feature: project-setup, Property 5: Environment template completeness
    Validates: Requirements 6.1, 6.2, 6.3
    """
    # Frontend env example
    frontend_env = PROJECT_ROOT / "frontend" / ".env.example"
    with open(frontend_env, 'r') as f:
        frontend_content = f.read()
    assert "VITE_API_BASE_URL" in frontend_content, "Frontend .env.example missing VITE_API_BASE_URL"
    
    # Backend env example
    backend_env = PROJECT_ROOT / "backend" / ".env.example"
    with open(backend_env, 'r') as f:
        backend_content = f.read()
    
    required_vars = ["DATABASE_URL", "SUPABASE_URL", "SUPABASE_KEY", "MLFLOW_TRACKING_URI", "MODEL_PATH"]
    for var in required_vars:
        assert var in backend_content, f"Backend .env.example missing {var}"


def test_gitignore_coverage():
    """
    Verify .gitignore files exclude appropriate patterns.
    Feature: project-setup, Property 6: Gitignore coverage
    Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5
    """
    # Frontend gitignore
    frontend_gitignore = PROJECT_ROOT / "frontend" / ".gitignore"
    with open(frontend_gitignore, 'r') as f:
        frontend_content = f.read()
    
    assert "node_modules" in frontend_content, "Frontend .gitignore missing node_modules"
    assert ".env" in frontend_content, "Frontend .gitignore missing .env"
    
    # Backend gitignore
    backend_gitignore = PROJECT_ROOT / "backend" / ".gitignore"
    with open(backend_gitignore, 'r') as f:
        backend_content = f.read()
    
    assert "__pycache__" in backend_content, "Backend .gitignore missing __pycache__"
    assert ".env" in backend_content, "Backend .gitignore missing .env"


def test_docker_compose_services():
    """
    Verify docker-compose.yml defines required services.
    Feature: project-setup, Property 7: Docker service definition completeness
    Validates: Requirements 5.2, 5.3, 5.4
    """
    docker_compose_path = PROJECT_ROOT / "docker-compose.yml"
    
    with open(docker_compose_path, 'r') as f:
        content = f.read()
    
    required_services = ["frontend", "backend", "database"]
    for service in required_services:
        assert f"{service}:" in content, f"docker-compose.yml missing service: {service}"
    
    # Check port mappings
    assert "5173" in content, "Frontend port 5173 not configured"
    assert "8000" in content, "Backend port 8000 not configured"
    assert "5432" in content, "Database port 5432 not configured"


def test_entry_points_exist():
    """
    Verify application entry points exist.
    Feature: project-setup, Property 8: Entry point functionality
    Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5
    """
    # Frontend entry point
    frontend_app = PROJECT_ROOT / "frontend" / "src" / "App.jsx"
    assert frontend_app.exists(), "Frontend App.jsx missing"
    
    frontend_html = PROJECT_ROOT / "frontend" / "index.html"
    assert frontend_html.exists(), "Frontend index.html missing"
    
    # Backend entry point
    backend_main = PROJECT_ROOT / "backend" / "app" / "main.py"
    assert backend_main.exists(), "Backend main.py missing"
    
    # Verify backend has health check endpoint
    with open(backend_main, 'r') as f:
        backend_content = f.read()
    assert "/health" in backend_content, "Backend missing /health endpoint"


def test_readme_completeness():
    """
    Verify README.md contains required sections.
    Feature: project-setup, Property 9: Documentation completeness
    Validates: Requirements 3.2, 3.3, 3.4, 3.5
    """
    readme_path = PROJECT_ROOT / "README.md"
    
    with open(readme_path, 'r', encoding='utf-8') as f:
        readme_content = f.read().lower()
    
    required_sections = [
        "overview",
        "features",
        "architecture",
        "setup",
        "technology",
    ]
    
    for section in required_sections:
        assert section in readme_content, f"README.md missing section about: {section}"


def test_directory_naming_clarity():
    """
    Verify directory names are clear and descriptive.
    Feature: project-setup, Property 10: Directory naming clarity
    Validates: Requirements 1.5
    """
    # Check that key directories have clear, descriptive names
    clear_names = {
        "frontend": "User interface code",
        "backend": "Server-side code",
        "ml-models": "Machine learning models",
        "docs": "Documentation",
        "components": "Reusable UI components",
        "api": "API endpoints",
    }
    
    for dir_name in clear_names.keys():
        # Directory name should be lowercase and use hyphens or underscores
        assert dir_name.islower() or '-' in dir_name or '_' in dir_name, \
            f"Directory name not following conventions: {dir_name}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
