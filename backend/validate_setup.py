#!/usr/bin/env python3
"""
Project Setup Validation Script

Standalone script to validate the NutriLearn AI project structure and configuration.
Can be run independently without pytest to quickly verify project setup.

Usage:
    python validate_setup.py
    python validate_setup.py --verbose
"""

import os
import json
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import argparse


# ANSI color codes for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_success(message: str) -> None:
    """Print success message in green."""
    print(f"{Colors.GREEN}✓{Colors.RESET} {message}")


def print_error(message: str) -> None:
    """Print error message in red."""
    print(f"{Colors.RED}✗{Colors.RESET} {message}")


def print_warning(message: str) -> None:
    """Print warning message in yellow."""
    print(f"{Colors.YELLOW}⚠{Colors.RESET} {message}")


def print_section(message: str) -> None:
    """Print section header in blue."""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{message}{Colors.RESET}")


class ProjectValidator:
    """Validates NutriLearn AI project structure and configuration."""
    
    def __init__(self, project_root: Path, verbose: bool = False):
        """
        Initialize validator.
        
        Args:
            project_root: Path to project root directory
            verbose: Enable verbose output
        """
        self.project_root = project_root
        self.verbose = verbose
        self.errors: List[str] = []
        self.warnings: List[str] = []
    
    def validate_directories(self) -> bool:
        """
        Validate all required directories exist.
        
        Returns:
            True if all directories exist, False otherwise
        """
        print_section("Validating Directory Structure")
        
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
        
        all_exist = True
        for dir_path in required_dirs:
            full_path = self.project_root / dir_path
            if full_path.exists() and full_path.is_dir():
                if self.verbose:
                    print_success(f"Directory exists: {dir_path}")
            else:
                print_error(f"Missing directory: {dir_path}")
                self.errors.append(f"Missing directory: {dir_path}")
                all_exist = False
        
        if all_exist:
            print_success("All required directories exist")
        
        return all_exist
    
    def validate_configuration_files(self) -> bool:
        """
        Validate all required configuration files exist.
        
        Returns:
            True if all files exist, False otherwise
        """
        print_section("Validating Configuration Files")
        
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
        
        all_exist = True
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists() and full_path.is_file():
                if self.verbose:
                    print_success(f"File exists: {file_path}")
            else:
                print_error(f"Missing file: {file_path}")
                self.errors.append(f"Missing file: {file_path}")
                all_exist = False
        
        if all_exist:
            print_success("All required configuration files exist")
        
        return all_exist
    
    def validate_package_json(self) -> bool:
        """
        Validate package.json is valid JSON and contains required fields.
        
        Returns:
            True if valid, False otherwise
        """
        print_section("Validating Frontend Package Configuration")
        
        package_json_path = self.project_root / "frontend" / "package.json"
        
        if not package_json_path.exists():
            print_error("package.json not found")
            return False
        
        try:
            with open(package_json_path, 'r') as f:
                package_data = json.load(f)
            
            print_success("package.json is valid JSON")
            
            # Check required fields
            required_fields = ["name", "version", "dependencies", "scripts"]
            missing_fields = [field for field in required_fields if field not in package_data]
            
            if missing_fields:
                for field in missing_fields:
                    print_error(f"Missing required field: {field}")
                    self.errors.append(f"package.json missing field: {field}")
                return False
            
            print_success("All required fields present in package.json")
            
            # Check key dependencies
            required_deps = ["react", "axios"]
            all_deps = {**package_data.get("dependencies", {}), **package_data.get("devDependencies", {})}
            missing_deps = [dep for dep in required_deps if dep not in all_deps]
            
            if missing_deps:
                for dep in missing_deps:
                    print_error(f"Missing required dependency: {dep}")
                    self.errors.append(f"Missing dependency: {dep}")
                return False
            
            print_success("All required dependencies present")
            return True
            
        except json.JSONDecodeError as e:
            print_error(f"Invalid JSON in package.json: {e}")
            self.errors.append(f"Invalid JSON in package.json: {e}")
            return False
        except Exception as e:
            print_error(f"Error reading package.json: {e}")
            self.errors.append(f"Error reading package.json: {e}")
            return False
    
    def validate_requirements_txt(self) -> bool:
        """
        Validate requirements.txt contains required packages.
        
        Returns:
            True if valid, False otherwise
        """
        print_section("Validating Backend Dependencies")
        
        requirements_path = self.project_root / "backend" / "requirements.txt"
        
        if not requirements_path.exists():
            print_error("requirements.txt not found")
            return False
        
        try:
            with open(requirements_path, 'r') as f:
                requirements_content = f.read().lower()
            
            required_packages = ["fastapi", "pydantic", "torch", "mlflow", "supabase"]
            missing_packages = []
            
            for package in required_packages:
                if package in requirements_content:
                    if self.verbose:
                        print_success(f"Package found: {package}")
                else:
                    print_error(f"Missing required package: {package}")
                    missing_packages.append(package)
            
            if missing_packages:
                self.errors.extend([f"Missing package: {pkg}" for pkg in missing_packages])
                return False
            
            print_success("All required packages present in requirements.txt")
            return True
            
        except Exception as e:
            print_error(f"Error reading requirements.txt: {e}")
            self.errors.append(f"Error reading requirements.txt: {e}")
            return False
    
    def validate_env_examples(self) -> bool:
        """
        Validate .env.example files contain required variables.
        
        Returns:
            True if valid, False otherwise
        """
        print_section("Validating Environment Templates")
        
        all_valid = True
        
        # Frontend env example
        frontend_env = self.project_root / "frontend" / ".env.example"
        if frontend_env.exists():
            with open(frontend_env, 'r') as f:
                frontend_content = f.read()
            
            if "VITE_API_BASE_URL" in frontend_content:
                print_success("Frontend .env.example contains VITE_API_BASE_URL")
            else:
                print_error("Frontend .env.example missing VITE_API_BASE_URL")
                self.errors.append("Frontend .env.example missing VITE_API_BASE_URL")
                all_valid = False
        else:
            print_error("Frontend .env.example not found")
            all_valid = False
        
        # Backend env example
        backend_env = self.project_root / "backend" / ".env.example"
        if backend_env.exists():
            with open(backend_env, 'r') as f:
                backend_content = f.read()
            
            required_vars = ["DATABASE_URL", "SUPABASE_URL", "SUPABASE_KEY", "MLFLOW_TRACKING_URI", "MODEL_PATH"]
            missing_vars = [var for var in required_vars if var not in backend_content]
            
            if missing_vars:
                for var in missing_vars:
                    print_error(f"Backend .env.example missing: {var}")
                    self.errors.append(f"Backend .env.example missing: {var}")
                all_valid = False
            else:
                print_success("Backend .env.example contains all required variables")
        else:
            print_error("Backend .env.example not found")
            all_valid = False
        
        return all_valid
    
    def validate_gitignore(self) -> bool:
        """
        Validate .gitignore files have appropriate patterns.
        
        Returns:
            True if valid, False otherwise
        """
        print_section("Validating Gitignore Configuration")
        
        all_valid = True
        
        # Frontend gitignore
        frontend_gitignore = self.project_root / "frontend" / ".gitignore"
        if frontend_gitignore.exists():
            with open(frontend_gitignore, 'r') as f:
                frontend_content = f.read()
            
            required_patterns = ["node_modules", ".env"]
            missing = [p for p in required_patterns if p not in frontend_content]
            
            if missing:
                for pattern in missing:
                    print_error(f"Frontend .gitignore missing pattern: {pattern}")
                    self.errors.append(f"Frontend .gitignore missing: {pattern}")
                all_valid = False
            else:
                print_success("Frontend .gitignore has required patterns")
        else:
            print_error("Frontend .gitignore not found")
            all_valid = False
        
        # Backend gitignore
        backend_gitignore = self.project_root / "backend" / ".gitignore"
        if backend_gitignore.exists():
            with open(backend_gitignore, 'r') as f:
                backend_content = f.read()
            
            required_patterns = ["__pycache__", ".env"]
            missing = [p for p in required_patterns if p not in backend_content]
            
            if missing:
                for pattern in missing:
                    print_error(f"Backend .gitignore missing pattern: {pattern}")
                    self.errors.append(f"Backend .gitignore missing: {pattern}")
                all_valid = False
            else:
                print_success("Backend .gitignore has required patterns")
        else:
            print_error("Backend .gitignore not found")
            all_valid = False
        
        return all_valid
    
    def validate_docker_compose(self) -> bool:
        """
        Validate docker-compose.yml defines required services.
        
        Returns:
            True if valid, False otherwise
        """
        print_section("Validating Docker Configuration")
        
        docker_compose_path = self.project_root / "docker-compose.yml"
        
        if not docker_compose_path.exists():
            print_error("docker-compose.yml not found")
            return False
        
        try:
            with open(docker_compose_path, 'r') as f:
                content = f.read()
            
            required_services = ["frontend", "backend", "database"]
            missing_services = []
            
            for service in required_services:
                if f"{service}:" in content:
                    if self.verbose:
                        print_success(f"Service defined: {service}")
                else:
                    print_error(f"Missing service: {service}")
                    missing_services.append(service)
            
            if missing_services:
                self.errors.extend([f"Missing service: {svc}" for svc in missing_services])
                return False
            
            # Check port mappings
            required_ports = {"5173": "frontend", "8000": "backend", "5432": "database"}
            missing_ports = []
            
            for port, service in required_ports.items():
                if port in content:
                    if self.verbose:
                        print_success(f"Port {port} configured for {service}")
                else:
                    print_warning(f"Port {port} not found for {service}")
                    self.warnings.append(f"Port {port} not configured")
            
            print_success("All required services defined in docker-compose.yml")
            return True
            
        except Exception as e:
            print_error(f"Error reading docker-compose.yml: {e}")
            self.errors.append(f"Error reading docker-compose.yml: {e}")
            return False
    
    def validate_entry_points(self) -> bool:
        """
        Validate application entry points exist.
        
        Returns:
            True if valid, False otherwise
        """
        print_section("Validating Application Entry Points")
        
        all_valid = True
        
        # Frontend entry points
        frontend_app = self.project_root / "frontend" / "src" / "App.jsx"
        if frontend_app.exists():
            print_success("Frontend App.jsx exists")
        else:
            print_error("Frontend App.jsx not found")
            self.errors.append("Frontend App.jsx missing")
            all_valid = False
        
        frontend_html = self.project_root / "frontend" / "index.html"
        if frontend_html.exists():
            print_success("Frontend index.html exists")
        else:
            print_error("Frontend index.html not found")
            self.errors.append("Frontend index.html missing")
            all_valid = False
        
        # Backend entry point
        backend_main = self.project_root / "backend" / "app" / "main.py"
        if backend_main.exists():
            print_success("Backend main.py exists")
            
            # Check for health endpoint
            with open(backend_main, 'r') as f:
                backend_content = f.read()
            
            if "/health" in backend_content:
                print_success("Backend has /health endpoint")
            else:
                print_warning("Backend missing /health endpoint")
                self.warnings.append("Backend missing /health endpoint")
        else:
            print_error("Backend main.py not found")
            self.errors.append("Backend main.py missing")
            all_valid = False
        
        return all_valid
    
    def validate_readme(self) -> bool:
        """
        Validate README.md contains required sections.
        
        Returns:
            True if valid, False otherwise
        """
        print_section("Validating Documentation")
        
        readme_path = self.project_root / "README.md"
        
        if not readme_path.exists():
            print_error("README.md not found")
            return False
        
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                readme_content = f.read().lower()
            
            required_sections = [
                ("overview", "Project overview"),
                ("features", "Features list"),
                ("architecture", "Architecture description"),
                ("setup", "Setup instructions"),
                ("technology", "Technology stack"),
            ]
            
            missing_sections = []
            for keyword, description in required_sections:
                if keyword in readme_content:
                    if self.verbose:
                        print_success(f"Section found: {description}")
                else:
                    print_warning(f"Section may be missing: {description}")
                    self.warnings.append(f"README may be missing: {description}")
            
            print_success("README.md exists with content")
            return True
            
        except Exception as e:
            print_error(f"Error reading README.md: {e}")
            self.errors.append(f"Error reading README.md: {e}")
            return False
    
    def run_all_validations(self) -> bool:
        """
        Run all validation checks.
        
        Returns:
            True if all validations pass, False otherwise
        """
        print(f"\n{Colors.BOLD}NutriLearn AI - Project Setup Validation{Colors.RESET}")
        print(f"Project Root: {self.project_root}\n")
        
        validations = [
            self.validate_directories,
            self.validate_configuration_files,
            self.validate_package_json,
            self.validate_requirements_txt,
            self.validate_env_examples,
            self.validate_gitignore,
            self.validate_docker_compose,
            self.validate_entry_points,
            self.validate_readme,
        ]
        
        results = [validation() for validation in validations]
        all_passed = all(results)
        
        # Print summary
        print_section("Validation Summary")
        
        if all_passed and not self.warnings:
            print_success(f"{Colors.BOLD}All validations passed!{Colors.RESET}")
            return True
        elif all_passed and self.warnings:
            print_success(f"{Colors.BOLD}All validations passed with {len(self.warnings)} warnings{Colors.RESET}")
            for warning in self.warnings:
                print_warning(warning)
            return True
        else:
            print_error(f"{Colors.BOLD}Validation failed with {len(self.errors)} errors{Colors.RESET}")
            for error in self.errors:
                print_error(error)
            if self.warnings:
                print(f"\n{Colors.YELLOW}Warnings:{Colors.RESET}")
                for warning in self.warnings:
                    print_warning(warning)
            return False


def main():
    """Main entry point for validation script."""
    parser = argparse.ArgumentParser(
        description="Validate NutriLearn AI project setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Project root directory (default: parent of script directory)"
    )
    
    args = parser.parse_args()
    
    # Determine project root
    if args.root:
        project_root = Path(args.root).resolve()
    else:
        # Assume script is in backend/ directory
        project_root = Path(__file__).parent.parent.resolve()
    
    if not project_root.exists():
        print_error(f"Project root does not exist: {project_root}")
        sys.exit(1)
    
    # Run validation
    validator = ProjectValidator(project_root, verbose=args.verbose)
    success = validator.run_all_validations()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
