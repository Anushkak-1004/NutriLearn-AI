# Project Setup Validation

This directory contains validation tests and scripts to verify the NutriLearn AI project structure and configuration.

## Overview

The validation infrastructure ensures that:
- All required directories exist
- Configuration files are present and valid
- Dependencies are properly specified
- Environment templates are complete
- Gitignore patterns are appropriate
- Docker services are configured
- Application entry points exist
- Documentation is comprehensive

## Running Validation

### Option 1: Using pytest (Recommended for CI/CD)

Run all validation tests:
```bash
cd backend
pytest tests/test_project_setup.py -v
```

Run specific test:
```bash
pytest tests/test_project_setup.py::test_directory_structure_exists -v
```

### Option 2: Using Standalone Script (Quick Check)

Run the standalone validation script:
```bash
cd backend
python validate_setup.py
```

With verbose output:
```bash
python validate_setup.py --verbose
```

From a different directory:
```bash
python backend/validate_setup.py --root /path/to/project
```

## Test Coverage

The validation suite covers all 10 correctness properties defined in the design document:

1. **Directory structure completeness** - Verifies all required directories exist
2. **Configuration file presence** - Checks for required config files
3. **Package configuration validity** - Validates package.json structure
4. **Dependency specification completeness** - Ensures all dependencies are listed
5. **Environment template completeness** - Verifies .env.example files
6. **Gitignore coverage** - Checks gitignore patterns
7. **Docker service definition completeness** - Validates docker-compose.yml
8. **Entry point functionality** - Verifies app entry points exist
9. **Documentation completeness** - Checks README sections
10. **Directory naming clarity** - Validates naming conventions

## Validation Results

### Success
All validations pass - project setup is complete and correct.

### Warnings
Non-critical issues that should be reviewed but don't prevent development.

### Errors
Critical issues that must be fixed before development can proceed.

## Integration with Development Workflow

### Pre-commit Hook
Add validation to git pre-commit hooks:
```bash
# .git/hooks/pre-commit
#!/bin/bash
python backend/validate_setup.py
if [ $? -ne 0 ]; then
    echo "Project validation failed. Please fix errors before committing."
    exit 1
fi
```

### CI/CD Pipeline
Include in GitHub Actions or GitLab CI:
```yaml
- name: Validate Project Setup
  run: |
    cd backend
    pytest tests/test_project_setup.py -v
```

### Docker Health Check
Run validation when containers start:
```dockerfile
HEALTHCHECK --interval=30s --timeout=3s \
  CMD python validate_setup.py || exit 1
```

## Adding New Validations

To add a new validation check:

1. Add a test function to `test_project_setup.py`:
```python
def test_new_validation():
    """
    Description of what this validates.
    Feature: project-setup, Property X: Property name
    Validates: Requirements X.Y
    """
    # Validation logic
    assert condition, "Error message"
```

2. Add corresponding method to `validate_setup.py`:
```python
def validate_new_check(self) -> bool:
    """Validate new requirement."""
    print_section("Validating New Requirement")
    # Validation logic
    return True
```

3. Add to `run_all_validations()` method:
```python
validations = [
    # ... existing validations
    self.validate_new_check,
]
```

## Troubleshooting

### Common Issues

**pytest not found**
```bash
pip install pytest
```

**Import errors**
```bash
# Ensure you're in the backend directory
cd backend
python -m pytest tests/test_project_setup.py
```

**Permission denied on validate_setup.py**
```bash
chmod +x validate_setup.py
./validate_setup.py
```

**Path issues**
```bash
# Specify project root explicitly
python validate_setup.py --root /path/to/nutrilearn-ai
```

## Requirements Mapping

Each test maps to specific requirements from the requirements document:

- **Requirement 1** (Project Structure): Tests 1.1-1.5
- **Requirement 2** (Package Configuration): Tests 2.1-2.5
- **Requirement 3** (Documentation): Tests 3.1-3.5
- **Requirement 4** (Gitignore): Tests 4.1-4.5
- **Requirement 5** (Docker): Tests 5.1-5.5
- **Requirement 6** (Environment Config): Tests 6.1-6.5
- **Requirement 7** (Entry Points): Tests 7.1-7.5

## Property-Based Testing

While the current validation uses example-based testing, the design document defines correctness properties that could be extended with property-based testing using Hypothesis:

```python
from hypothesis import given, strategies as st

@given(st.text())
def test_directory_names_are_valid(dirname):
    """Property: All directory names should be lowercase or use separators."""
    assert dirname.islower() or '-' in dirname or '_' in dirname
```

This would provide more comprehensive coverage by testing with generated inputs.

## Maintenance

- Review validation tests when requirements change
- Update property mappings when design document is modified
- Keep validation scripts in sync with project structure changes
- Document any new validation patterns or conventions
