# Design Document: Project Setup

## Overview

The project setup phase establishes the foundational infrastructure for NutriLearn AI, a production-ready MLOps platform for food recognition and nutrition education. This design focuses on creating a well-organized, maintainable codebase structure with proper configuration, documentation, and containerization that will support full-stack development with integrated machine learning capabilities.

The setup prioritizes developer experience through clear separation of concerns, comprehensive documentation, and consistent tooling across frontend (React/Vite), backend (FastAPI), and ML pipeline (PyTorch/MLflow) components.

## Architecture

### High-Level Structure

```
nutrilearn-ai/
├── frontend/               # React application with Vite
├── backend/                # FastAPI service
├── ml-models/              # Trained model artifacts
├── docs/                   # Project documentation
├── docker-compose.yml      # Container orchestration
├── .gitignore              # Root-level ignore rules
└── README.md               # Project documentation
```

### Component Organization

**Frontend Application**
- Built with React 18 and Vite for fast development
- Tailwind CSS for styling
- Organized into components, pages, utils, and hooks
- Axios for HTTP communication with backend

**Backend Service**
- FastAPI framework for high-performance API
- Structured into api (routes), models (Pydantic), ml (inference), mlops (tracking)
- Supabase client for PostgreSQL database access
- PyTorch for ML model inference

**ML Pipeline**
- Separate directory for model artifacts and weights
- MLflow integration for experiment tracking
- Version control ready with DVC support

**Development Environment**
- Docker Compose for consistent local development
- Hot-reloading support for both frontend and backend
- Isolated services with proper networking

## Components and Interfaces

### Directory Structure


#### Frontend Structure
```
frontend/
├── src/
│   ├── components/        # Reusable UI components
│   ├── pages/             # Page-level components
│   ├── utils/             # Helper functions
│   ├── hooks/             # Custom React hooks
│   └── App.jsx            # Main application component
├── public/                # Static assets
├── index.html             # HTML entry point
├── package.json           # Dependencies and scripts
├── .env.example           # Environment variable template
└── .gitignore             # Frontend-specific ignores
```

**Design Rationale**: This structure follows React best practices with clear separation between reusable components, page-level views, utility functions, and custom hooks. The organization scales well as the application grows.

#### Backend Structure
```
backend/
├── app/
│   ├── api/               # API route handlers
│   ├── models/            # Pydantic data models
│   ├── ml/                # ML inference logic
│   ├── mlops/             # MLflow integration
│   ├── database.py        # Database connection
│   └── main.py            # FastAPI application entry
├── tests/                 # Test suite
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variable template
└── .gitignore             # Backend-specific ignores
```

**Design Rationale**: The backend follows a modular architecture separating API routes, data models, ML logic, and MLOps concerns. This enables independent development and testing of each component while maintaining clear boundaries.

### Configuration Files

#### Package Management

**Frontend (package.json)**
- React 18.x for modern component features
- Vite for fast build tooling and HMR
- Tailwind CSS for utility-first styling
- Axios for HTTP requests
- Node.js version specification for consistency

**Backend (requirements.txt)**
- FastAPI with uvicorn for ASGI server
- Pydantic for data validation
- PyTorch for ML inference
- MLflow for experiment tracking
- Supabase client for database operations
- Python 3.9+ compatibility

**Design Rationale**: These dependencies represent the minimal viable set for a production-ready MLOps application. Each library serves a specific purpose without redundancy.

#### Environment Configuration

**Frontend (.env.example)**
```
VITE_API_BASE_URL=http://localhost:8000/api/v1
```

**Backend (.env.example)**
```
DATABASE_URL=postgresql://user:password@localhost:5432/nutrilearn
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_KEY=your-anon-key
MLFLOW_TRACKING_URI=http://localhost:5000
MODEL_PATH=../ml-models/food_classifier.pth
```

**Design Rationale**: Environment templates document all required configuration without exposing secrets. Developers copy these to create local .env files that are gitignored.

#### Docker Configuration

**docker-compose.yml Services**
- **frontend**: React dev server on port 5173 with volume mount for hot-reload
- **backend**: FastAPI server on port 8000 with volume mount for hot-reload
- **database**: PostgreSQL on port 5432 with persistent volume

**Design Rationale**: Docker Compose provides consistent development environments across machines. Volume mounts enable hot-reloading without rebuilding containers. Port mappings follow standard conventions.

### Gitignore Strategy

**Root .gitignore**
- Docker volumes and data
- IDE configurations (.vscode, .idea)
- OS files (.DS_Store, Thumbs.db)
- Secret files (*.env, *.key)

**Frontend .gitignore**
- node_modules/
- dist/ and build/
- .env and .env.local

**Backend .gitignore**
- __pycache__/ and *.pyc
- venv/ and .venv/
- *.pth and *.pkl (model files)
- .env

**Design Rationale**: Multi-level gitignore files prevent committing dependencies, build artifacts, and secrets while allowing configuration templates. This protects sensitive data and keeps the repository clean.

## Data Models

### File System Structure Model

The project structure itself serves as a data model defining the organization of code, configuration, and documentation:

```
Project
├── Frontend (React SPA)
├── Backend (FastAPI Service)
├── ML Models (Artifacts)
├── Documentation
└── Configuration (Docker, Git, Env)
```

### Configuration Models

**Package Configuration**
- Dependencies with version constraints
- Scripts for common development tasks
- Metadata (name, version, description)

**Environment Configuration**
- API endpoints and URLs
- Database credentials
- Service API keys
- File paths

**Design Rationale**: These models establish contracts between different parts of the system. Package files define what libraries are available, environment files define how services connect.

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system—essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Directory structure completeness
*For any* valid project setup, all required directories (frontend/src/components, frontend/src/pages, frontend/src/utils, frontend/src/hooks, backend/app/api, backend/app/models, backend/app/ml, backend/app/mlops, backend/tests, ml-models, docs) must exist in the file system.

**Validates: Requirements 1.1, 1.2, 1.3, 1.4**

### Property 2: Configuration file presence
*For any* valid project setup, all required configuration files (frontend/package.json, backend/requirements.txt, docker-compose.yml, README.md, .gitignore files) must exist at their specified locations.

**Validates: Requirements 2.1, 2.2, 3.1, 4.1, 4.2, 4.3, 5.1**

### Property 3: Package configuration validity
*For any* package.json file in the project, it must be valid JSON and contain required fields (name, version, dependencies, scripts).

**Validates: Requirements 2.1, 2.2**

### Property 4: Dependency specification completeness
*For any* package configuration file (package.json or requirements.txt), all dependencies specified in the requirements document must be present.

**Validates: Requirements 2.1, 2.3**

### Property 5: Environment template completeness
*For any* .env.example file, all environment variables required by the corresponding service must be documented with placeholder values.

**Validates: Requirements 6.1, 6.2, 6.3**

### Property 6: Gitignore coverage
*For any* file type that should be excluded from version control (node_modules, __pycache__, .env, build artifacts), there must exist a .gitignore rule that matches it.

**Validates: Requirements 4.1, 4.2, 4.3, 4.4, 4.5**

### Property 7: Docker service definition completeness
*For any* required service (frontend, backend, database), the docker-compose.yml must define that service with proper port mappings and volume mounts.

**Validates: Requirements 5.2, 5.3, 5.4**

### Property 8: Entry point functionality
*For any* application entry point (frontend App.jsx, backend main.py), executing it must not produce errors and must provide the expected initial response.

**Validates: Requirements 7.1, 7.2, 7.3, 7.4, 7.5**

### Property 9: Documentation completeness
*For any* README.md file, it must contain all required sections (project overview, features, architecture, setup instructions, technology stack, API documentation structure, MLOps pipeline explanation).

**Validates: Requirements 3.2, 3.3, 3.4, 3.5**

### Property 10: Directory naming clarity
*For any* directory in the project structure, its name must clearly indicate its purpose without requiring additional documentation.

**Validates: Requirements 1.5**

## Error Handling

### File System Operations

**Missing Directories**
- Validation scripts should check for required directories
- Clear error messages indicating which directories are missing
- Automated creation scripts to fix missing structure

**Invalid Configuration Files**
- JSON/YAML parsing errors should be caught and reported
- Validation against schemas where applicable
- Helpful error messages pointing to the problematic line

**Permission Issues**
- Check write permissions before creating files
- Provide clear instructions for fixing permission problems
- Graceful degradation when possible

### Dependency Management

**Missing Dependencies**
- Package managers (npm, pip) handle missing dependencies
- Lock files ensure consistent versions
- Clear error messages when installation fails

**Version Conflicts**
- Specify compatible version ranges in configuration
- Document known incompatibilities
- Use lock files to prevent drift

### Docker Issues

**Port Conflicts**
- Check if ports are already in use before starting
- Provide alternative port configuration options
- Clear error messages with resolution steps

**Volume Mount Failures**
- Validate paths exist before mounting
- Check permissions on mounted directories
- Provide troubleshooting guidance

### Environment Configuration

**Missing Environment Variables**
- Application startup should validate required variables
- Fail fast with clear error messages
- Reference .env.example for required variables

**Invalid Values**
- Validate format of URLs, paths, and credentials
- Provide examples of valid values
- Test connections during startup when possible

## Testing Strategy

### Validation Scripts

**Structure Validation**
- Script to verify all required directories exist
- Check for required configuration files
- Validate file permissions

**Configuration Validation**
- Parse and validate JSON/YAML files
- Check for required fields in package files
- Verify environment templates are complete

**Integration Testing**
- Test Docker Compose can start all services
- Verify services can communicate
- Check health endpoints respond correctly

### Property-Based Testing

The correctness properties defined above will be implemented using property-based testing to verify the project setup across various scenarios:

**Testing Framework**: pytest with hypothesis for Python validation scripts

**Test Configuration**: Each property test will run a minimum of 100 iterations to ensure robustness

**Test Organization**:
- Setup validation tests in `tests/test_project_setup.py`
- Each correctness property maps to one property-based test
- Tests tagged with property numbers for traceability

**Example Test Structure**:
```python
# Feature: project-setup, Property 1: Directory structure completeness
@given(project_root=valid_project_paths())
def test_directory_structure_completeness(project_root):
    """Verify all required directories exist."""
    required_dirs = [
        "frontend/src/components",
        "frontend/src/pages",
        # ... all required directories
    ]
    for dir_path in required_dirs:
        assert (project_root / dir_path).exists()
```

### Unit Testing

**Configuration File Tests**
- Test package.json contains required dependencies
- Test requirements.txt has correct format
- Test docker-compose.yml is valid YAML

**Documentation Tests**
- Test README.md contains required sections
- Test all links in documentation are valid
- Test code examples in docs are syntactically correct

**Gitignore Tests**
- Test specific file patterns are ignored
- Test templates are not ignored
- Test no secrets are committed

### Manual Verification

**Developer Experience**
- Follow setup instructions from README
- Verify all services start successfully
- Check hot-reloading works for frontend and backend
- Confirm environment variables are properly documented

**Documentation Quality**
- Review README for clarity and completeness
- Verify architecture diagrams are accurate
- Check setup instructions are step-by-step

## Implementation Notes

### Setup Order

1. Create root directory structure
2. Initialize frontend with package.json and basic structure
3. Initialize backend with requirements.txt and basic structure
4. Create Docker Compose configuration
5. Write comprehensive README
6. Configure gitignore files
7. Create environment templates
8. Add minimal entry points for verification

**Rationale**: This order ensures dependencies are available before they're needed and allows incremental verification at each step.

### Technology Choices

**Vite over Create React App**
- Faster build times and HMR
- Better developer experience
- Modern tooling with ES modules

**FastAPI over Flask**
- Built-in data validation with Pydantic
- Automatic API documentation
- Async support for better performance
- Type hints throughout

**Docker Compose over Manual Setup**
- Consistent environments across developers
- Easy service orchestration
- Simplified onboarding for new developers

**Supabase over Raw PostgreSQL**
- Managed database with built-in auth
- Real-time subscriptions
- RESTful API out of the box
- Reduces infrastructure complexity

### Scalability Considerations

**Modular Structure**
- Each component can be developed independently
- Clear interfaces between frontend, backend, and ML
- Easy to add new features without restructuring

**Container-Ready**
- Docker setup prepares for cloud deployment
- Services can be scaled independently
- Configuration externalized for different environments

**Documentation-First**
- Comprehensive README reduces onboarding time
- Clear architecture enables parallel development
- Well-documented setup prevents common issues

### Security Considerations

**Secret Management**
- No secrets in version control
- Environment templates guide proper configuration
- Gitignore prevents accidental commits

**Dependency Security**
- Lock files ensure reproducible builds
- Regular updates to patch vulnerabilities
- Minimal dependencies reduce attack surface

## Future Enhancements

While not part of the initial setup, the structure supports:

- CI/CD pipeline integration (GitHub Actions, GitLab CI)
- Production deployment configurations
- Monitoring and logging infrastructure
- Additional services (Redis cache, message queue)
- Multi-environment configurations (dev, staging, prod)

The modular design ensures these enhancements can be added without major restructuring.
