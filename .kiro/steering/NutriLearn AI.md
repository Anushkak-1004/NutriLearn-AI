# NutriLearn AI - Agent Steering Document

## Project Overview
Name: NutriLearn AI
Type: Food Recognition & Nutrition Education Platform  
Purpose: B.Tech Final Year AI/ML Project
Goal: Production-ready MLOps project for job interviews

## Technical Stack
### Frontend
- Framework: React 18 with Vite
- Styling: Tailwind CSS
- Language: JavaScript (TypeScript if complex)
- State: React Hooks (useState, useEffect)
- HTTP Client: Axios

### Backend
- Framework: FastAPI (Python 3.9+)
- Validation: Pydantic models
- Database: Supabase (PostgreSQL)
- ORM: SQLAlchemy or Supabase client

### Machine Learning
- Framework: PyTorch
- Pre-trained: Use MobileNet/ResNet for transfer learning
- Food dataset: Food-101 or custom Indian foods
- Inference: Optimized for CPU (lightweight)

### MLOps
- Experiment Tracking: MLflow
- Version Control: Git + DVC for data
- Containerization: Docker + docker-compose
- Deployment: Hugging Face Spaces

## Coding Standards

### Python (Backend & ML)
- Always use type hints for function parameters and returns
- Write comprehensive docstrings (Google style)
- Follow PEP 8 style guide strictly
- Use f-strings for string formatting
- Implement proper error handling with try-except
- Add logging for debugging (using Python logging module)
- Validate all inputs with Pydantic models

Example:
```python
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

def process_food_image(
    image_path: str,
    confidence_threshold: float = 0.8
) -> Optional[dict]:
    """
    Process uploaded food image and return predictions.
    
    Args:
        image_path: Path to the uploaded image file
        confidence_threshold: Minimum confidence score to accept prediction
        
    Returns:
        Dictionary with food name, confidence, and nutrition info
        Returns None if prediction confidence is below threshold
        
    Raises:
        ValueError: If image_path is invalid
        IOError: If image cannot be read
    """
    try:
        # Implementation
        logger.info(f"Processing image: {image_path}")
        # ...
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise
```

### JavaScript/React (Frontend)
- Use functional components (no class components)
- Use React Hooks for state and side effects
- Keep components small and focused (single responsibility)
- Use meaningful variable names (avoid single letters except i, j)
- Add PropTypes or TypeScript for type checking
- Implement error boundaries for error handling
- Use async/await for API calls (not .then())

Example:
```javascript
import { useState, useEffect } from 'react';
import axios from 'axios';

function FoodAnalyzer({ userId }) {
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const analyzeFood = async (imageFile) => {
    setLoading(true);
    setError(null);
    
    try {
      const formData = new FormData();
      formData.append('file', imageFile);
      
      const response = await axios.post('/api/v1/predict', formData);
      setPrediction(response.data);
    } catch (err) {
      setError(err.message);
      console.error('Prediction failed:', err);
    } finally {
      setLoading(false);
    }
  };

  return (
    // Component JSX
  );
}
```

### API Design
- RESTful conventions: GET (read), POST (create), PUT (update), DELETE (delete)
- Versioned endpoints: /api/v1/resource
- Consistent response format: { status, data, error, timestamp }
- Proper HTTP status codes: 200 (success), 201 (created), 400 (bad request), 404 (not found), 500 (server error)
- Include request validation with detailed error messages
- Add rate limiting for production
- CORS configuration for frontend access

### Testing
- Write unit tests for all business logic functions
- Backend: Use pytest with fixtures
- Frontend: Use Jest + React Testing Library
- Aim for 80%+ code coverage
- Include edge cases and error scenarios
- Test API endpoints with different inputs

### Documentation
- Comprehensive README.md with:
  - Project overview and features
  - Architecture diagram
  - Setup instructions
  - API documentation
  - MLOps pipeline explanation
- Inline code comments for complex logic
- Docstrings for all functions/classes
- API documentation with request/response examples

## Project Structure
```
nutrilearn-ai/
├── frontend/               # React application
│   ├── src/
│   │   ├── components/    # Reusable UI components
│   │   ├── pages/         # Page-level components
│   │   ├── utils/         # Helper functions
│   │   ├── hooks/         # Custom React hooks
│   │   └── App.jsx        # Main app component
│   ├── public/
│   └── package.json
├── backend/               # FastAPI application
│   ├── app/
│   │   ├── api/          # API routes
│   │   ├── models/       # Pydantic models
│   │   ├── ml/           # ML model code
│   │   ├── mlops/        # MLflow integration
│   │   ├── database.py   # DB connection
│   │   └── main.py       # FastAPI app
│   ├── tests/
│   └── requirements.txt
├── ml-models/             # Trained model files
├── docs/                  # Documentation
├── docker-compose.yml
└── README.md
```

## Development Workflow
1. Use SPECS mode for planning complex features
2. Break features into small, testable tasks
3. Write tests alongside code (TDD when possible)
4. Commit frequently with descriptive messages
5. Document as you code (not after)
6. Run tests before committing
7. Update README with new features

## Special Considerations

### Performance
- Optimize ML model inference (use smaller models if needed)
- Implement image compression before upload
- Add caching for repeated predictions
- Use pagination for meal history
- Lazy load images and components

### Security
- Never commit secrets or API keys
- Use environment variables for configuration
- Validate all user inputs server-side
- Sanitize data before database operations
- Implement rate limiting on API endpoints

### MLOps Best Practices
- Log all model predictions with MLflow
- Track model versions (v1, v2, etc.)
- Monitor prediction confidence distribution
- Implement A/B testing capability
- Add model performance dashboards
- Document model training process

### Indian Context
- Support Indian food items (biryani, dosa, dal, etc.)
- Include regional cuisines
- Use appropriate nutrition databases
- Support multiple languages (future enhancement)

## Interview Talking Points
When explaining this project:
1. Emphasize end-to-end ML pipeline (not just model training)
2. Highlight MLOps practices (tracking, versioning, monitoring)
3. Discuss trade-offs in tech choices
4. Explain scalability considerations
5. Show understanding of production challenges
6. Demonstrate full-stack capabilities

## Communication Style
- Explain complex concepts clearly
- Provide context for technical decisions
- Suggest best practices proactively
- Ask clarifying questions when requirements are ambiguous
- Offer alternatives when appropriate

## What to Avoid
- Hardcoded credentials or secrets
- Overly complex solutions when simple ones work
- Missing error handling
- Incomplete tests
- Poor variable naming
- Missing documentation
- console.log in production code
- Unvalidated user inputs
```

---

## ✅ **After Pasting:**

**Save the file** (Ctrl+S / Cmd+S)

You should see:
- ✅ Green checkmark or "Saved" indicator
- The steering document is now active for this project

---

## 🚀 **Next Step:**

Now you're ready to start building!

**Open the Kiro chat panel and paste:**
```
Hello! I've set up my agent steering for NutriLearn AI. 
Let's start by creating the initial project structure.

Please create:
1. Folder structure (frontend/, backend/, ml-models/, docs/)
2. Package configuration files (package.json, requirements.txt)
3. Initial README.md with project overview
4. .gitignore files for both frontend and backend
5. docker-compose.yml skeleton

Follow the structure defined in my agent steering document.