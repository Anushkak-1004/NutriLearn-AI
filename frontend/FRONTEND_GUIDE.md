# NutriLearn AI - Frontend Guide

## 🎨 Overview

The NutriLearn AI frontend is a modern React application built with Vite, React Router, and Tailwind CSS. It provides an intuitive interface for food recognition, meal tracking, dietary analysis, and personalized nutrition education.

## 📁 Project Structure

```
frontend/src/
├── App.jsx                    # Main app with routing
├── main.jsx                   # Entry point
├── index.css                  # Global styles
├── components/
│   └── Navigation.jsx         # Top navigation bar
├── pages/
│   ├── HomePage.jsx           # Landing page
│   ├── AnalyzePage.jsx        # Food analysis
│   ├── DashboardPage.jsx      # User dashboard
│   └── LearningPage.jsx       # Learning modules
└── utils/
    ├── api.js                 # API client
    └── storage.js             # LocalStorage utilities
```

## 🚀 Getting Started

### Install Dependencies

```bash
cd frontend
npm install
```

### Start Development Server

```bash
npm run dev
```

The app will be available at http://localhost:5173

### Build for Production

```bash
npm run build
```

## 📄 Pages

### 1. HomePage (`/`)

**Purpose:** Landing page with hero section and action cards

**Features:**
- Hero section with gradient background
- Brain icon and tagline
- Points badge display
- Two action cards: "Analyze Food" and "My Progress"
- "How It Works" section with 3 steps

**Key Components:**
- Gradient background: `bg-gradient-to-br from-green-50 to-blue-50`
- Action cards with hover effects
- Icon-based step indicators

### 2. AnalyzePage (`/analyze`)

**Purpose:** Upload and analyze food images

**Features:**
- File upload with drag-and-drop area
- Image preview
- AI-powered food recognition
- Nutrition display (5 colored cards)
- Meal type selection (breakfast/lunch/dinner/snack)
- Meal logging functionality
- Success message with redirect

**User Flow:**
1. Upload food image
2. Click "Analyze Food"
3. View prediction results
4. Select meal type
5. Log meal
6. Redirect to dashboard

**API Calls:**
- `predictFood(formData)` - Analyze image
- `logMeal(userId, mealData)` - Log meal

### 3. DashboardPage (`/dashboard`)

**Purpose:** User progress tracking and personalized recommendations

**Features:**
- Top stats cards (meals, modules, points)
- Today's nutrition summary
- Recent meals list
- Dietary pattern analysis
- Learning module recommendations
- Minimum 3 meals requirement for analysis

**Layout:**
- Responsive grid (1 col mobile, 2 cols desktop)
- Left panel: Nutrition & meals
- Right panel: Learning path

**API Calls:**
- `getUserStats(userId)` - Get user statistics
- `getMealHistory(userId, limit)` - Get recent meals
- `getDietaryAnalysis(userId)` - Get patterns and recommendations

### 4. LearningPage (`/learning/:moduleId`)

**Purpose:** Educational content and quizzes

**Features:**
- Module header with title and points
- Content sections (text, lists, comparisons, tips)
- Interactive quiz with multiple choice
- Immediate feedback on answers
- Score calculation
- Points award for passing (70%+)
- Retry option for failed attempts

**Content Types:**
- `text` - Paragraphs
- `infographic` - Highlighted info
- `list` - Bullet points
- `comparison` - Good vs. limit foods
- `tips` - Actionable advice
- `schedule` - Meal timing

**API Calls:**
- `completeModule(userId, moduleId, quizScore)` - Submit completion

## 🧩 Components

### Navigation

**Features:**
- Logo with Brain icon
- Navigation links (Home, Analyze, Dashboard)
- Points badge with live updates
- Active link highlighting
- Mobile-responsive menu

**State Management:**
- Fetches points every 30 seconds
- Highlights current route

## 🔧 Utilities

### API Client (`utils/api.js`)

Axios-based client with:
- Base URL configuration
- Request/response interceptors
- Error handling
- Logging

**Functions:**
- `predictFood(formData)` - Upload image for prediction
- `logMeal(userId, mealData)` - Log a meal
- `getUserStats(userId)` - Get user statistics
- `getMealHistory(userId, limit)` - Get meal history
- `getDietaryAnalysis(userId, days)` - Get dietary analysis
- `completeModule(userId, moduleId, quizScore)` - Complete module

### Storage (`utils/storage.js`)

LocalStorage management:
- `getUserId()` - Get or generate user ID
- `clearUserData()` - Clear user data
- `getUserPoints()` - Get stored points
- `setUserPoints(points)` - Store points

**User ID Generation:**
- Uses `crypto.randomUUID()` if available
- Fallback UUID generator for older browsers
- Stored in `localStorage` with key `nutrilearn_userId`

## 🎨 Styling

### Tailwind CSS

**Color Scheme:**
- Primary: Emerald (`emerald-600`, `emerald-700`)
- Secondary: Blue (`blue-600`, `blue-700`)
- Accent: Amber/Orange (`amber-400`, `orange-500`)
- Nutrition colors:
  - Calories: Red/Orange gradient
  - Protein: Pink/Rose gradient
  - Carbs: Amber/Yellow gradient
  - Fat: Blue/Cyan gradient
  - Fiber: Green/Emerald gradient

**Common Patterns:**
- Cards: `rounded-2xl shadow-lg hover:shadow-xl transition-all`
- Buttons: `rounded-xl py-3 px-6 font-semibold`
- Gradients: `bg-gradient-to-br from-{color}-50 to-{color}-50`
- Hover effects: `transform hover:-translate-y-1`

**Responsive Design:**
- Mobile-first approach
- Breakpoints: `md:` (768px), `lg:` (1024px)
- Grid layouts: `grid-cols-1 md:grid-cols-2 lg:grid-cols-3`

## 🔄 State Management

### Local State (useState)

Each page manages its own state:
- Loading states
- Form data
- API responses
- UI states (modals, dropdowns)

### User Context

User ID is managed globally via localStorage:
- Generated on first visit
- Persists across sessions
- Used for all API calls

## 📡 API Integration

### Base URL

Configured via environment variable:
```javascript
baseURL: import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'
```

### Error Handling

All API calls use try-catch:
```javascript
try {
  const result = await predictFood(formData);
  // Handle success
} catch (error) {
  console.error('Error:', error);
  alert('Failed to process request');
}
```

### Loading States

Show spinners during API calls:
```javascript
const [loading, setLoading] = useState(false);

const handleSubmit = async () => {
  setLoading(true);
  try {
    await apiCall();
  } finally {
    setLoading(false);
  }
};
```

## 🎯 User Flows

### First-Time User

1. Land on HomePage
2. Click "Analyze Food"
3. Upload first meal
4. View prediction
5. Log meal
6. Redirected to Dashboard
7. See "Log 3 meals" message
8. Repeat until 3 meals logged
9. View dietary analysis
10. Complete recommended modules

### Returning User

1. Land on HomePage (see points badge)
2. Navigate to Dashboard
3. View patterns and recommendations
4. Click learning module
5. Complete quiz
6. Earn points
7. Continue learning

## 🧪 Testing

### Manual Testing Checklist

**HomePage:**
- [ ] Hero section displays correctly
- [ ] Action cards navigate properly
- [ ] Points badge shows correct value
- [ ] "How It Works" section visible

**AnalyzePage:**
- [ ] File upload works
- [ ] Image preview displays
- [ ] Analyze button triggers prediction
- [ ] Results show nutrition data
- [ ] Meal type selection works
- [ ] Log meal succeeds
- [ ] Redirect to dashboard

**DashboardPage:**
- [ ] Stats cards show correct data
- [ ] Nutrition summary calculates correctly
- [ ] Recent meals list displays
- [ ] "Log 3 meals" message shows when needed
- [ ] Patterns display after 3+ meals
- [ ] Module recommendations appear
- [ ] Module cards navigate to learning page

**LearningPage:**
- [ ] Module content displays
- [ ] Quiz questions render
- [ ] Answer selection works
- [ ] Submit calculates score
- [ ] Feedback shows correct/incorrect
- [ ] Points awarded for passing
- [ ] Retry works for failed attempts

## 🚀 Deployment

### Environment Variables

Create `.env` file:
```env
VITE_API_BASE_URL=https://api.nutrilearn.ai
```

### Build

```bash
npm run build
```

Output in `dist/` directory.

### Deploy

Compatible with:
- Vercel
- Netlify
- GitHub Pages
- AWS S3 + CloudFront
- Any static hosting

## 📝 Code Conventions

### Component Structure

```javascript
import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Icon } from 'lucide-react';
import { apiFunction } from '../utils/api';

function ComponentName() {
  // State
  const [data, setData] = useState(null);
  
  // Hooks
  const navigate = useNavigate();
  
  // Effects
  useEffect(() => {
    // Fetch data
  }, []);
  
  // Handlers
  const handleAction = async () => {
    // Handle action
  };
  
  // Render
  return (
    <div>
      {/* JSX */}
    </div>
  );
}

export default ComponentName;
```

### Naming Conventions

- Components: PascalCase (`HomePage.jsx`)
- Functions: camelCase (`getUserId()`)
- Constants: UPPER_SNAKE_CASE (`API_BASE_URL`)
- CSS classes: kebab-case (Tailwind utilities)

### Best Practices

- Use functional components
- Prefer hooks over class components
- Use async/await over promises
- Handle errors with try-catch
- Show loading states
- Provide user feedback
- Keep components focused
- Extract reusable logic

## 🐛 Common Issues

### CORS Errors

Ensure backend CORS is configured:
```python
allow_origins=["http://localhost:5173"]
```

### API Connection Failed

Check:
1. Backend is running on port 8000
2. VITE_API_BASE_URL is correct
3. Network tab in DevTools for errors

### Images Not Uploading

Verify:
1. File input accepts images
2. FormData is created correctly
3. Content-Type is multipart/form-data

### State Not Updating

Remember:
1. State updates are asynchronous
2. Use functional updates for dependent state
3. Check useEffect dependencies

## 📚 Resources

- [React Documentation](https://react.dev/)
- [React Router](https://reactrouter.com/)
- [Tailwind CSS](https://tailwindcss.com/)
- [Lucide Icons](https://lucide.dev/)
- [Vite](https://vitejs.dev/)
- [Axios](https://axios-http.com/)

## 🎓 Next Steps

1. Add user authentication
2. Implement offline support
3. Add push notifications
4. Create mobile app (React Native)
5. Add social features
6. Implement meal planning
7. Add barcode scanning
8. Create recipe suggestions

---

**Built with ❤️ for learning MLOps and full-stack AI development**
