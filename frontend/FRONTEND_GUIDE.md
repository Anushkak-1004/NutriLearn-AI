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

## 🔐 Authentication

### Overview

The frontend implements JWT-based authentication that integrates with the backend API. Users can sign up, log in, log out, and access protected routes. Authentication state is managed using localStorage and Axios interceptors.

### Authentication Flow

**Signup Flow:**
```
User fills form → Validate input → POST /api/v1/auth/signup →
Backend creates user → Returns JWT → Store in localStorage →
Update Axios headers → Redirect to /dashboard
```

**Login Flow:**
```
User fills form → POST /api/v1/auth/login →
Backend validates credentials → Returns JWT → Store in localStorage →
Update Axios headers → Redirect to /dashboard
```

**Logout Flow:**
```
User clicks logout → Remove token from localStorage →
Clear Axios headers → Redirect to home page
```

**Protected Route Access:**
```
User navigates to protected route → Check isAuthenticated() →
If token exists → Render page → API calls include Bearer token →
If no token → Redirect to /login
```

### Authentication Components

#### LoginPage (`/login`)

**Features:**
- Email and password input fields
- Form validation (required fields)
- Loading state during authentication
- Error message display
- Link to signup page

**User Flow:**
1. Enter email and password
2. Click "Login" button
3. View loading state
4. On success: Redirect to dashboard
5. On error: Display error message

#### SignupPage (`/signup`)

**Features:**
- Email, password, and confirm password fields
- Client-side validation:
  - Password minimum 8 characters
  - Passwords must match
- Loading state during registration
- Error message display
- Link to login page

**User Flow:**
1. Enter email, password, and confirm password
2. Client validates password length and match
3. Click "Sign Up" button
4. View loading state
5. On success: Redirect to dashboard
6. On error: Display error message

#### ProtectedRoute Component

**Purpose:** Wrapper component that protects routes requiring authentication

**Usage:**
```javascript
<Route
  path="/dashboard"
  element={
    <ProtectedRoute>
      <DashboardPage />
    </ProtectedRoute>
  }
/>
```

**Behavior:**
- Checks if user is authenticated using `isAuthenticated()`
- If authenticated: Renders child components
- If not authenticated: Redirects to `/login` with `replace` prop

### Authentication Utilities

#### auth.js

Centralized authentication functions:

```javascript
// Register new user
await signup(email, password);

// Authenticate user
await login(email, password);

// Log out user
logout();

// Get stored token
const token = getToken();

// Check authentication status
const isAuth = isAuthenticated();
```

**Key Functions:**

- `signup(email, password)` - Creates account, stores token, updates headers
- `login(email, password)` - Authenticates user, stores token, updates headers
- `logout()` - Removes token, clears headers
- `getToken()` - Retrieves JWT from localStorage
- `isAuthenticated()` - Returns true if valid token exists

#### API Integration

**Token Management:**

The API client automatically handles authentication tokens:

```javascript
// Set or clear token in Axios headers
setAuthToken(token);

// Initialize auth on app load
initializeAuth();
```

**Request Interceptor:**
- Automatically includes JWT token in Authorization header
- Format: `Authorization: Bearer <token>`
- Applied to all API requests

**Response Interceptor:**
- Catches 401 Unauthorized responses
- Automatically removes invalid/expired tokens
- Redirects to login page
- Prevents manual error handling in components

### Protected Routes

The following routes require authentication:
- `/dashboard` - User dashboard
- `/analyze` - Food analysis
- `/learning/:moduleId` - Learning modules
- `/mlops` - MLOps dashboard

Public routes (no authentication required):
- `/` - Home page
- `/login` - Login page
- `/signup` - Signup page

### Navigation Updates

The navigation menu dynamically updates based on authentication state:

**When Not Authenticated:**
- Shows: "Login" and "Sign Up" buttons
- Hides: Protected route links, Points badge, Logout button

**When Authenticated:**
- Shows: Protected route links (Dashboard, Analyze, Learning, MLOps)
- Shows: Points badge, Logout button
- Hides: Login and Sign Up buttons

### localStorage Structure

**Token Storage:**
```javascript
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Key:** `token`
**Value:** JWT string

### Authentication Persistence

**Page Refresh:**
- Token is retrieved from localStorage on app initialization
- Axios headers are configured with stored token
- User remains authenticated across page refreshes
- Protected routes remain accessible

**Token Expiration:**
- Backend validates token on each request
- Expired tokens return 401 Unauthorized
- Frontend automatically logs out user
- User is redirected to login page

### Error Handling

**Client-Side Validation Errors:**
- Password too short (< 8 characters)
- Passwords don't match
- Invalid email format

**API Error Responses:**
- 401 Unauthorized - Invalid credentials or expired token
- 400 Bad Request - Duplicate email during signup
- 422 Unprocessable Entity - Validation errors
- Network errors - Connection issues

**Error Display:**
All errors are displayed in a consistent error banner:
```javascript
{error && (
  <div className="bg-red-50 text-red-600 p-3 rounded border border-red-200">
    {error}
  </div>
)}
```

### Security Considerations

**Token Storage:**
- JWT tokens stored in localStorage
- Vulnerable to XSS attacks (mitigated by React's built-in XSS protection)
- Should only be used over HTTPS in production
- Tokens have reasonable expiration (7 days)

**Best Practices:**
- Never commit tokens to version control
- Use environment variables for API URLs
- Validate all inputs on client and server
- Use HTTPS in production
- Implement rate limiting on backend
- Clear tokens on logout

### Troubleshooting

**Issue: User not redirected after login**
- Check that `navigate('/dashboard')` is called after successful login
- Verify no errors in browser console
- Ensure token is stored in localStorage

**Issue: Protected routes not working**
- Verify `ProtectedRoute` component is wrapping protected routes
- Check that `isAuthenticated()` returns correct value
- Ensure token exists in localStorage

**Issue: API requests return 401**
- Check that token is included in Authorization header
- Verify token hasn't expired
- Ensure `initializeAuth()` is called on app mount
- Check Axios interceptors are configured

**Issue: Token not persisting across page refresh**
- Verify `initializeAuth()` is called in `App.jsx` useEffect
- Check that token is stored in localStorage (not sessionStorage)
- Ensure no code is clearing localStorage on mount

**Issue: Navigation menu not updating**
- Verify `isAuthenticated()` is being called in Navigation component
- Check that component re-renders when auth state changes
- Ensure logout function is properly clearing token

### Testing Authentication

See `AUTHENTICATION_TESTING_GUIDE.md` for comprehensive manual testing instructions.

**Quick Test:**
1. Sign up with new account
2. Verify redirect to dashboard
3. Refresh page - should stay logged in
4. Log out - should redirect to home
5. Try accessing `/dashboard` - should redirect to login
6. Log in - should redirect to dashboard

## 🎓 Next Steps

1. ~~Add user authentication~~ ✅ Completed
2. Implement offline support
3. Add push notifications
4. Create mobile app (React Native)
5. Add social features
6. Implement meal planning
7. Add barcode scanning
8. Create recipe suggestions
9. Add password reset functionality
10. Implement email verification
11. Add two-factor authentication
12. Create user profile settings

---

**Built with ❤️ for learning MLOps and full-stack AI development**
