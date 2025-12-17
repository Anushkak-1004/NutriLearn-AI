# Frontend Authentication - Manual Testing Guide

This guide provides step-by-step instructions for manually testing the authentication functionality.

## Prerequisites

1. Backend server running on `http://localhost:8000`
2. Frontend development server running on `http://localhost:5173` (or configured port)
3. Database properly configured and migrations applied

## Test Scenarios

### 1. Test Signup with Valid Credentials

**Steps:**
1. Navigate to `http://localhost:5173/signup`
2. Enter a valid email (e.g., `test@example.com`)
3. Enter a password with at least 8 characters (e.g., `TestPass123!`)
4. Re-enter the same password in the confirm password field
5. Click "Sign Up" button

**Expected Result:**
- Loading state shows "Creating account..."
- After successful signup, user is redirected to `/dashboard`
- JWT token is stored in localStorage (check browser DevTools > Application > Local Storage)
- Navigation menu shows "Logout" button and protected links
- No error messages displayed

---

### 2. Test Signup with Duplicate Email

**Steps:**
1. Navigate to `http://localhost:5173/signup`
2. Enter an email that's already registered
3. Enter a valid password
4. Re-enter the same password
5. Click "Sign Up" button

**Expected Result:**
- Error message displays: "Email is already registered" (or similar)
- User remains on signup page
- No redirect occurs
- No token stored in localStorage

---

### 3. Test Signup with Mismatched Passwords

**Steps:**
1. Navigate to `http://localhost:5173/signup`
2. Enter a valid email
3. Enter a password (e.g., `Password123!`)
4. Enter a different password in confirm field (e.g., `DifferentPass123!`)
5. Click "Sign Up" button

**Expected Result:**
- Error message displays: "Passwords do not match"
- Form submission is prevented
- No API call is made
- User remains on signup page

---

### 4. Test Signup with Short Password

**Steps:**
1. Navigate to `http://localhost:5173/signup`
2. Enter a valid email
3. Enter a password with less than 8 characters (e.g., `Pass123`)
4. Enter the same password in confirm field
5. Click "Sign Up" button

**Expected Result:**
- Error message displays: "Password must be at least 8 characters long"
- Form submission is prevented
- No API call is made
- User remains on signup page

---

### 5. Test Login with Valid Credentials

**Steps:**
1. Navigate to `http://localhost:5173/login`
2. Enter a registered email
3. Enter the correct password
4. Click "Login" button

**Expected Result:**
- Loading state shows "Logging in..."
- After successful login, user is redirected to `/dashboard`
- JWT token is stored in localStorage
- Navigation menu shows "Logout" button and protected links
- No error messages displayed

---

### 6. Test Login with Invalid Credentials

**Steps:**
1. Navigate to `http://localhost:5173/login`
2. Enter a valid email format but wrong credentials
3. Enter an incorrect password
4. Click "Login" button

**Expected Result:**
- Error message displays: "Invalid email or password" (or similar)
- User remains on login page
- No redirect occurs
- No token stored in localStorage

---

### 7. Test Logout Functionality

**Steps:**
1. Ensure you're logged in (token exists in localStorage)
2. Navigate to any page
3. Click the "Logout" button in the navigation menu

**Expected Result:**
- JWT token is removed from localStorage
- User is redirected to home page (`/`)
- Navigation menu shows "Login" and "Sign Up" buttons
- Protected route links are hidden from navigation

---

### 8. Test Protected Route Access Without Authentication

**Steps:**
1. Ensure you're logged out (no token in localStorage)
2. Manually navigate to a protected route (e.g., `http://localhost:5173/dashboard`)

**Expected Result:**
- User is immediately redirected to `/login`
- Protected page content is not rendered
- URL changes to `/login`

---

### 9. Test Protected Route Access With Authentication

**Steps:**
1. Log in successfully
2. Navigate to protected routes:
   - `/dashboard`
   - `/analyze`
   - `/learning/:moduleId`
   - `/mlops`

**Expected Result:**
- All protected pages render correctly
- No redirects occur
- Navigation menu shows all protected links
- User can access all features

---

### 10. Test Navigation Menu Updates

**Steps:**
1. Start logged out
2. Observe navigation menu (should show "Login" and "Sign Up")
3. Log in
4. Observe navigation menu (should show "Logout" and protected links)
5. Log out
6. Observe navigation menu (should revert to "Login" and "Sign Up")

**Expected Result:**
- Navigation menu dynamically updates based on authentication state
- Protected links (Dashboard, Analyze, Learning, MLOps) only visible when authenticated
- Points badge only visible when authenticated
- Login/Signup buttons only visible when not authenticated
- Logout button only visible when authenticated

---

### 11. Test Page Refresh Maintains Authentication

**Steps:**
1. Log in successfully
2. Navigate to `/dashboard`
3. Refresh the page (F5 or Ctrl+R)

**Expected Result:**
- User remains logged in
- Dashboard page renders correctly
- No redirect to login page
- Token still exists in localStorage
- Navigation menu still shows authenticated state

---

### 12. Test Token Expiration Handling

**Steps:**
1. Log in successfully
2. Manually expire or delete the token from localStorage, OR wait for token to expire naturally
3. Make an API request (e.g., navigate to dashboard which fetches data)

**Expected Result:**
- Backend returns 401 Unauthorized
- Axios interceptor catches the 401 response
- Token is removed from localStorage
- User is redirected to `/login`
- Error message may be displayed

---

### 13. Test API Requests Include Token

**Steps:**
1. Log in successfully
2. Open browser DevTools > Network tab
3. Navigate to a page that makes API requests (e.g., `/dashboard`)
4. Inspect the API request headers

**Expected Result:**
- All API requests include `Authorization: Bearer <token>` header
- Token matches the one stored in localStorage
- Backend accepts the requests and returns data

---

### 14. Test Network Error Handling

**Steps:**
1. Stop the backend server
2. Navigate to `/login`
3. Enter valid credentials
4. Click "Login" button

**Expected Result:**
- Error message displays: "Connection error. Please check your internet connection and try again." (or similar)
- User remains on login page
- No redirect occurs
- Loading state ends

---

### 15. Test Form Validation

**Steps:**
1. Navigate to `/login` or `/signup`
2. Try to submit the form with empty fields
3. Try to submit with invalid email format

**Expected Result:**
- Browser's built-in validation prevents submission
- Appropriate validation messages appear
- No API calls are made

---

## Verification Checklist

After completing all test scenarios, verify:

- [ ] Signup with valid credentials creates account and logs in
- [ ] Signup with duplicate email shows error
- [ ] Signup with mismatched passwords shows error
- [ ] Signup with short password shows error
- [ ] Login with valid credentials logs in successfully
- [ ] Login with invalid credentials shows error
- [ ] Logout clears session and redirects
- [ ] Protected routes redirect to login when not authenticated
- [ ] Protected routes accessible when authenticated
- [ ] Navigation menu updates based on auth state
- [ ] Page refresh maintains authentication
- [ ] Token expiration triggers logout
- [ ] API requests include Authorization header
- [ ] Network errors show appropriate messages
- [ ] Form validation works correctly

---

## Debugging Tips

### Check localStorage
```javascript
// In browser console
localStorage.getItem('token')
```

### Check Axios headers
```javascript
// In browser console
import api from './utils/api';
console.log(api.defaults.headers.common);
```

### Clear authentication state
```javascript
// In browser console
localStorage.removeItem('token');
window.location.reload();
```

### Check if authenticated
```javascript
// In browser console
import { isAuthenticated } from './utils/auth';
console.log(isAuthenticated());
```

---

## Common Issues and Solutions

### Issue: Redirects not working
**Solution:** Check that `react-router-dom` is properly configured and `useNavigate` is being used correctly.

### Issue: Token not being sent with requests
**Solution:** Verify that `initializeAuth()` is called in `App.jsx` on mount, and check Axios interceptors.

### Issue: 401 errors not triggering logout
**Solution:** Check the Axios response interceptor in `api.js` is properly configured.

### Issue: Navigation menu not updating
**Solution:** Ensure `isAuthenticated()` is being called and the component re-renders when auth state changes.

### Issue: CORS errors
**Solution:** Verify backend CORS configuration allows requests from frontend origin.

---

## Next Steps

After completing manual testing:
1. Document any bugs or issues found
2. Fix any failing test scenarios
3. Update this guide with any new test cases
4. Consider automating these tests with Jest and React Testing Library
