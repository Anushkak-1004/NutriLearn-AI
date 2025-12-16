/**
 * Local Storage Utility for NutriLearn AI
 * Manages user identification and persistent data
 */

const USER_ID_KEY = 'nutrilearn_userId';

/**
 * Generate a UUID v4
 * @returns {string} UUID string
 */
function generateUUID() {
  // Use crypto.randomUUID if available (modern browsers)
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  
  // Fallback for older browsers
  return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
    const r = Math.random() * 16 | 0;
    const v = c === 'x' ? r : (r & 0x3 | 0x8);
    return v.toString(16);
  });
}

/**
 * Get or create user ID
 * @returns {string} User ID
 */
export function getUserId() {
  let userId = localStorage.getItem(USER_ID_KEY);
  
  if (!userId) {
    userId = generateUUID();
    localStorage.setItem(USER_ID_KEY, userId);
    console.log('New user ID generated:', userId);
  }
  
  return userId;
}

/**
 * Clear user data (for testing/logout)
 */
export function clearUserData() {
  localStorage.removeItem(USER_ID_KEY);
  console.log('User data cleared');
}

/**
 * Get user points from localStorage
 * @returns {number} User points
 */
export function getUserPoints() {
  const points = localStorage.getItem('nutrilearn_points');
  return points ? parseInt(points, 10) : 0;
}

/**
 * Set user points in localStorage
 * @param {number} points - Points to store
 */
export function setUserPoints(points) {
  localStorage.setItem('nutrilearn_points', points.toString());
}
