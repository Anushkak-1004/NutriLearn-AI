/**
 * Authentication Utility Module
 * Handles user authentication operations including signup, login, logout,
 * and token management for the NutriLearn AI application.
 */

import api, { setAuthToken } from './api';

/**
 * Register a new user account
 * @param {string} email - User's email address
 * @param {string} password - User's password
 * @returns {Promise<Object>} Response data with access_token
 * @throws {Error} If signup fails (validation, duplicate email, network)
 */
export async function signup(email, password) {
  const response = await api.post('/api/v1/auth/signup', { email, password });
  const { access_token } = response.data;
  localStorage.setItem('token', access_token);
  setAuthToken(access_token);
  return response.data;
}

/**
 * Authenticate user and obtain access token
 * @param {string} email - User's email address
 * @param {string} password - User's password
 * @returns {Promise<Object>} Response data with access_token
 * @throws {Error} If login fails (invalid credentials, network)
 */
export async function login(email, password) {
  const response = await api.post('/api/v1/auth/login', { email, password });
  const { access_token } = response.data;
  localStorage.setItem('token', access_token);
  setAuthToken(access_token);
  return response.data;
}

/**
 * Log out current user and clear authentication state
 */
export function logout() {
  localStorage.removeItem('token');
  setAuthToken(null);
}

/**
 * Retrieve stored JWT token from localStorage
 * @returns {string|null} JWT token or null if not found
 */
export function getToken() {
  return localStorage.getItem('token');
}

/**
 * Check if user is currently authenticated
 * @returns {boolean} True if valid token exists, false otherwise
 */
export function isAuthenticated() {
  const token = getToken();
  return token !== null && token !== undefined && token !== '';
}
