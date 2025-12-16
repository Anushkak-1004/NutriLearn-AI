# Requirements Document

## Introduction

This document specifies the requirements for implementing JWT (JSON Web Token) based authentication in the NutriLearn AI backend API. The authentication system will secure user-specific endpoints, manage user registration and login, and ensure that only authenticated users can access protected resources such as meal logs and personal statistics.

## Glossary

- **JWT (JSON Web Token)**: A compact, URL-safe token format used for securely transmitting information between parties as a JSON object
- **Authentication System**: The backend service responsible for user registration, login, and token management
- **Protected Endpoint**: An API endpoint that requires a valid JWT token for access
- **Public Endpoint**: An API endpoint accessible without authentication (e.g., /predict, /health)
- **Access Token**: A JWT token issued upon successful authentication, used to access protected resources
- **Password Hash**: A one-way cryptographic hash of a user's password using bcrypt algorithm
- **Token Payload**: The data contained within a JWT, including user_id and expiration time
- **Bearer Token**: An HTTP authentication scheme where the token is sent in the Authorization header

## Requirements

### Requirement 1

**User Story:** As a new user, I want to create an account with my email and password, so that I can securely access personalized features of the application.

#### Acceptance Criteria

1. WHEN a user submits valid email and password credentials to the signup endpoint, THEN the Authentication System SHALL create a new user record with a hashed password and return an Access Token
2. WHEN a user attempts to signup with an email that already exists, THEN the Authentication System SHALL reject the request and return an error indicating the email is already registered
3. WHEN a user submits a password during signup, THEN the Authentication System SHALL hash the password using bcrypt before storing it in the database
4. WHEN a user submits an invalid email format during signup, THEN the Authentication System SHALL reject the request and return a validation error
5. WHEN a user submits a password shorter than 8 characters during signup, THEN the Authentication System SHALL reject the request and return a validation error

### Requirement 2

**User Story:** As a registered user, I want to log in with my email and password, so that I can access my personalized meal logs and statistics.

#### Acceptance Criteria

1. WHEN a user submits correct email and password credentials to the login endpoint, THEN the Authentication System SHALL verify the credentials and return a valid Access Token
2. WHEN a user submits an incorrect password, THEN the Authentication System SHALL reject the request and return an authentication error without revealing whether the email exists
3. WHEN a user submits an email that does not exist, THEN the Authentication System SHALL reject the request and return an authentication error without revealing that the email doesn't exist
4. WHEN a user successfully logs in, THEN the Authentication System SHALL generate a JWT token with a 7-day expiration period
5. WHEN generating an Access Token, THEN the Authentication System SHALL include the user_id in the Token Payload

### Requirement 3

**User Story:** As an authenticated user, I want my access token to be validated on protected endpoints, so that only I can access my personal data.

#### Acceptance Criteria

1. WHEN a request to a Protected Endpoint includes a valid Access Token in the Authorization header, THEN the Authentication System SHALL extract the user_id from the Token Payload and allow the request to proceed
2. WHEN a request to a Protected Endpoint includes an expired Access Token, THEN the Authentication System SHALL reject the request and return an authentication error
3. WHEN a request to a Protected Endpoint includes a malformed or invalid Access Token, THEN the Authentication System SHALL reject the request and return an authentication error
4. WHEN a request to a Protected Endpoint does not include an Access Token, THEN the Authentication System SHALL reject the request and return an authentication error
5. WHEN a request to a Protected Endpoint includes a valid token, THEN the Authentication System SHALL verify the token signature using the secret key before granting access

### Requirement 4

**User Story:** As an authenticated user, I want to retrieve my profile information, so that I can verify my account details.

#### Acceptance Criteria

1. WHEN an authenticated user requests their profile information, THEN the Authentication System SHALL return the user's email and user_id from the database
2. WHEN an authenticated user requests their profile information, THEN the Authentication System SHALL NOT return the password hash in the response
3. WHEN a request to retrieve profile information includes a valid Access Token, THEN the Authentication System SHALL extract the user_id from the token and fetch the corresponding user record

### Requirement 5

**User Story:** As a system administrator, I want user passwords to be securely stored, so that user credentials are protected even if the database is compromised.

#### Acceptance Criteria

1. WHEN the Authentication System stores a password, THEN it SHALL use the bcrypt hashing algorithm with appropriate salt rounds
2. WHEN the Authentication System verifies a password, THEN it SHALL compare the plain text password against the stored hash using bcrypt's verification function
3. WHEN the Authentication System generates JWT tokens, THEN it SHALL use a secret key stored in environment variables and not hardcoded in the source code
4. WHEN the Authentication System signs a JWT token, THEN it SHALL use the HS256 algorithm for token signature

### Requirement 6

**User Story:** As a developer, I want to easily protect API endpoints with authentication, so that I can secure user-specific resources without duplicating code.

#### Acceptance Criteria

1. WHEN a developer adds authentication to an endpoint, THEN they SHALL use a dependency injection pattern to require authentication
2. WHEN a Protected Endpoint is accessed, THEN the authentication dependency SHALL automatically extract and validate the Access Token
3. WHEN a Protected Endpoint is accessed with a valid token, THEN the authentication dependency SHALL provide the authenticated user_id to the endpoint handler
4. WHEN the Authentication System validates a token, THEN it SHALL handle all token-related errors (expired, invalid, malformed) and return appropriate HTTP status codes

### Requirement 7

**User Story:** As a user, I want certain endpoints to remain publicly accessible, so that I can use features like food prediction without requiring authentication.

#### Acceptance Criteria

1. WHEN a request is made to the /api/v1/predict endpoint, THEN the Authentication System SHALL allow access without requiring an Access Token
2. WHEN a request is made to the /api/v1/health endpoint, THEN the Authentication System SHALL allow access without requiring an Access Token
3. WHEN a request is made to the /api/v1/auth/signup endpoint, THEN the Authentication System SHALL allow access without requiring an Access Token
4. WHEN a request is made to the /api/v1/auth/login endpoint, THEN the Authentication System SHALL allow access without requiring an Access Token

### Requirement 8

**User Story:** As a developer, I want the database schema to support authentication, so that user credentials and metadata can be properly stored.

#### Acceptance Criteria

1. WHEN the database schema is created, THEN the users table SHALL include an email column with a unique constraint
2. WHEN the database schema is created, THEN the users table SHALL include a password_hash column to store hashed passwords
3. WHEN the database schema is created, THEN the users table SHALL include a created_at timestamp column to track account creation time
4. WHEN a new user is created, THEN the Authentication System SHALL store the email, password_hash, and created_at timestamp in the users table

### Requirement 9

**User Story:** As a user, I want my meal logs to be associated with my account, so that only I can view and manage my personal meal history.

#### Acceptance Criteria

1. WHEN a user logs a meal through the /api/v1/meals/log endpoint, THEN the Authentication System SHALL extract the user_id from the Access Token rather than accepting it in the request body
2. WHEN a user requests their statistics through the /api/v1/users/{user_id}/stats endpoint, THEN the Authentication System SHALL verify that the user_id in the URL matches the user_id in the Access Token
3. WHEN a user attempts to access another user's statistics, THEN the Authentication System SHALL reject the request and return an authorization error
4. WHEN a Protected Endpoint receives a user_id parameter, THEN it SHALL validate that the authenticated user has permission to access that user's data
