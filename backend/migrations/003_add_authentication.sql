-- NutriLearn AI Database Schema Update
-- Migration to add JWT authentication support
-- Version: 003
-- Created: 2025-12-17
-- Requirements: 8.1, 8.2, 8.3

-- ============================================================================
-- ADD AUTHENTICATION COLUMNS TO USERS TABLE
-- ============================================================================

-- Add email column (unique, not null)
-- This will store the user's email address for authentication
ALTER TABLE users 
ADD COLUMN IF NOT EXISTS email VARCHAR(255) UNIQUE;

-- Add password_hash column (not null)
-- This will store the bcrypt-hashed password
ALTER TABLE users 
ADD COLUMN IF NOT EXISTS password_hash TEXT;

-- Note: created_at column already exists in the users table from migration 001
-- If it doesn't exist for some reason, we'll add it here
ALTER TABLE users 
ADD COLUMN IF NOT EXISTS created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW();

-- ============================================================================
-- CREATE INDEX FOR FAST EMAIL LOOKUPS
-- ============================================================================

-- Create index on email column for fast authentication queries
-- This is critical for login performance
CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);

-- ============================================================================
-- ADD COLUMN COMMENTS
-- ============================================================================

COMMENT ON COLUMN users.email IS 'User email address for authentication (unique)';
COMMENT ON COLUMN users.password_hash IS 'Bcrypt-hashed password for secure authentication';

-- ============================================================================
-- UPDATE ROW LEVEL SECURITY POLICIES
-- ============================================================================

-- Drop existing anonymous policies if they exist (for production security)
-- Uncomment these in production to remove anonymous access
-- DROP POLICY IF EXISTS "Allow anonymous read users" ON users;
-- DROP POLICY IF EXISTS "Allow anonymous insert users" ON users;
-- DROP POLICY IF EXISTS "Allow anonymous update users" ON users;

-- Add policy for user signup (allow insert with email and password)
-- This allows new users to create accounts
CREATE POLICY IF NOT EXISTS "Allow user signup" ON users
    FOR INSERT
    WITH CHECK (email IS NOT NULL AND password_hash IS NOT NULL);

-- ============================================================================
-- VALIDATION CONSTRAINTS
-- ============================================================================

-- Add constraint to ensure email is not empty when provided
ALTER TABLE users 
ADD CONSTRAINT IF NOT EXISTS users_email_not_empty 
CHECK (email IS NULL OR LENGTH(email) > 0);

-- Add constraint to ensure password_hash is not empty when provided
ALTER TABLE users 
ADD CONSTRAINT IF NOT EXISTS users_password_hash_not_empty 
CHECK (password_hash IS NULL OR LENGTH(password_hash) > 0);

-- ============================================================================
-- COMPLETION MESSAGE
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE '✅ Authentication columns added to users table successfully!';
    RAISE NOTICE '📧 Added: email column (VARCHAR(255), UNIQUE)';
    RAISE NOTICE '🔒 Added: password_hash column (TEXT)';
    RAISE NOTICE '📅 Verified: created_at column exists';
    RAISE NOTICE '🔍 Created: Index on email column for fast lookups';
    RAISE NOTICE '';
    RAISE NOTICE '📝 Next steps:';
    RAISE NOTICE '   1. Update backend/app/auth.py with JWT token generation';
    RAISE NOTICE '   2. Create authentication routes in backend/app/api/auth_routes.py';
    RAISE NOTICE '   3. Add authentication dependencies for protected endpoints';
    RAISE NOTICE '   4. Test signup and login flows';
    RAISE NOTICE '';
    RAISE NOTICE '⚠️  Note: Existing users will have NULL email and password_hash';
    RAISE NOTICE '   These users will need to be migrated or re-registered';
END $$;
