-- NutriLearn AI Database Schema Update
-- Migration to update existing schema to use TEXT user IDs
-- Version: 002
-- Created: 2025-12-17

-- ============================================================================
-- DROP EXISTING OBJECTS (if they exist)
-- ============================================================================

-- Drop triggers first
DROP TRIGGER IF EXISTS update_users_updated_at ON users;
DROP TRIGGER IF EXISTS ensure_user_before_meal ON meals;
DROP TRIGGER IF EXISTS increment_meal_count ON meals;
DROP TRIGGER IF EXISTS update_points_on_module_completion ON user_modules;

-- Drop views
DROP VIEW IF EXISTS user_stats_view;
DROP VIEW IF EXISTS recent_meals_view;

-- Drop policies
DROP POLICY IF EXISTS "Users can view own data" ON users;
DROP POLICY IF EXISTS "Users can update own data" ON users;
DROP POLICY IF EXISTS "Users can view own meals" ON meals;
DROP POLICY IF EXISTS "Users can insert own meals" ON meals;
DROP POLICY IF EXISTS "Users can view own modules" ON user_modules;
DROP POLICY IF EXISTS "Users can complete modules" ON user_modules;
DROP POLICY IF EXISTS "Allow anonymous read users" ON users;
DROP POLICY IF EXISTS "Allow anonymous insert users" ON users;
DROP POLICY IF EXISTS "Allow anonymous update users" ON users;
DROP POLICY IF EXISTS "Allow anonymous read meals" ON meals;
DROP POLICY IF EXISTS "Allow anonymous insert meals" ON meals;
DROP POLICY IF EXISTS "Allow anonymous read modules" ON user_modules;
DROP POLICY IF EXISTS "Allow anonymous insert modules" ON user_modules;

-- Drop tables (cascade to remove dependencies)
DROP TABLE IF EXISTS user_modules CASCADE;
DROP TABLE IF EXISTS meals CASCADE;
DROP TABLE IF EXISTS users CASCADE;

-- Drop functions
DROP FUNCTION IF EXISTS update_updated_at_column();
DROP FUNCTION IF EXISTS ensure_user_exists();
DROP FUNCTION IF EXISTS update_user_meal_count();
DROP FUNCTION IF EXISTS update_user_points();

-- ============================================================================
-- CREATE FRESH SCHEMA
-- ============================================================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================================================
-- USERS TABLE
-- ============================================================================
CREATE TABLE users (
    id TEXT PRIMARY KEY,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    total_meals INTEGER DEFAULT 0 CHECK (total_meals >= 0),
    total_points INTEGER DEFAULT 0 CHECK (total_points >= 0),
    current_streak INTEGER DEFAULT 0 CHECK (current_streak >= 0),
    
    CONSTRAINT users_id_check CHECK (id IS NOT NULL AND LENGTH(id) > 0)
);

CREATE INDEX idx_users_created_at ON users(created_at);

COMMENT ON TABLE users IS 'User profiles and statistics';
COMMENT ON COLUMN users.id IS 'Unique user identifier (TEXT for flexibility)';
COMMENT ON COLUMN users.total_meals IS 'Total number of meals logged by user';
COMMENT ON COLUMN users.total_points IS 'Total learning points earned';
COMMENT ON COLUMN users.current_streak IS 'Current daily logging streak in days';

-- ============================================================================
-- MEALS TABLE
-- ============================================================================
CREATE TABLE meals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT NOT NULL,
    food_name TEXT NOT NULL CHECK (LENGTH(food_name) > 0),
    
    -- Nutrition information
    calories INTEGER NOT NULL CHECK (calories >= 0 AND calories <= 5000),
    protein DECIMAL(10, 2) NOT NULL CHECK (protein >= 0),
    carbs DECIMAL(10, 2) NOT NULL CHECK (carbs >= 0),
    fat DECIMAL(10, 2) NOT NULL CHECK (fat >= 0),
    fiber DECIMAL(10, 2) NOT NULL CHECK (fiber >= 0),
    
    -- Meal metadata
    meal_type TEXT NOT NULL CHECK (meal_type IN ('breakfast', 'lunch', 'dinner', 'snack')),
    category TEXT,
    cuisine TEXT,
    
    -- Timestamps
    logged_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT meals_id_check CHECK (id IS NOT NULL),
    CONSTRAINT meals_user_id_check CHECK (user_id IS NOT NULL)
);

CREATE INDEX idx_meals_user_id ON meals(user_id);
CREATE INDEX idx_meals_logged_at ON meals(logged_at DESC);
CREATE INDEX idx_meals_user_logged ON meals(user_id, logged_at DESC);
CREATE INDEX idx_meals_meal_type ON meals(meal_type);

COMMENT ON TABLE meals IS 'User meal logs with nutrition information';
COMMENT ON COLUMN meals.user_id IS 'Reference to user who logged the meal';
COMMENT ON COLUMN meals.food_name IS 'Name of the food item';
COMMENT ON COLUMN meals.calories IS 'Total calories in kcal';
COMMENT ON COLUMN meals.meal_type IS 'Type of meal: breakfast, lunch, dinner, or snack';
COMMENT ON COLUMN meals.logged_at IS 'When the meal was consumed';

-- ============================================================================
-- USER_MODULES TABLE
-- ============================================================================
CREATE TABLE user_modules (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id TEXT NOT NULL,
    module_id TEXT NOT NULL CHECK (LENGTH(module_id) > 0),
    completed_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    quiz_score INTEGER CHECK (quiz_score >= 0 AND quiz_score <= 100),
    points_earned INTEGER DEFAULT 0 CHECK (points_earned >= 0),
    
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    UNIQUE(user_id, module_id),
    
    CONSTRAINT user_modules_id_check CHECK (id IS NOT NULL),
    CONSTRAINT user_modules_user_id_check CHECK (user_id IS NOT NULL)
);

CREATE INDEX idx_user_modules_user_id ON user_modules(user_id);
CREATE INDEX idx_user_modules_module_id ON user_modules(module_id);
CREATE INDEX idx_user_modules_completed_at ON user_modules(completed_at DESC);
CREATE INDEX idx_user_modules_user_module ON user_modules(user_id, module_id);

COMMENT ON TABLE user_modules IS 'Tracks completed learning modules for each user';
COMMENT ON COLUMN user_modules.user_id IS 'Reference to user who completed the module';
COMMENT ON COLUMN user_modules.module_id IS 'Identifier of the completed module';
COMMENT ON COLUMN user_modules.quiz_score IS 'Score achieved on module quiz (0-100)';
COMMENT ON COLUMN user_modules.points_earned IS 'Points awarded for completing the module';

-- ============================================================================
-- FUNCTIONS AND TRIGGERS
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for users table
CREATE TRIGGER update_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Function to automatically create user stats when first meal is logged
CREATE OR REPLACE FUNCTION ensure_user_exists()
RETURNS TRIGGER AS $$
BEGIN
    INSERT INTO users (id, created_at, total_meals, total_points, current_streak)
    VALUES (NEW.user_id, NOW(), 0, 0, 0)
    ON CONFLICT (id) DO NOTHING;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to ensure user exists before inserting meal
CREATE TRIGGER ensure_user_before_meal
    BEFORE INSERT ON meals
    FOR EACH ROW
    EXECUTE FUNCTION ensure_user_exists();

-- Function to update user meal count
CREATE OR REPLACE FUNCTION update_user_meal_count()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE users
    SET total_meals = total_meals + 1,
        updated_at = NOW()
    WHERE id = NEW.user_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to increment meal count
CREATE TRIGGER increment_meal_count
    AFTER INSERT ON meals
    FOR EACH ROW
    EXECUTE FUNCTION update_user_meal_count();

-- Function to update user points when module is completed
CREATE OR REPLACE FUNCTION update_user_points()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE users
    SET total_points = total_points + NEW.points_earned,
        updated_at = NOW()
    WHERE id = NEW.user_id;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to update points
CREATE TRIGGER update_points_on_module_completion
    AFTER INSERT ON user_modules
    FOR EACH ROW
    EXECUTE FUNCTION update_user_points();

-- ============================================================================
-- VIEWS FOR COMMON QUERIES
-- ============================================================================

CREATE OR REPLACE VIEW user_stats_view AS
SELECT 
    u.id AS user_id,
    u.created_at,
    u.updated_at,
    u.total_meals,
    u.total_points,
    u.current_streak,
    COUNT(DISTINCT um.module_id) AS completed_modules_count,
    COALESCE(SUM(m.calories), 0) AS total_calories_logged,
    COALESCE(AVG(m.calories), 0) AS avg_calories_per_meal
FROM users u
LEFT JOIN meals m ON u.id = m.user_id
LEFT JOIN user_modules um ON u.id = um.user_id
GROUP BY u.id, u.created_at, u.updated_at, u.total_meals, u.total_points, u.current_streak;

COMMENT ON VIEW user_stats_view IS 'Aggregated user statistics including meals and modules';

CREATE OR REPLACE VIEW recent_meals_view AS
SELECT 
    m.id,
    m.user_id,
    m.food_name,
    m.calories,
    m.protein,
    m.carbs,
    m.fat,
    m.fiber,
    m.meal_type,
    m.category,
    m.cuisine,
    m.logged_at,
    DATE(m.logged_at) AS meal_date
FROM meals m
ORDER BY m.logged_at DESC;

COMMENT ON VIEW recent_meals_view IS 'Recent meals with date extraction for easy filtering';

-- ============================================================================
-- PERMISSIONS (Supabase RLS - Row Level Security)
-- ============================================================================

ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE meals ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_modules ENABLE ROW LEVEL SECURITY;

-- Authenticated user policies
CREATE POLICY "Users can view own data" ON users
    FOR SELECT
    USING (auth.uid()::text = id);

CREATE POLICY "Users can update own data" ON users
    FOR UPDATE
    USING (auth.uid()::text = id);

CREATE POLICY "Users can view own meals" ON meals
    FOR SELECT
    USING (auth.uid()::text = user_id);

CREATE POLICY "Users can insert own meals" ON meals
    FOR INSERT
    WITH CHECK (auth.uid()::text = user_id);

CREATE POLICY "Users can view own modules" ON user_modules
    FOR SELECT
    USING (auth.uid()::text = user_id);

CREATE POLICY "Users can complete modules" ON user_modules
    FOR INSERT
    WITH CHECK (auth.uid()::text = user_id);

-- Anonymous access policies for development (REMOVE IN PRODUCTION)
CREATE POLICY "Allow anonymous read users" ON users
    FOR SELECT
    USING (true);

CREATE POLICY "Allow anonymous insert users" ON users
    FOR INSERT
    WITH CHECK (true);

CREATE POLICY "Allow anonymous update users" ON users
    FOR UPDATE
    USING (true);

CREATE POLICY "Allow anonymous read meals" ON meals
    FOR SELECT
    USING (true);

CREATE POLICY "Allow anonymous insert meals" ON meals
    FOR INSERT
    WITH CHECK (true);

CREATE POLICY "Allow anonymous read modules" ON user_modules
    FOR SELECT
    USING (true);

CREATE POLICY "Allow anonymous insert modules" ON user_modules
    FOR INSERT
    WITH CHECK (true);

-- ============================================================================
-- COMPLETION MESSAGE
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE '✅ NutriLearn AI database schema updated successfully!';
    RAISE NOTICE '📊 Tables recreated: users, meals, user_modules';
    RAISE NOTICE '🔄 User IDs now use TEXT format (flexible for development)';
    RAISE NOTICE '🔍 Indexes created for optimal query performance';
    RAISE NOTICE '⚡ Triggers configured for automatic updates';
    RAISE NOTICE '🔒 Row Level Security enabled (anonymous access for development)';
    RAISE NOTICE '';
    RAISE NOTICE '⚠️  WARNING: All existing data has been cleared!';
    RAISE NOTICE '';
    RAISE NOTICE '📝 Next steps:';
    RAISE NOTICE '   1. Test the connection: python -c "from app.database import test_connection; test_connection()"';
    RAISE NOTICE '   2. Start the backend: uvicorn app.main:app --reload';
    RAISE NOTICE '   3. Test API endpoints with your frontend';
END $$;
