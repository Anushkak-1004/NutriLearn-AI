# Supabase Database Setup Guide

This guide will help you set up Supabase as the database for NutriLearn AI.

## 📋 Prerequisites

- Supabase account (free tier available at [supabase.com](https://supabase.com))
- Python 3.9+ with pip installed
- Backend dependencies installed (`pip install -r requirements.txt`)

---

## 🚀 Quick Start

### Step 1: Create Supabase Project

1. Go to [https://app.supabase.com](https://app.supabase.com)
2. Click "New Project"
3. Fill in project details:
   - **Name**: nutrilearn-ai (or your preferred name)
   - **Database Password**: Choose a strong password (save it!)
   - **Region**: Select closest to your users
4. Click "Create new project"
5. Wait 2-3 minutes for project to be ready

### Step 2: Get Your Credentials

1. In your Supabase project dashboard, click "Settings" (gear icon)
2. Go to "API" section
3. Copy the following:
   - **Project URL** (looks like: `https://xxxxx.supabase.co`)
   - **anon public** key (under "Project API keys")

### Step 3: Configure Backend

1. Navigate to `backend/` directory
2. Copy `.env.example` to `.env`:
   ```bash
   cp .env.example .env
   ```

3. Edit `.env` and update:
   ```env
   USE_SUPABASE=true
   SUPABASE_URL=https://your-project-id.supabase.co
   SUPABASE_KEY=your-anon-public-key-here
   ```

### Step 4: Run Database Migration

1. In Supabase dashboard, go to "SQL Editor"
2. Click "New query"
3. Copy the entire contents of `backend/migrations/001_initial_schema.sql`
4. Paste into the SQL editor
5. Click "Run" (or press Ctrl+Enter)
6. You should see: "Success. No rows returned"

### Step 5: Verify Setup

1. In Supabase dashboard, go to "Table Editor"
2. You should see three tables:
   - `users`
   - `meals`
   - `user_modules`

3. Test the connection from your backend:
   ```bash
   cd backend
   python -c "from app.database import test_connection; test_connection()"
   ```

   You should see:
   ```
   ✅ Database connection test successful
   Connected to Supabase at: https://xxxxx.supabase.co
   ```

### Step 6: Start the Backend

```bash
cd backend
uvicorn app.main:app --reload
```

Check the logs for:
```
DATABASE: Supabase PostgreSQL
URL: https://xxxxx.supabase.co
```

---

## 📊 Database Schema

### Tables

#### `users`
Stores user profiles and statistics.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key (user identifier) |
| created_at | TIMESTAMP | When user was created |
| updated_at | TIMESTAMP | Last update timestamp |
| total_meals | INTEGER | Total meals logged |
| total_points | INTEGER | Total learning points earned |
| current_streak | INTEGER | Current daily logging streak |

#### `meals`
Stores meal logs with nutrition information.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key (meal log ID) |
| user_id | UUID | Reference to user |
| food_name | TEXT | Name of the food |
| calories | INTEGER | Calories (kcal) |
| protein | DECIMAL | Protein (grams) |
| carbs | DECIMAL | Carbohydrates (grams) |
| fat | DECIMAL | Fat (grams) |
| fiber | DECIMAL | Fiber (grams) |
| meal_type | TEXT | breakfast/lunch/dinner/snack |
| category | TEXT | Food category (optional) |
| cuisine | TEXT | Cuisine type (optional) |
| logged_at | TIMESTAMP | When meal was consumed |
| created_at | TIMESTAMP | When record was created |

#### `user_modules`
Tracks completed learning modules.

| Column | Type | Description |
|--------|------|-------------|
| id | UUID | Primary key |
| user_id | UUID | Reference to user |
| module_id | TEXT | Module identifier |
| completed_at | TIMESTAMP | When module was completed |
| quiz_score | INTEGER | Quiz score (0-100) |
| points_earned | INTEGER | Points awarded |
| created_at | TIMESTAMP | When record was created |

### Indexes

- `idx_meals_user_id` - Fast user meal lookups
- `idx_meals_logged_at` - Fast date-based queries
- `idx_meals_user_logged` - Combined user + date queries
- `idx_user_modules_user_id` - Fast module completion lookups

### Triggers

- **Auto-create users**: Automatically creates user record when first meal is logged
- **Update meal count**: Increments `total_meals` when meal is logged
- **Update points**: Increments `total_points` when module is completed
- **Update timestamps**: Automatically updates `updated_at` on changes

---

## 🔒 Security (Row Level Security)

The migration includes Row Level Security (RLS) policies:

### Development Mode (Current)
- Anonymous access enabled for all operations
- **⚠️ REMOVE IN PRODUCTION**

### Production Mode (Recommended)
1. Remove anonymous policies
2. Enable Supabase Auth
3. Use authenticated user policies:
   - Users can only access their own data
   - Enforced at database level

To disable anonymous access:
```sql
-- Run in Supabase SQL Editor
DROP POLICY "Allow anonymous read users" ON users;
DROP POLICY "Allow anonymous insert users" ON users;
DROP POLICY "Allow anonymous update users" ON users;
DROP POLICY "Allow anonymous read meals" ON meals;
DROP POLICY "Allow anonymous insert meals" ON meals;
DROP POLICY "Allow anonymous read modules" ON user_modules;
DROP POLICY "Allow anonymous insert modules" ON user_modules;
```

---

## 🧪 Testing

### Test Database Connection

```bash
cd backend
python -c "from app.database import test_connection, get_database_info; test_connection(); print(get_database_info())"
```

### Test API Endpoints

1. Start the backend:
   ```bash
   uvicorn app.main:app --reload
   ```

2. Test meal logging:
   ```bash
   curl -X POST http://localhost:8000/api/v1/meals/log \
     -H "Content-Type: application/json" \
     -d '{
       "user_id": "test-user-123",
       "food_name": "Chicken Biryani",
       "nutrition": {
         "calories": 450,
         "protein": 25.0,
         "carbs": 55.0,
         "fat": 12.0,
         "fiber": 3.0
       },
       "meal_type": "lunch"
     }'
   ```

3. Check Supabase Table Editor to see the new meal

### View Data in Supabase

1. Go to Supabase dashboard
2. Click "Table Editor"
3. Select table (users, meals, or user_modules)
4. View and filter data

---

## 🔄 Fallback Behavior

The backend automatically falls back to in-memory storage if:
- Supabase credentials are not configured
- Supabase connection fails
- `USE_SUPABASE=false` in `.env`

Check logs for:
```
DATABASE: In-Memory Storage (Development Only)
Configure Supabase for production use
```

---

## 📈 Monitoring

### View Logs

In Supabase dashboard:
1. Go to "Logs" section
2. Select "Database" logs
3. View queries and errors

### Query Performance

1. Go to "Database" → "Query Performance"
2. View slow queries
3. Optimize with indexes if needed

### Storage Usage

1. Go to "Settings" → "Usage"
2. Monitor:
   - Database size
   - API requests
   - Bandwidth

---

## 🐛 Troubleshooting

### Connection Failed

**Error**: `Supabase connection test failed`

**Solutions**:
1. Check SUPABASE_URL and SUPABASE_KEY in `.env`
2. Verify project is active in Supabase dashboard
3. Check internet connection
4. Verify API key is the "anon public" key (not service_role)

### Migration Failed

**Error**: SQL errors when running migration

**Solutions**:
1. Ensure you're using PostgreSQL 14+ (Supabase default)
2. Run migration in Supabase SQL Editor (not local psql)
3. Check for syntax errors in migration file
4. Drop existing tables if re-running:
   ```sql
   DROP TABLE IF EXISTS user_modules CASCADE;
   DROP TABLE IF EXISTS meals CASCADE;
   DROP TABLE IF EXISTS users CASCADE;
   ```

### Import Error

**Error**: `No module named 'supabase'`

**Solution**:
```bash
pip install supabase==2.3.0
```

### RLS Policy Errors

**Error**: `new row violates row-level security policy`

**Solutions**:
1. Ensure anonymous policies are enabled (development)
2. Or implement proper authentication (production)
3. Check user_id matches authenticated user

---

## 🚀 Production Deployment

### Checklist

- [ ] Remove anonymous RLS policies
- [ ] Enable Supabase Auth
- [ ] Set up proper user authentication
- [ ] Configure environment variables securely
- [ ] Enable database backups
- [ ] Set up monitoring and alerts
- [ ] Review and optimize indexes
- [ ] Enable SSL/TLS for connections
- [ ] Set up rate limiting
- [ ] Configure CORS properly

### Environment Variables

For production, use secure secret management:
- AWS Secrets Manager
- Azure Key Vault
- Google Secret Manager
- Or your platform's secret management

Never commit `.env` file to version control!

---

## 📚 Additional Resources

- [Supabase Documentation](https://supabase.com/docs)
- [Supabase Python Client](https://github.com/supabase-community/supabase-py)
- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [Row Level Security Guide](https://supabase.com/docs/guides/auth/row-level-security)

---

## 💡 Tips

1. **Use Supabase Studio**: Visual interface for managing data
2. **Enable Realtime**: Get live updates when data changes
3. **Use Supabase Storage**: Store food images
4. **Set up Backups**: Enable automatic daily backups
5. **Monitor Usage**: Stay within free tier limits or upgrade

---

## 🆘 Need Help?

- Check Supabase [Discord](https://discord.supabase.com)
- Review [GitHub Issues](https://github.com/supabase/supabase/issues)
- Read [Supabase Blog](https://supabase.com/blog)

---

**Happy Building! 🎉**
