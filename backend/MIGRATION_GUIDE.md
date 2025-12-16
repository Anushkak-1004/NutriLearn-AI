# 🔄 Database Migration Guide

## Quick Fix for "Trigger Already Exists" Error

You're seeing this error because the tables were partially created from a previous migration attempt.

---

## ✅ Solution: Use the Clean Migration Script

### Option 1: Run the Clean Migration (Recommended)

1. **Open Supabase Dashboard**
   - Go to: https://app.supabase.com
   - Select your project
   - Navigate to: **SQL Editor**

2. **Run the Clean Migration**
   - Open the file: `backend/migrations/002_update_schema.sql`
   - Copy the **entire contents**
   - Paste into Supabase SQL Editor
   - Click **"Run"**

3. **Verify Success**
   - You should see: ✅ "Success. No rows returned"
   - Check **Table Editor** → You should see 3 tables:
     - `users`
     - `meals`
     - `user_modules`

**⚠️ Note**: This will **drop and recreate** all tables, clearing any existing data.

---

### Option 2: Manual Cleanup (If you want to keep data)

If you have important data and want to preserve it:

1. **Export existing data** (if any):
   ```sql
   -- In Supabase SQL Editor
   SELECT * FROM users;
   SELECT * FROM meals;
   SELECT * FROM user_modules;
   ```
   Copy the results to save them.

2. **Drop only the problematic triggers**:
   ```sql
   DROP TRIGGER IF EXISTS update_users_updated_at ON users;
   DROP TRIGGER IF EXISTS ensure_user_before_meal ON meals;
   DROP TRIGGER IF EXISTS increment_meal_count ON meals;
   DROP TRIGGER IF EXISTS update_points_on_module_completion ON user_modules;
   ```

3. **Then run** `001_initial_schema.sql` again

---

## 🧪 Test the Connection

After running the migration:

```bash
cd backend
.\venv\Scripts\python.exe -c "from app.database import test_connection; test_connection()"
```

Expected output:
```
✅ Database connection test successful
Connected to Supabase at: https://invdhwdrlzflrzgqzryp.supabase.co
```

---

## 🚀 Start Using the Database

### 1. Start the Backend
```bash
cd backend
.\venv\Scripts\activate
uvicorn app.main:app --reload
```

### 2. Test with API
```bash
# Log a meal
curl -X POST http://localhost:8000/api/v1/meals/log \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user123",
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

### 3. Verify in Supabase
- Go to **Table Editor** → **meals**
- You should see your logged meal!

---

## 📋 What Changed

### Key Update: TEXT User IDs

The schema now uses **TEXT** for user IDs instead of UUID:

**Before** (001_initial_schema.sql):
```sql
id UUID PRIMARY KEY
user_id UUID NOT NULL
```

**After** (002_update_schema.sql):
```sql
id TEXT PRIMARY KEY
user_id TEXT NOT NULL
```

**Why?** This makes it more flexible for development and works with simple string user IDs like `"user123"` instead of requiring UUID format.

---

## 🔍 Troubleshooting

### Error: "relation already exists"
**Solution**: Use `002_update_schema.sql` which drops existing tables first.

### Error: "trigger already exists"
**Solution**: Use `002_update_schema.sql` which drops existing triggers first.

### Error: "invalid input syntax for type uuid"
**Solution**: This is fixed! The new schema uses TEXT for user IDs.

### Connection fails
**Check**:
1. `.env` file has correct `SUPABASE_URL` and `SUPABASE_KEY`
2. Dependencies are updated: `pip install --upgrade supabase`
3. Supabase project is active and accessible

---

## ✅ Success Checklist

- [ ] Ran `002_update_schema.sql` in Supabase SQL Editor
- [ ] Saw success message with no errors
- [ ] Verified 3 tables exist in Table Editor
- [ ] Connection test passes
- [ ] Backend starts without errors
- [ ] Can log meals via API
- [ ] Data appears in Supabase Table Editor

---

## 📚 Related Files

- `migrations/001_initial_schema.sql` - Original schema (UUID user IDs)
- `migrations/002_update_schema.sql` - Updated schema (TEXT user IDs) ← **Use this one**
- `SUPABASE_COMPLETE.md` - Full integration documentation
- `test_supabase_integration.py` - Integration test suite

---

**Need Help?** Check `SUPABASE_COMPLETE.md` for comprehensive documentation.
