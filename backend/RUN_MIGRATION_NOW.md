# 🚀 RUN THIS MIGRATION NOW

## Quick 3-Step Process

### Step 1: Open Supabase SQL Editor
1. Go to: **https://app.supabase.com**
2. Select your project: **NutriLearn AI**
3. Click: **SQL Editor** (left sidebar)

### Step 2: Run the Migration
1. Open file: `backend/migrations/002_update_schema.sql`
2. **Copy ALL the contents** (Ctrl+A, Ctrl+C)
3. **Paste** into Supabase SQL Editor
4. Click: **"Run"** button (or press Ctrl+Enter)

### Step 3: Verify Success
You should see:
```
✅ Success. No rows returned
```

Then check **Table Editor** (left sidebar):
- ✅ `users` table exists
- ✅ `meals` table exists  
- ✅ `user_modules` table exists

---

## ✅ Done! Now Test It

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

## 🎉 That's It!

Your database is now ready. Start the backend:

```bash
uvicorn app.main:app --reload
```

---

**Need help?** See `MIGRATION_GUIDE.md` for detailed instructions.
