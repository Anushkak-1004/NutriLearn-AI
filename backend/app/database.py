"""
Database Configuration with Supabase Integration
Handles data persistence using Supabase PostgreSQL database.
Falls back to in-memory storage if Supabase is not configured.
"""

import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from collections import defaultdict
from pathlib import Path

from .models import MealLog, UserStats, LearningModule, NutritionInfo

logger = logging.getLogger(__name__)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    # Try to find .env file in backend directory
    env_path = Path(__file__).parent.parent / '.env'
    if env_path.exists():
        load_dotenv(dotenv_path=env_path)
        logger.debug(f"Loaded environment from: {env_path}")
except ImportError:
    logger.warning("python-dotenv not installed, environment variables must be set manually")
except Exception as e:
    logger.warning(f"Could not load .env file: {e}")

# Environment variables for Supabase
SUPABASE_URL = os.getenv("SUPABASE_URL", "")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "")
USE_SUPABASE = os.getenv("USE_SUPABASE", "true").lower() == "true"


# In-memory storage (fallback only)
meal_logs: Dict[str, List[MealLog]] = defaultdict(list)
user_stats: Dict[str, UserStats] = {}
completed_modules: Dict[str, List[str]] = defaultdict(list)


def init_supabase_client():
    """
    Initialize Supabase client for database operations.
    
    Returns:
        Supabase client instance or None if not configured
        
    Raises:
        Exception: If Supabase credentials are invalid
    """
    if not USE_SUPABASE:
        logger.info("Supabase disabled - using in-memory storage")
        return None
    
    if not SUPABASE_URL or not SUPABASE_KEY:
        logger.warning("Supabase credentials not configured - using in-memory storage")
        logger.warning("Set SUPABASE_URL and SUPABASE_KEY in .env file")
        return None
    
    try:
        from supabase import create_client
        
        # Create client (supabase-py 2.3.0)
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        # Test connection by querying users table
        try:
            result = client.table("users").select("id").limit(1).execute()
            logger.info("✅ Supabase client initialized successfully")
            logger.info(f"Connected to: {SUPABASE_URL}")
            return client
        except Exception as e:
            logger.error(f"Supabase connection test failed: {str(e)}")
            logger.error(f"Error details: {type(e).__name__}: {str(e)}")
            logger.warning("Falling back to in-memory storage")
            return None
            
    except ImportError as e:
        logger.error(f"Supabase library not installed: {e}")
        logger.error("Install with: pip install supabase==2.3.0")
        logger.warning("Falling back to in-memory storage")
        return None
    except Exception as e:
        logger.error(f"Failed to initialize Supabase client: {str(e)}")
        logger.error(f"Error type: {type(e).__name__}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        logger.warning("Falling back to in-memory storage")
        return None


def add_meal_log(meal: MealLog) -> str:
    """
    Add a meal log entry to the database.
    
    Args:
        meal: MealLog object to store
        
    Returns:
        Log ID of the created entry
        
    Raises:
        Exception: If database operation fails
    """
    if supabase_client:
        try:
            # Prepare data for Supabase
            data = {
                "user_id": meal.user_id,
                "food_name": meal.food_name,
                "calories": meal.nutrition.calories,
                "protein": float(meal.nutrition.protein),
                "carbs": float(meal.nutrition.carbs),
                "fat": float(meal.nutrition.fat),
                "fiber": float(meal.nutrition.fiber),
                "meal_type": meal.meal_type.value if hasattr(meal.meal_type, 'value') else meal.meal_type,
                "logged_at": meal.timestamp.isoformat(),
                "category": getattr(meal, 'category', None),
                "cuisine": getattr(meal, 'cuisine', None)
            }
            
            # Insert into Supabase
            result = supabase_client.table("meals").insert(data).execute()
            
            if result.data and len(result.data) > 0:
                log_id = str(result.data[0]["id"])
                meal.log_id = log_id
                logger.info(f"Meal logged to Supabase: {meal.food_name} for user {meal.user_id}")
                return log_id
            else:
                raise Exception("No data returned from Supabase insert")
                
        except Exception as e:
            logger.error(f"Supabase insert failed: {str(e)}")
            logger.warning("Falling back to in-memory storage")
            # Fall through to in-memory storage
    
    # Fallback: In-memory storage
    log_id = f"log_{meal.user_id}_{len(meal_logs[meal.user_id]) + 1}_{int(datetime.utcnow().timestamp())}"
    meal.log_id = log_id
    meal_logs[meal.user_id].append(meal)
    logger.info(f"Meal logged to memory: {meal.food_name} for user {meal.user_id}")
    return log_id


def get_user_meals(
    user_id: str,
    limit: int = 50,
    offset: int = 0,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None
) -> List[MealLog]:
    """
    Retrieve meal logs for a user with pagination and filtering.
    
    Args:
        user_id: User identifier
        limit: Maximum number of records to return
        offset: Number of records to skip
        start_date: Filter meals after this date
        end_date: Filter meals before this date
        
    Returns:
        List of MealLog objects
        
    Raises:
        Exception: If database query fails
    """
    if supabase_client:
        try:
            # Build query
            query = supabase_client.table("meals").select("*").eq("user_id", user_id)
            
            # Apply date filters
            if start_date:
                query = query.gte("logged_at", start_date.isoformat())
            if end_date:
                query = query.lte("logged_at", end_date.isoformat())
            
            # Apply ordering and pagination
            query = query.order("logged_at", desc=True).range(offset, offset + limit - 1)
            
            # Execute query
            result = query.execute()
            
            # Convert to MealLog objects
            meals = []
            for item in result.data:
                nutrition = NutritionInfo(
                    calories=item["calories"],
                    protein=item["protein"],
                    carbs=item["carbs"],
                    fat=item["fat"],
                    fiber=item["fiber"]
                )
                
                meal = MealLog(
                    user_id=item["user_id"],
                    food_name=item["food_name"],
                    nutrition=nutrition,
                    meal_type=item["meal_type"],
                    timestamp=datetime.fromisoformat(item["logged_at"].replace('Z', '+00:00')),
                    log_id=str(item["id"])
                )
                meals.append(meal)
            
            logger.info(f"Retrieved {len(meals)} meals from Supabase for user {user_id}")
            return meals
            
        except Exception as e:
            logger.error(f"Supabase query failed: {str(e)}")
            logger.warning("Falling back to in-memory storage")
            # Fall through to in-memory storage
    
    # Fallback: In-memory storage
    user_meals = meal_logs.get(user_id, [])
    
    # Apply date filters
    filtered_meals = user_meals
    if start_date or end_date:
        filtered_meals = [
            meal for meal in user_meals
            if (not start_date or meal.timestamp >= start_date) and
               (not end_date or meal.timestamp <= end_date)
        ]
    
    # Sort by timestamp (newest first)
    filtered_meals.sort(key=lambda x: x.timestamp, reverse=True)
    
    # Apply pagination
    return filtered_meals[offset:offset + limit]


def get_user_stats(user_id: str) -> UserStats:
    """
    Get or create user statistics.
    
    Args:
        user_id: User identifier
        
    Returns:
        UserStats object
        
    Raises:
        Exception: If database operation fails
    """
    if supabase_client:
        try:
            # Query user from Supabase
            result = supabase_client.table("users").select("*").eq("id", user_id).execute()
            
            if result.data and len(result.data) > 0:
                user_data = result.data[0]
                
                # Get completed modules
                modules_result = supabase_client.table("user_modules").select("module_id").eq("user_id", user_id).execute()
                completed = [m["module_id"] for m in modules_result.data] if modules_result.data else []
                
                stats = UserStats(
                    user_id=user_id,
                    total_meals=user_data.get("total_meals", 0),
                    total_points=user_data.get("total_points", 0),
                    completed_modules=completed,
                    current_streak=user_data.get("current_streak", 0),
                    last_updated=datetime.fromisoformat(user_data["updated_at"].replace('Z', '+00:00'))
                )
                
                logger.info(f"Retrieved stats from Supabase for user {user_id}")
                return stats
            else:
                # Create new user
                new_user_data = {
                    "id": user_id,
                    "total_meals": 0,
                    "total_points": 0,
                    "current_streak": 0
                }
                supabase_client.table("users").insert(new_user_data).execute()
                
                logger.info(f"Created new user in Supabase: {user_id}")
                return UserStats(user_id=user_id)
                
        except Exception as e:
            logger.error(f"Supabase query failed: {str(e)}")
            logger.warning("Falling back to in-memory storage")
            # Fall through to in-memory storage
    
    # Fallback: In-memory storage
    if user_id not in user_stats:
        user_stats[user_id] = UserStats(user_id=user_id)
    
    return user_stats[user_id]


def update_user_stats(stats: UserStats) -> UserStats:
    """
    Update user statistics in the database.
    
    Args:
        stats: Updated UserStats object
        
    Returns:
        Updated UserStats object
        
    Raises:
        Exception: If database operation fails
    """
    if supabase_client:
        try:
            # Prepare update data
            update_data = {
                "total_meals": stats.total_meals,
                "total_points": stats.total_points,
                "current_streak": stats.current_streak,
                "updated_at": datetime.utcnow().isoformat()
            }
            
            # Update in Supabase
            supabase_client.table("users").update(update_data).eq("id", stats.user_id).execute()
            
            stats.last_updated = datetime.utcnow()
            logger.info(f"Updated stats in Supabase for user {stats.user_id}")
            return stats
            
        except Exception as e:
            logger.error(f"Supabase update failed: {str(e)}")
            logger.warning("Falling back to in-memory storage")
            # Fall through to in-memory storage
    
    # Fallback: In-memory storage
    stats.last_updated = datetime.utcnow()
    user_stats[stats.user_id] = stats
    logger.info(f"Updated stats in memory for user {stats.user_id}")
    return stats


def mark_module_completed(user_id: str, module_id: str, quiz_score: int = 0, points_earned: int = 0) -> bool:
    """
    Mark a learning module as completed for a user.
    
    Args:
        user_id: User identifier
        module_id: Module identifier
        quiz_score: Score achieved on quiz (0-100)
        points_earned: Points awarded for completion
        
    Returns:
        True if successfully marked, False if already completed
        
    Raises:
        Exception: If database operation fails
    """
    if supabase_client:
        try:
            # Check if already completed
            existing = supabase_client.table("user_modules").select("*").eq("user_id", user_id).eq("module_id", module_id).execute()
            
            if existing.data and len(existing.data) > 0:
                logger.info(f"Module {module_id} already completed by user {user_id}")
                return False
            
            # Insert completion record
            data = {
                "user_id": user_id,
                "module_id": module_id,
                "quiz_score": quiz_score,
                "points_earned": points_earned,
                "completed_at": datetime.utcnow().isoformat()
            }
            supabase_client.table("user_modules").insert(data).execute()
            
            logger.info(f"Module {module_id} completed by user {user_id} in Supabase")
            return True
            
        except Exception as e:
            logger.error(f"Supabase insert failed: {str(e)}")
            logger.warning("Falling back to in-memory storage")
            # Fall through to in-memory storage
    
    # Fallback: In-memory storage
    if module_id in completed_modules[user_id]:
        return False
    
    completed_modules[user_id].append(module_id)
    logger.info(f"Module {module_id} completed by user {user_id} in memory")
    return True


def get_completed_modules(user_id: str) -> List[str]:
    """
    Get list of completed module IDs for a user.
    
    Args:
        user_id: User identifier
        
    Returns:
        List of module IDs
        
    Raises:
        Exception: If database query fails
    """
    if supabase_client:
        try:
            result = supabase_client.table("user_modules").select("module_id").eq("user_id", user_id).execute()
            
            modules = [item["module_id"] for item in result.data] if result.data else []
            logger.info(f"Retrieved {len(modules)} completed modules from Supabase for user {user_id}")
            return modules
            
        except Exception as e:
            logger.error(f"Supabase query failed: {str(e)}")
            logger.warning("Falling back to in-memory storage")
            # Fall through to in-memory storage
    
    # Fallback: In-memory storage
    return completed_modules.get(user_id, [])


def calculate_streak(user_id: str) -> int:
    """
    Calculate the current daily logging streak for a user.
    
    Args:
        user_id: User identifier
        
    Returns:
        Number of consecutive days with logged meals
    """
    user_meals = meal_logs.get(user_id, [])
    if not user_meals:
        return 0
    
    # Sort meals by date
    sorted_meals = sorted(user_meals, key=lambda x: x.timestamp, reverse=True)
    
    # Get unique dates
    logged_dates = set()
    for meal in sorted_meals:
        date = meal.timestamp.date()
        logged_dates.add(date)
    
    # Calculate streak
    streak = 0
    current_date = datetime.utcnow().date()
    
    while current_date in logged_dates:
        streak += 1
        current_date -= timedelta(days=1)
    
    return streak


def create_user(email: str, password_hash: str) -> str:
    """
    Create a new user in the database.
    
    Args:
        email: User's email address
        password_hash: Bcrypt hashed password
        
    Returns:
        user_id: UUID of created user
        
    Raises:
        Exception: If email already exists or database operation fails
    """
    if supabase_client:
        try:
            # Prepare user data
            user_data = {
                "email": email,
                "password_hash": password_hash,
                "total_meals": 0,
                "total_points": 0,
                "current_streak": 0
            }
            
            # Insert into Supabase
            result = supabase_client.table("users").insert(user_data).execute()
            
            if result.data and len(result.data) > 0:
                user_id = str(result.data[0]["id"])
                logger.info(f"User created in Supabase: {email} (ID: {user_id})")
                return user_id
            else:
                raise Exception("No data returned from Supabase insert")
                
        except Exception as e:
            # Check if it's a duplicate email error
            error_msg = str(e).lower()
            if "duplicate" in error_msg or "unique" in error_msg or "already exists" in error_msg:
                logger.warning(f"Duplicate email attempted: {email}")
                raise Exception(f"Email {email} is already registered")
            
            logger.error(f"Supabase user creation failed: {str(e)}")
            logger.warning("Falling back to in-memory storage")
            # Fall through to in-memory storage
    
    # Fallback: In-memory storage
    # We need a separate dictionary for user credentials since UserStats is a Pydantic model
    # and doesn't allow arbitrary attributes
    if not hasattr(create_user, '_user_credentials'):
        create_user._user_credentials = {}
    
    # Check for duplicate email in memory
    for uid, creds in create_user._user_credentials.items():
        if creds['email'] == email:
            raise Exception(f"Email {email} is already registered")
    
    # Generate a simple UUID-like ID for in-memory storage
    import uuid
    user_id = str(uuid.uuid4())
    
    # Store user credentials separately
    create_user._user_credentials[user_id] = {
        'email': email,
        'password_hash': password_hash,
        'created_at': datetime.utcnow()
    }
    
    # Create user stats
    user_stats[user_id] = UserStats(user_id=user_id)
    
    logger.info(f"User created in memory: {email} (ID: {user_id})")
    return user_id


def get_user_by_email(email: str) -> Optional[Dict[str, Any]]:
    """
    Fetch user by email address.
    
    Args:
        email: User's email address
        
    Returns:
        User dictionary with id, email, password_hash, created_at
        Returns None if user not found
    """
    if supabase_client:
        try:
            # Query user by email
            result = supabase_client.table("users").select("*").eq("email", email).execute()
            
            if result.data and len(result.data) > 0:
                user_data = result.data[0]
                logger.info(f"User found in Supabase: {email}")
                return {
                    "id": str(user_data["id"]),
                    "email": user_data["email"],
                    "password_hash": user_data["password_hash"],
                    "created_at": datetime.fromisoformat(user_data["created_at"].replace('Z', '+00:00')) if user_data.get("created_at") else datetime.utcnow(),
                    "total_meals": user_data.get("total_meals", 0),
                    "total_points": user_data.get("total_points", 0),
                    "current_streak": user_data.get("current_streak", 0)
                }
            else:
                logger.info(f"User not found in Supabase: {email}")
                return None
                
        except Exception as e:
            logger.error(f"Supabase query failed: {str(e)}")
            logger.warning("Falling back to in-memory storage")
            # Fall through to in-memory storage
    
    # Fallback: In-memory storage
    if hasattr(create_user, '_user_credentials'):
        for user_id, creds in create_user._user_credentials.items():
            if creds['email'] == email:
                stats = user_stats.get(user_id, UserStats(user_id=user_id))
                logger.info(f"User found in memory: {email}")
                return {
                    "id": user_id,
                    "email": creds['email'],
                    "password_hash": creds['password_hash'],
                    "created_at": creds['created_at'],
                    "total_meals": stats.total_meals,
                    "total_points": stats.total_points,
                    "current_streak": stats.current_streak
                }
    
    logger.info(f"User not found in memory: {email}")
    return None


def get_user_by_id(user_id: str) -> Optional[Dict[str, Any]]:
    """
    Fetch user by ID.
    
    Args:
        user_id: User's UUID
        
    Returns:
        User dictionary with id, email, created_at (no password_hash)
        Returns None if user not found
    """
    if supabase_client:
        try:
            # Query user by ID
            result = supabase_client.table("users").select("id, email, created_at, total_meals, total_points, current_streak").eq("id", user_id).execute()
            
            if result.data and len(result.data) > 0:
                user_data = result.data[0]
                logger.info(f"User found in Supabase by ID: {user_id}")
                return {
                    "id": str(user_data["id"]),
                    "email": user_data["email"],
                    "created_at": datetime.fromisoformat(user_data["created_at"].replace('Z', '+00:00')) if user_data.get("created_at") else datetime.utcnow(),
                    "total_meals": user_data.get("total_meals", 0),
                    "total_points": user_data.get("total_points", 0),
                    "current_streak": user_data.get("current_streak", 0)
                }
            else:
                logger.info(f"User not found in Supabase by ID: {user_id}")
                return None
                
        except Exception as e:
            logger.error(f"Supabase query failed: {str(e)}")
            logger.warning("Falling back to in-memory storage")
            # Fall through to in-memory storage
    
    # Fallback: In-memory storage
    if hasattr(create_user, '_user_credentials') and user_id in create_user._user_credentials:
        creds = create_user._user_credentials[user_id]
        stats = user_stats.get(user_id, UserStats(user_id=user_id))
        logger.info(f"User found in memory by ID: {user_id}")
        return {
            "id": user_id,
            "email": creds['email'],
            "created_at": creds['created_at'],
            "total_meals": stats.total_meals,
            "total_points": stats.total_points,
            "current_streak": stats.current_streak
        }
    
    logger.info(f"User not found in memory by ID: {user_id}")
    return None


def clear_all_data():
    """
    Clear all in-memory data (for testing purposes).
    
    WARNING: This will delete all data in development mode.
    Does NOT affect Supabase data.
    """
    meal_logs.clear()
    user_stats.clear()
    completed_modules.clear()
    logger.warning("All in-memory data cleared")


def test_connection() -> bool:
    """
    Test the database connection.
    
    Returns:
        True if connection is successful, False otherwise
    """
    if supabase_client:
        try:
            # Try to query users table
            result = supabase_client.table("users").select("id").limit(1).execute()
            logger.info("✅ Database connection test successful")
            logger.info(f"Connected to Supabase at: {SUPABASE_URL}")
            return True
        except Exception as e:
            logger.error(f"❌ Database connection test failed: {str(e)}")
            return False
    else:
        logger.warning("⚠️  Supabase not configured - using in-memory storage")
        return False


def get_database_info() -> Dict[str, Any]:
    """
    Get information about the current database configuration.
    
    Returns:
        Dictionary with database configuration details
    """
    return {
        "using_supabase": supabase_client is not None,
        "supabase_url": SUPABASE_URL if SUPABASE_URL else "Not configured",
        "supabase_configured": bool(SUPABASE_URL and SUPABASE_KEY),
        "fallback_storage": "in-memory" if not supabase_client else "none"
    }


# Initialize database connection
supabase_client = init_supabase_client()

# Log database status on module import
if supabase_client:
    logger.info("=" * 60)
    logger.info("DATABASE: Supabase PostgreSQL")
    logger.info(f"URL: {SUPABASE_URL}")
    logger.info("=" * 60)
else:
    logger.warning("=" * 60)
    logger.warning("DATABASE: In-Memory Storage (Development Only)")
    logger.warning("Configure Supabase for production use")
    logger.warning("=" * 60)
