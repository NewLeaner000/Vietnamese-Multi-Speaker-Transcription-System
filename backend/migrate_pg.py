from app.db.database import engine
from sqlalchemy import text

try:
    with engine.begin() as conn:
        conn.execute(text('ALTER TABLE job ADD COLUMN has_enrollment BOOLEAN DEFAULT FALSE'))
        print("Migration success!")
except Exception as e:
    print(f"Error: {e}")
