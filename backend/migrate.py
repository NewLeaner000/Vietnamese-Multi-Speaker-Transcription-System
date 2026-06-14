import sqlite3
import os

db_path = os.path.join(os.path.dirname(__file__), 'app.db')
conn = sqlite3.connect(db_path)
try:
    conn.execute('ALTER TABLE job ADD COLUMN has_enrollment BOOLEAN DEFAULT 0')
    conn.commit()
    print("Migration success")
except Exception as e:
    print(e)
