from app.db.database import engine
from sqlmodel import SQLModel
from app.models.job import Job
from app.models.transcript import Transcript
from app.models.summary import Summary
from app.models.user import User
import shutil
import os

def recreate_tables():
    print("Dropping tables...")
    Transcript.__table__.drop(engine, checkfirst=True)
    Summary.__table__.drop(engine, checkfirst=True)
    Job.__table__.drop(engine, checkfirst=True)
    User.__table__.drop(engine, checkfirst=True) # Xóa cả dữ liệu User
    
    print("Recreating tables...")
    SQLModel.metadata.create_all(engine)
    
    print("Cleaning directories...")
    # Xóa file audio
    if os.path.exists("uploads"):
        shutil.rmtree("uploads")
    os.makedirs("uploads")
        
    # Xóa file AI output
    output_dir = os.path.join("app", "ai_core", "output")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    print("Database and directories have been completely wiped. Clean state!")

if __name__ == "__main__":
    recreate_tables()
