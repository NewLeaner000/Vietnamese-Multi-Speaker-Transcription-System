from sqlmodel import SQLModel
from app.db.database import engine
from app.models.job import Job
from app.models.user import User
from app.models.transcript import Transcript
from app.models.summary import Summary

def reset_db():
    print("Dropping all tables...")
    SQLModel.metadata.drop_all(engine)
    print("Dropped.")
    print("Recreating all tables...")
    SQLModel.metadata.create_all(engine)
    print("Done.")

if __name__ == "__main__":
    reset_db()
