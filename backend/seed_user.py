from sqlmodel import Session
from app.db.database import engine
from app.models.user import User

def seed():
    with Session(engine) as session:
        user = User(username="admin", email="admin@example.com", hashed_password="abc", is_active=True)
        session.add(user)
        session.commit()
        print("User 1 created!")

if __name__ == "__main__":
    seed()
