from app.db.database import engine
from sqlmodel import Session, select
from app.models.user import User
from app.core.security import get_password_hash

def create_demo_user():
    with Session(engine) as session:
        # Check if exists
        user = session.exec(select(User).where(User.username == "demouser")).first()
        if user:
            print("Demo user already exists.")
            return

        demo_user = User(
            username="demouser",
            email="demouser@vimeet.vn",
            hashed_password=get_password_hash("123"),
            is_active=True
        )
        session.add(demo_user)
        session.commit()
        print("Demo user created successfully: demouser / 123")

if __name__ == "__main__":
    create_demo_user()
