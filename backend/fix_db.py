from sqlmodel import Session, select
from app.db.database import engine
from app.models.user import User
from app.core.security import get_password_hash

def fix():
    with Session(engine) as session:
        # Get admin
        admin_user = session.exec(select(User).where(User.username == "admin")).first()
        if admin_user:
            admin_user.hashed_password = get_password_hash("secretpassword")
            session.commit()
            print("Fixed admin user. Login with admin / secretpassword")
        else:
            print("Admin user not found.")

if __name__ == "__main__":
    fix()
