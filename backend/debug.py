from app.db.database import get_session
from app.api.auth import register
from app.schemas.user import UserCreate

def test_register():
    # Simulate API Request
    session = next(get_session())
    user_in = UserCreate(username="test_admin2", email="test2@gmail.com", password="123")
    
    print("Starting API simulation...")
    try:
        result = register(user_in, session)
        print("Success:", result)
    except Exception as e:
        print("CRITICAL ERROR FOUND:")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_register()
