from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from fastapi.security import OAuth2PasswordRequestForm
from sqlmodel import Session, select

from app.db.database import get_session
from app.models.user import User
from app.schemas.user import UserCreate, UserResponse, VerificationRequest
from app.core.security import get_password_hash, verify_password, create_access_token
from app.core.email import send_verification_email
import redis
import random
from email_validator import validate_email, EmailNotValidError
from app.core.config import settings
import ssl

router = APIRouter()

# Sử dụng chung Redis URL từ Upstash
# Nếu URL là rediss:// thì tự động kích hoạt TLS
is_secure = settings.REDIS_URL.startswith("rediss://")
redis_client = redis.from_url(
    settings.REDIS_URL, 
    decode_responses=True,
    ssl_cert_reqs=ssl.CERT_NONE if is_secure else None
)

@router.post("/send-verification-code")
def send_code(req: VerificationRequest, background_tasks: BackgroundTasks):
    # Kiểm tra Email có tồn tại thật (MX Records) hay không
    # Tắt check_deliverability để tăng tốc độ nếu cần, nhưng MX record check thường mất ~1s
    try:
        validate_email(req.email, check_deliverability=True)
    except EmailNotValidError as e:
        raise HTTPException(status_code=400, detail=f"Email không hợp lệ hoặc không có thực: {str(e)}")

    # Check if email already exists
    from app.db.database import engine
    with Session(engine) as session:
        email_exists = session.exec(select(User).where(User.email == req.email)).first()
        if email_exists:
            raise HTTPException(status_code=400, detail="Email này đã được đăng ký")
    
    code = f"{random.randint(100000, 999999)}"
    redis_client.setex(f"verify_code:{req.email}", 300, code)
    
    # Bỏ việc gửi Email vào BackgroundTask để API trả về kết quả ngay lập tức (không bắt user đợi)
    background_tasks.add_task(send_verification_email, req.email, code)
    return {"message": "Mã xác thực đang được gửi tới email của bạn"}

@router.post("/register", response_model=UserResponse)
def register(user_in: UserCreate, session: Session = Depends(get_session)):
    """
    API Đăng ký:
    1. Nhận JSON từ Client (user_in).
    2. Kiểm tra trùng lặp.
    3. Băm mật khẩu (get_password_hash).
    4. Lưu vào PostgreSQL.
    """
    # Kiểm tra Username đã có ai xài chưa
    user_exists = session.exec(select(User).where(User.username == user_in.username)).first()
    if user_exists:
        raise HTTPException(status_code=400, detail="Username đã tồn tại trong hệ thống")
        
    # Kiểm tra Email
    email_exists = session.exec(select(User).where(User.email == user_in.email)).first()
    if email_exists:
        raise HTTPException(status_code=400, detail="Email này đã được đăng ký")
        
    # Xác thực mã OTP từ Redis
    saved_code = redis_client.get(f"verify_code:{user_in.email}")
    if not saved_code or saved_code != user_in.verification_code:
        raise HTTPException(status_code=400, detail="Mã xác thực không đúng hoặc đã hết hạn")
    
    # Xóa mã OTP sau khi dùng thành công
    redis_client.delete(f"verify_code:{user_in.email}")

    # Đưa mật khẩu thô vào "Máy xay" Bcrypt
    hashed_pwd = get_password_hash(user_in.password)
    
    # Tạo bản ghi User mới để lưu DB
    db_user = User(
        username=user_in.username,
        email=user_in.email,
        hashed_password=hashed_pwd
    )
    
    # Ra lệnh cho SQLAlchemy đẩy dữ liệu xuống PostgreSQL
    session.add(db_user)
    session.commit()
    session.refresh(db_user) # Kéo dữ liệu ngược lên để lấy được cái 'id' vừa tự động tăng
    
    # Mặc dù ta return db_user (chứa cả hashed_password), 
    # nhưng nhờ response_model=UserResponse ở trên cùng, FastAPI sẽ tự động "cắt bỏ" cái cột đó đi!
    return db_user

@router.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), session: Session = Depends(get_session)):
    """
    API Đăng nhập: Trả về chiếc vé thông hành JWT (Token)
    """
    # Tìm user trong DB
    user = session.exec(select(User).where(User.username == form_data.username)).first()
    
    # Nếu không thấy user, HOẶC "máy xay" báo mật khẩu gõ vào không khớp với mã băm trong DB
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, 
            detail="Tài khoản hoặc Mật khẩu không chính xác",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    # Nếu đúng, cấp phát vé JWT (Gói cái username và user_id vào trong vé)
    access_token = create_access_token(data={"sub": user.username, "user_id": user.id})
    return {"access_token": access_token, "token_type": "bearer"}
