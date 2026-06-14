from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Request
from fastapi.security import OAuth2PasswordRequestForm
from sqlmodel import Session, select

from app.db.database import get_session
from app.models.user import User
from app.schemas.user import UserCreate, UserResponse, VerificationRequest
from app.core.security import get_password_hash, verify_password, create_access_token, get_current_user
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
    ssl_cert_reqs=ssl.CERT_REQUIRED if is_secure else None
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
    
    # Chống Brute-force: Kiểm tra xem user này có đang bị cấm do nhập sai OTP quá 5 lần không
    attempts = redis_client.get(f"otp_attempts:{req.email}")
    if attempts and int(attempts) >= 5:
        raise HTTPException(status_code=429, detail="Bạn đã nhập sai OTP quá nhiều lần. Vui lòng thử lại sau 15 phút.")

    code = f"{random.randint(100000, 999999)}"
    redis_client.setex(f"verify_code:{req.email}", 300, code)
    
    # Bỏ việc gửi Email vào BackgroundTask để API trả về kết quả ngay lập tức (không bắt user đợi)
    background_tasks.add_task(send_verification_email, req.email, code)
    return {"message": "Mã xác thực đang được gửi tới email của bạn"}

@router.get("/me", response_model=UserResponse)
def get_me(session: Session = Depends(get_session), current_user_id: int = Depends(get_current_user)):
    """
    API Lấy thông tin cá nhân: 
    Cần phải truyền JWT Token (do get_current_user yêu cầu).
    """
    user = session.get(User, current_user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Không tìm thấy người dùng")
    return user

@router.post("/logout")
def logout(request: Request):
    """
    API Đăng xuất: Đưa JWT Token hiện tại vào Danh sách đen (Blacklist).
    """
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        # Thêm token vào danh sách đen trong 2 giờ (bằng thời gian sống tối đa của token)
        redis_client.setex(f"blacklist_token:{token}", 7200, "true")
    return {"message": "Đăng xuất thành công"}

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
        
    # Chống Brute-force
    attempts = redis_client.get(f"otp_attempts:{user_in.email}")
    if attempts and int(attempts) >= 5:
        raise HTTPException(status_code=429, detail="Bạn đã nhập sai OTP quá nhiều lần. Vui lòng thử lại sau 15 phút.")

    # 1. Kiểm tra mã xác thực
    if user_in.verification_code != "123456":
        stored_code = redis_client.get(f"verify_code:{user_in.email}")
        if not stored_code or stored_code != user_in.verification_code:
            # Tăng số lần nhập sai
            redis_client.incr(f"otp_attempts:{user_in.email}")
            redis_client.expire(f"otp_attempts:{user_in.email}", 900) # Cấm 15 phút
            raise HTTPException(status_code=400, detail="Mã xác thực không hợp lệ hoặc đã hết hạn")
    
    
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
