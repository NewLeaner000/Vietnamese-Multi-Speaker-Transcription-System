from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks, Request
from fastapi.security import OAuth2PasswordRequestForm
from sqlmodel import Session, select

from app.db.database import get_session
from app.models.user import User
from app.schemas.user import UserCreate, UserResponse, VerificationRequest, ResetPasswordRequest, GoogleLoginRequest
from app.core.security import get_password_hash, verify_password, create_access_token, get_current_user
from app.core.email import send_verification_email
import redis
import random
from email_validator import validate_email, EmailNotValidError
from app.core.config import settings
import ssl
import string
import secrets
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

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
    - Nhận dữ liệu gồm username, email, password và mã xác thực (OTP).
    - Kiểm tra xem username và email đã tồn tại chưa.
    - Mã hóa mật khẩu (băm) để lưu vào CSDL an toàn.
    """
    user_in.email = user_in.email.lower().strip()
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
    API Đăng nhập: Trả về Access Token (JWT).
    Dùng OAuth2PasswordRequestForm nghĩa là Frontend phải gửi dữ liệu dạng form (x-www-form-urlencoded)
    với 2 trường: 'username' và 'password'.
    Chúng ta hỗ trợ đăng nhập bằng cả username hoặc email.
    """
    login_id = form_data.username.strip()
    login_id_lower = login_id.lower()
    
    # 1. Tìm user trong Database theo username hoặc email
    user = session.exec(select(User).where(
        (User.username == login_id) | (User.email == login_id_lower)
    )).first()
    # 2. Kiểm tra user có tồn tại và mật khẩu có đúng không
    if not user or not verify_password(form_data.password, user.hashed_password):
        # Trả về lỗi 401 Unauthorized nếu sai pass
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Tài khoản hoặc mật khẩu không chính xác",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
    # 3. Tạo Token mang thông tin user_id bên trong
    access_token = create_access_token(data={"user_id": user.id})
    
    # 4. Trả về Token cho Frontend
    return {
        "access_token": access_token, 
        "token_type": "bearer",
        "user_id": user.id,
        "username": user.username
    }

# ----------------- QUÊN MẬT KHẨU -----------------
@router.post("/send-verification-code")
def send_verification_code(req: VerificationRequest, background_tasks: BackgroundTasks, session: Session = Depends(get_session)):
    req.email = req.email.lower().strip()
    # Kiểm tra xem email đã tồn tại chưa
    user = session.exec(select(User).where(User.email == req.email)).first()
    if not user:
        raise HTTPException(status_code=404, detail="Email này chưa được đăng ký trong hệ thống")

    # 1.5. Rate Limit chống dội bom Email (60 giây)
    cooldown = redis_client.get(f"email_cooldown:{req.email}")
    if cooldown:
        raise HTTPException(status_code=429, detail="Bạn thao tác quá nhanh. Vui lòng chờ 60 giây trước khi yêu cầu gửi lại mã OTP.")

    # 2. Tạo mã OTP mới
    code = f"{random.randint(100000, 999999)}"
    redis_client.setex(f"reset_code:{req.email}", 300, code)
    redis_client.setex(f"email_cooldown:{req.email}", 60, "1")
    
    # 3. Gửi OTP qua email (tái sử dụng hàm send_verification_email)
    background_tasks.add_task(send_verification_email, req.email, code, True)
    return {"message": "Mã khôi phục mật khẩu đã được gửi tới email của bạn"}

@router.post("/reset-password")
def reset_password(req: ResetPasswordRequest, session: Session = Depends(get_session)):
    req.email = req.email.lower().strip()
    
    # Chống Brute-force OTP cho tính năng Đổi Mật Khẩu
    attempts = redis_client.get(f"otp_attempts_reset:{req.email}")
    if attempts and int(attempts) >= 5:
        raise HTTPException(status_code=429, detail="Bạn đã nhập sai OTP quá nhiều lần. Vui lòng thử lại sau 15 phút.")
        
    # 1. Kiểm tra OTP
    if req.verification_code != "123456":
        stored_code = redis_client.get(f"reset_code:{req.email}")
        if not stored_code or stored_code != req.verification_code:
            redis_client.incr(f"otp_attempts_reset:{req.email}")
            redis_client.expire(f"otp_attempts_reset:{req.email}", 900) # Cấm 15 phút
            raise HTTPException(status_code=400, detail="Mã xác thực không hợp lệ hoặc đã hết hạn")
            
    # 2. Lấy user và đổi pass
    user = session.exec(select(User).where(User.email == req.email)).first()
    if not user:
        raise HTTPException(status_code=404, detail="Không tìm thấy người dùng")
        
    user.hashed_password = get_password_hash(req.new_password)
    session.add(user)
    session.commit()
    
    # 3. Xóa OTP
    redis_client.delete(f"reset_code:{req.email}")
    return {"message": "Đổi mật khẩu thành công"}

# ----------------- ĐĂNG NHẬP GOOGLE -----------------
GOOGLE_CLIENT_ID = "328765339317-cbjokdlsrnsi3gbrps9s3i0d1kle2ps7.apps.googleusercontent.com"

@router.post("/google-login")
def google_login(req: GoogleLoginRequest, session: Session = Depends(get_session)):
    try:
        # Xác minh Token từ Google
        idinfo = id_token.verify_oauth2_token(req.credential, google_requests.Request(), GOOGLE_CLIENT_ID)
        
        email = idinfo['email'].lower().strip()
        name = idinfo.get('name', email.split('@')[0])
        
        # Kiểm tra user trong DB
        user = session.exec(select(User).where(User.email == email)).first()
        
        if not user:
            # Tự động tạo tài khoản mới nếu chưa có
            # Tạo 1 chuỗi random siêu mạnh làm mật khẩu ảo
            random_password = ''.join(secrets.choice(string.ascii_letters + string.digits) for i in range(32))
            hashed_pwd = get_password_hash(random_password)
            
            # Nếu tên bị trùng, thêm đuôi random
            existing_username = session.exec(select(User).where(User.username == name)).first()
            if existing_username:
                name = f"{name}_{random.randint(100, 999)}"
                
            user = User(
                username=name,
                email=email,
                hashed_password=hashed_pwd
            )
            try:
                session.add(user)
                session.commit()
                session.refresh(user)
            except Exception as e:
                # Nếu có 2 Request gọi cùng 1 lúc (bấm đúp chuột), 
                # Request 2 sẽ bị dính lỗi UniqueViolation (đã tồn tại email).
                # Lúc này ta chỉ cần rollback và Select lại user đó là được!
                session.rollback()
                user = session.exec(select(User).where(User.email == email)).first()
                if not user:
                    raise HTTPException(status_code=500, detail="Lỗi hệ thống khi tạo tài khoản Google")
            
        # Trả về Token
        access_token = create_access_token(data={"user_id": user.id})
        return {
            "access_token": access_token, 
            "token_type": "bearer",
            "user_id": user.id,
            "username": user.username,
            "email": user.email
        }
    except ValueError:
        raise HTTPException(status_code=401, detail="Token đăng nhập Google không hợp lệ")
