from passlib.context import CryptContext
from datetime import datetime, timedelta
from jose import jwt, JWTError
import os
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

# 1. Cấu hình công cụ Băm Mật Khẩu (Hashing)
# Bcrypt là thuật toán mã hoá tiêu chuẩn công nghiệp. Nó không chỉ mã hóa mà còn thêm "muối" (salt) 
# để đảm bảo 2 người dùng đặt cùng 1 mật khẩu "123456" sẽ sinh ra 2 chuỗi băm hoàn toàn khác nhau.
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# 2. Cấu hình vé thông hành JWT (JSON Web Token)
# SECRET_KEY là "Con Dấu Khắc" của Server. Bất cứ Token nào không được đóng dấu bằng chữ ký này đều là vé giả.
# Lời khuyên MLOps: Khi đưa lên Production (Triton Server), bắt buộc giấu chuỗi này vào file .env!
SECRET_KEY = os.getenv("SECRET_KEY", "antigravity-super-secret-key-2026")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 7 # Token có hiệu lực trong 7 ngày

def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Kiểm tra mật khẩu thô người dùng gõ vào có khớp với mật khẩu băm trong DB hay không."""
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password: str) -> str:
    """Băm mật khẩu thô thành một chuỗi không thể dịch ngược (VD: $2b$12$Kix...)."""
    return pwd_context.hash(password)

def create_access_token(data: dict) -> str:
    """Phát hành JWT Token sau khi user đăng nhập thành công."""
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire}) # Hạn sử dụng của vé
    
    # Đóng mộc đỏ bằng SECRET_KEY
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

import redis
import ssl
from app.core.config import settings

is_secure = settings.REDIS_URL.startswith("rediss://")
redis_client = redis.from_url(
    settings.REDIS_URL, 
    decode_responses=True,
    ssl_cert_reqs=ssl.CERT_REQUIRED if is_secure else None
)

def get_current_user(token: str = Depends(oauth2_scheme)):
    """Dependency trích xuất user_id từ JWT Token và kiểm tra danh sách đen (Blacklist)."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Không thể xác thực thông tin đăng nhập",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    # KHIẾM KHUYẾT 4: Kiểm tra Token có nằm trong sổ đen (đã Đăng xuất) hay không
    if redis_client.exists(f"blacklist_token:{token}"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Phiên đăng nhập đã kết thúc. Vui lòng đăng nhập lại.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("user_id")
        if user_id is None:
            raise credentials_exception
        return user_id
    except JWTError:
        raise credentials_exception
