from pydantic import BaseModel, EmailStr

# Tại sao phải tạo Schema (Pydantic) riêng thay vì dùng luôn SQLModel?
# => BỞI VÌ TÍNH BẢO MẬT: Dữ liệu người dùng gửi lên API (password thô) hoàn toàn khác với 
# cấu trúc lưu trong Database (hashed_password).

class UserCreate(BaseModel):
    """
    Schema hứng dữ liệu từ giao diện Next.js gửi lên cổng đăng ký API (/register).
    User sẽ gõ 'password' chứ họ không có 'hashed_password'.
    """
    username: str
    email: EmailStr
    password: str
    verification_code: str

class VerificationRequest(BaseModel):
    """
    Schema nhận email từ giao diện để gửi mã OTP.
    """
    email: EmailStr

class ResetPasswordRequest(BaseModel):
    email: EmailStr
    verification_code: str
    new_password: str

class GoogleLoginRequest(BaseModel):
    credential: str

from pydantic import ConfigDict

class UserResponse(BaseModel):
    """
    Schema trả dữ liệu về cho Frontend hiển thị.
    NGUYÊN TẮC VÀNG CỦA SENIOR ENGINEER: Dùng schema này để "lọc bỏ" cột hashed_password,
    tuyệt đối không để lọt mật khẩu (dù đã mã hoá) ra mạng Internet.
    """
    model_config = ConfigDict(from_attributes=True) # Rất quan trọng để FastAPI hiểu dữ liệu từ Database (ORM)
    
    id: int
    username: str
    email: EmailStr
    is_active: bool
