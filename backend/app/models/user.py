from typing import Optional
from sqlmodel import Field, SQLModel
from pydantic import EmailStr

class User(SQLModel, table=True):
    """
    Tham số `table=True`: Dòng này báo cho SQLModel biết class này không chỉ dùng để
    kiểm tra tính hợp lệ của dữ liệu đầu vào (Validation), mà nó còn là 1 Bảng (Table) 
    thực sự cần được tạo ra bên trong PostgreSQL.
    Tên bảng tự động sẽ được đặt là 'user'.
    """
    
    # 1. Khóa chính (Primary Key), tự động tăng ID từ 1, 2, 3... khi có user mới
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # 2. Username bắt buộc phải có, duy nhất (unique) và đánh index để tìm kiếm cực nhanh
    username: str = Field(index=True, unique=True)
    
    # 3. Email: dùng EmailStr của Pydantic để tự động báo lỗi API ngay lập tức 
    # nếu user nhập sai định dạng email (VD: abc@ thay vì abc@gmail.com)
    email: EmailStr = Field(unique=True, index=True)
    
    # 4. Mật khẩu sau khi đã được mã hoá (băm - hash). 
    # NGUYÊN TẮC VÀNG: KHÔNG BAO GIỜ lưu mật khẩu gốc (mật khẩu có thể đọc được) vào database!
    hashed_password: str
    
    # 5. Cờ trạng thái: Tài khoản có đang hoạt động hay không (Dùng để khóa acc nêú cần)
    is_active: bool = Field(default=True)
