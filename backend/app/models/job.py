from typing import Optional
from sqlmodel import Field, SQLModel
from datetime import datetime
from enum import Enum

# Khai báo kiểu dữ liệu Enum để giới hạn trạng thái của Job
class JobStatus(str, Enum):
    PENDING = "pending"       # Đang chờ trong hàng đợi Celery
    PROCESSING = "processing" # Đang được GPU xử lý (Whisper/Pyannote)
    COMPLETED = "completed"   # Xử lý xong thành công
    FAILED = "failed"         # Bị lỗi (Tràn RAM, file lỗi...)

class Job(SQLModel, table=True):
    """
    Bảng Job: Trái tim của hệ thống MLOps. 
    Nó giúp theo dõi vòng đời của một file âm thanh từ lúc người dùng tải lên cho đến khi xử lý xong.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Khóa ngoại (Foreign Key) trỏ tới ID của Bảng User.
    # Cho biết file âm thanh này thuộc về ai, giúp bảo mật dữ liệu không bị xem trộm.
    user_id: int = Field(foreign_key="user.id", index=True)
    
    # Tên gốc của file âm thanh người dùng tải lên
    filename: str
    
    # Số lượng người nói trong đoạn hội thoại (User bắt buộc phải nhập)
    num_speakers: int
    
    # Đường dẫn lưu file âm thanh tạm thời trên ổ cứng server
    file_path: str
    
    # Trạng thái hiện tại của tiến trình
    status: JobStatus = Field(default=JobStatus.PENDING)
    
    # Có upload file enrollment (mẫu giọng nói) hay không
    has_enrollment: bool = Field(default=False)
    
    # Ghi chú lỗi (nếu status = FAILED thì ghi nguyên nhân vào đây)
    error_message: Optional[str] = None
    
    # Theo dõi thời gian (Rất quan trọng để đo lường hiệu suất mô hình)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: Optional[datetime] = None

    # Soft Delete (Thùng rác)
    is_deleted: bool = Field(default=False)
    deleted_at: Optional[datetime] = None
