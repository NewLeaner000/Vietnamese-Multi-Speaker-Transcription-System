from typing import Optional
from sqlmodel import Field, SQLModel

class Transcript(SQLModel, table=True):
    """
    Bảng Transcript: Lưu trữ kết quả bóc băng (ASR) và nhận diện người nói (Diarization).
    Thay vì lưu thành 1 cục text khổng lồ, ta tách ra thành từng dòng để làm được tính năng
    "Tìm kiếm đoạn hội thoại" hoặc RAG Chatbot cực kỳ nhanh và chính xác.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Khóa ngoại trỏ về Bảng Job.
    # Mỗi Job sẽ có hàng ngàn dòng Transcript. Mối quan hệ 1-N (One-to-Many).
    job_id: int = Field(foreign_key="job.id", index=True)
    
    # Thời gian bắt đầu và kết thúc của câu nói (dùng để làm tính năng Click vào text tua video)
    start_time: float
    end_time: float
    
    # Người nói (VD: SPEAKER_00)
    speaker: str
    
    # Nội dung câu nói
    text: str
