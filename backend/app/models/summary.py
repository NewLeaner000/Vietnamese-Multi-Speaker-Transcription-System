from typing import Optional, List, Dict, Any
from sqlmodel import Field, SQLModel
from sqlalchemy import Column, JSON

class Summary(SQLModel, table=True):
    """
    Bảng Summary: Lưu trữ kết quả tóm tắt cuối cùng từ Qwen LLM.
    """
    id: Optional[int] = Field(default=None, primary_key=True)
    
    # Khóa ngoại trỏ về Bảng Job. Đánh dấu unique=True vì 1 Job chỉ có duy nhất 1 bản tóm tắt (Quan hệ 1-1).
    job_id: int = Field(foreign_key="job.id", index=True, unique=True)
    
    # Tổng quan cuộc họp (Dạng text thường)
    meeting_overview: Optional[str] = None
    
    # Vì Action Items và Segments là dạng danh sách cấu trúc phức tạp, 
    # ta dùng kiểu dữ liệu JSON của PostgreSQL để lưu trữ cho linh hoạt.
    action_items: List[Dict[str, Any]] = Field(default=[], sa_column=Column(JSON))
    segments: List[Dict[str, Any]] = Field(default=[], sa_column=Column(JSON))
    speaker_insights: List[Dict[str, Any]] = Field(default=[], sa_column=Column(JSON))
