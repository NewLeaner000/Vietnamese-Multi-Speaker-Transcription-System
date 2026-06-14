from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from sqlmodel import SQLModel, text
from app.db.database import create_db_and_tables, engine

# RẤT QUAN TRỌNG: Bạn phải import (gọi) các file Model vào đây. 
# Nếu không import, SQLModel sẽ không biết sự tồn tại của bảng User để ra lệnh tạo bảng.
from app.models.user import User
from app.models.job import Job
from app.models.transcript import Transcript
from app.models.summary import Summary

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Hàm Lifespan quản lý vòng đời của Ứng dụng.
    Đoạn code trước 'yield' sẽ chạy đúng 1 lần khi bạn vừa bật Server lên.
    """
    print(" Dang ket noi PostgreSQL va khoi tao cac Bang (Tables)...")
    create_db_and_tables()
    
    # Tự động bật RLS (Row Level Security) cho toàn bộ các bảng 
    # Điều này giúp dập tắt vĩnh viễn các cảnh báo (Advisor) khó chịu trên trang chủ Supabase
    try:
        with engine.begin() as conn:
            from sqlmodel import SQLModel
            for table_name in SQLModel.metadata.tables.keys():
                conn.execute(text(f'ALTER TABLE "{table_name}" ENABLE ROW LEVEL SECURITY;'))
    except Exception as e:
        print("Migration note (RLS):", e)

    try:
        with engine.begin() as conn:
            conn.execute(text('ALTER TABLE job ADD COLUMN has_enrollment BOOLEAN DEFAULT FALSE'))
    except Exception as e:
        print("Migration note (has_enrollment):", e)
        
    try:
        with engine.begin() as conn:
            conn.execute(text('ALTER TABLE job ADD COLUMN celery_task_id VARCHAR'))
    except Exception as e:
        print("Migration note (celery_task_id):", e)
        print("Migration note:", e)
    print(" Da tao bang thanh cong!")
    import asyncio
    from app.core.cleanup import cleanup_trash_loop
    
    # Khởi động Task dọn rác chạy ngầm
    cleanup_task = asyncio.create_task(cleanup_trash_loop())
    
    yield
    # Đoạn code sau 'yield' sẽ chạy khi bạn tắt Server (Ctrl+C)
    cleanup_task.cancel()
    print(" Đã ngắt kết nối Server.")

# Khởi tạo API Server
app = FastAPI(
    title="Antigravity API",
    description="Backend API cho hệ thống Meeting Transcription",
    version="1.0.0",
    lifespan=lifespan
)

from fastapi.middleware.cors import CORSMiddleware

# Cấu hình CORS cho phép Frontend (Vite chạy cổng 5173) gọi API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

from app.api import auth, upload, summary
app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
app.include_router(upload.router, prefix="/api/audio", tags=["Audio Processing"])
app.include_router(summary.router, prefix="/api/audio", tags=["AI Summary"])

# Đã xóa cấu hình phục vụ file Static (Âm thanh) qua /uploads để ngăn rò rỉ IDOR
import os
os.makedirs("uploads", exist_ok=True)

@app.get("/")
def read_root():
    return {
        "status": "success",
        "message": "Backend Server đang chạy hoàn hảo! Hãy thêm '/docs' vào cuối URL để mở giao diện Swagger UI."
    }
