from sqlmodel import SQLModel, create_engine, Session
from app.core.config import settings

# 1. Tạo Engine (Cỗ máy kết nối)
# Engine đóng vai trò như một "Đường ống mạng" kết nối liên tục từ ứng dụng Python tới máy chủ PostgreSQL (Docker)
engine = create_engine(
    settings.SQLALCHEMY_DATABASE_URI, 
    echo=True # echo=True giúp in ra log các câu lệnh SQL gốc để bạn dễ dàng học hỏi và sửa lỗi
)

# 2. Hàm khởi tạo cấu trúc Bảng (Tables)
def create_db_and_tables():
    """
    Hàm này sẽ quét tất cả các Models (SQLModel) mà bạn định nghĩa trong tương lai
    và tự động ra lệnh cho PostgreSQL tạo các bảng trống tương ứng nếu chúng chưa tồn tại.
    """
    SQLModel.metadata.create_all(engine)

# 3. Hàm cấp phát Phiên làm việc (Session Generator)
def get_session():
    """
    Session giống như một 'Giao dịch viên' tại ngân hàng. 
    Mỗi khi có 1 Request (Yêu cầu API) gửi đến, FastAPI sẽ lấy 1 Session để đọc/ghi Database, 
    xử lý xong thì tự động đóng Session lại để tối ưu bộ nhớ.
    Đây là tiêu chuẩn Dependency Injection cực kỳ mạnh mẽ của FastAPI.
    """
    with Session(engine) as session:
        yield session
