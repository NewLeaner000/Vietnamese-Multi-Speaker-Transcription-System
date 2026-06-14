import pytest
from fastapi.testclient import TestClient
from sqlmodel import SQLModel, Session, create_engine
from sqlmodel.pool import StaticPool
import os

from app.main import app
from app.db.database import get_session
from app.models.user import User

# Sử dụng in-memory database của SQLite để test nhanh và dọn dẹp dễ dàng
DATABASE_URL = "sqlite:///:memory:"
engine = create_engine(
    DATABASE_URL, 
    connect_args={"check_same_thread": False}, 
    poolclass=StaticPool
)

def get_session_override():
    with Session(engine) as session:
        yield session

# Ghi đè hàm kết nối DB mặc định thành hàm Override của Test
app.dependency_overrides[get_session] = get_session_override

@pytest.fixture(name="session")
def session_fixture():
    SQLModel.metadata.create_all(engine)
    with Session(engine) as session:
        yield session
    SQLModel.metadata.drop_all(engine)

@pytest.fixture(name="client")
def client_fixture(session: Session):
    return TestClient(app)

@pytest.fixture(autouse=True)
def mock_heavy_tasks(monkeypatch):
    """
    Mock Celery task và BackgroundTask để không thực sự chạy GPU model trong lúc test.
    """
    class MockAsyncResult:
        def __init__(self, id):
            self.id = id

    def mock_process_delay(*args, **kwargs):
        return MockAsyncResult("test_celery_task_id")
        
    def mock_qwen_summary(*args, **kwargs):
        pass # Không làm gì cả

    # Mock Llama cho phần Chat
    class MockLlama:
        def __init__(self, *args, **kwargs):
            pass
        def create_chat_completion(self, *args, **kwargs):
            return {
                "choices": [
                    {
                        "message": {
                            "content": "Đây là câu trả lời mock từ Qwen."
                        }
                    }
                ]
            }
            
    # Xử lý trường hợp không có llama_cpp khi test
    import sys
    from unittest.mock import MagicMock
    if "llama_cpp" not in sys.modules:
        sys.modules["llama_cpp"] = MagicMock()
    sys.modules["llama_cpp"].Llama = MockLlama

    # Patch process_audio_task.delay trong upload.py
    monkeypatch.setattr("app.api.upload.process_audio_task.delay", mock_process_delay)
    
    # Patch run_qwen_summary_task trong summary.py
    monkeypatch.setattr("app.api.summary.run_qwen_summary_task", mock_qwen_summary)

