import pytest
from sqlmodel import select
from app.models.job import Job, JobStatus
from app.models.transcript import Transcript

def test_auth_flow(client):
    # 1. Test đăng ký
    res_register = client.post("/api/auth/register", json={"username": "testuser", "email": "test@user.com", "password": "password123"})
    assert res_register.status_code == 200
    assert res_register.json()["username"] == "testuser"
    
    # 2. Test đăng nhập
    res_login = client.post("/api/auth/login", data={"username": "testuser", "password": "password123"})
    assert res_login.status_code == 200
    token = res_login.json()["access_token"]
    assert token is not None

@pytest.fixture
def auth_token(client):
    client.post("/api/auth/register", json={"username": "user2", "email": "user2@test.com", "password": "password123"})
    res = client.post("/api/auth/login", data={"username": "user2", "password": "password123"})
    return res.json()["access_token"]

def test_upload_and_rename_job(client, auth_token, session):
    headers = {"Authorization": f"Bearer {auth_token}"}
    
    # 1. Upload mock file
    with open(__file__, "rb") as f:
        res = client.post(
            "/api/audio/upload", 
            headers=headers,
            data={"num_speakers": 2},
            files={"file": ("test.wav", f, "audio/wav")}
        )
    assert res.status_code == 200
    job_id = res.json()["job_id"]
    
    # 2. Rename Job
    res_rename = client.put(
        f"/api/audio/jobs/{job_id}/rename_job",
        headers=headers,
        json={"new_filename": "Họp chiến lược"}
    )
    assert res_rename.status_code == 200
    
    # Verify in database
    job = session.get(Job, job_id)
    assert job.filename == "Họp chiến lược"

def test_history_jobs(client, auth_token):
    headers = {"Authorization": f"Bearer {auth_token}"}
    res = client.get("/api/audio/jobs", headers=headers)
    assert res.status_code == 200
    assert isinstance(res.json()["data"], list)

def test_rename_speaker(client, auth_token, session):
    headers = {"Authorization": f"Bearer {auth_token}"}
    
    # Setup dummy job and transcript in DB
    from app.models.user import User
    user = session.exec(select(User).where(User.username == "user2")).first()
    
    job = Job(user_id=user.id, filename="test_speaker.wav", num_speakers=2, file_path="dummy", status=JobStatus.COMPLETED)
    session.add(job)
    session.commit()
    
    t1 = Transcript(job_id=job.id, start_time=0.0, end_time=1.0, speaker="SPEAKER_00", text="Hello")
    session.add(t1)
    session.commit()
    
    # Rename API
    res = client.put(
        f"/api/audio/jobs/{job.id}/rename_speaker",
        headers=headers,
        json={"old_name": "SPEAKER_00", "new_name": "John Doe"}
    )
    assert res.status_code == 200
    
    session.refresh(t1)
    assert t1.speaker == "John Doe"

def test_summary_and_chat(client, auth_token, session):
    headers = {"Authorization": f"Bearer {auth_token}"}
    from app.models.user import User
    user = session.exec(select(User).where(User.username == "user2")).first()
    job = Job(user_id=user.id, filename="dummy", num_speakers=1, file_path="dummy", status=JobStatus.COMPLETED)
    session.add(job)
    session.commit()
    
    t1 = Transcript(job_id=job.id, start_time=0.0, end_time=1.0, speaker="SPEAKER_00", text="Hello")
    session.add(t1)
    session.commit()
    
    # Try fetching summary
    res_summary = client.get(f"/api/audio/jobs/{job.id}/summary", headers=headers)
    assert res_summary.status_code == 200
    
    # Test chat with mocked llama
    res_chat = client.post(
        f"/api/audio/chat/{job.id}",
        headers=headers,
        json={"message": "Tóm tắt giúp tôi"}
    )
    assert res_chat.status_code == 200
    assert "Đây là câu trả lời mock từ Qwen" in res_chat.json()["data"]
