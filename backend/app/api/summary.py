from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel
from sqlmodel import Session, select
import os
from typing import List

from app.db.database import get_session
from app.models.job import Job
from app.models.transcript import Transcript
from app.models.summary import Summary

router = APIRouter()



@router.post("/summarize/{job_id}")
def trigger_summarize(job_id: int):
    """API Kích hoạt Qwen Tóm tắt. Trả về ngay lập tức để giao diện không bị treo."""
    from app.db.database import engine
    with Session(engine) as session:
        job = session.get(Job, job_id)
        if not job:
            raise HTTPException(status_code=404, detail="Job không tồn tại")
            
        transcripts = session.exec(select(Transcript).where(Transcript.job_id == job_id)).all()
        if not transcripts:
            raise HTTPException(status_code=400, detail="Job này chưa có dữ liệu hội thoại. Hãy bóc băng lại!")
            
        # Xóa bản tóm tắt cũ nếu user muốn chạy lại
        summary = session.exec(select(Summary).where(Summary.job_id == job_id)).first()
        if summary:
            session.delete(summary)
            session.commit()
        
    # Gọi Celery Task để chạy trên máy local (nơi có GPU)
    from app.worker.tasks_ai import summarize_audio_task
    task = summarize_audio_task.delay(job_id)
    
    # Đứng chờ (block) tối đa 120 giây để lấy kết quả
    try:
        task.get(timeout=120)
    except Exception as e:
        print(f"Summary timeout or error: {e}")
        
    return {"message": "Đã tóm tắt thành công!"}

@router.get("/jobs/{job_id}/summary")
def get_job_summary(job_id: int, session: Session = Depends(get_session)):
    """API để Frontend lấy bản Tóm tắt (Nếu Qwen đã chạy xong)"""
    summary = session.exec(select(Summary).where(Summary.job_id == job_id)).first()
    if not summary:
        return {"status": "pending", "data": None}
    return {"status": "completed", "data": summary.meeting_overview}

@router.get("/jobs/{job_id}/transcripts")
def get_job_transcripts(job_id: int, session: Session = Depends(get_session)):
    """API Trả về danh sách chi tiết các câu thoại của cuộc họp"""
    transcripts = session.exec(select(Transcript).where(Transcript.job_id == job_id).order_by(Transcript.start_time)).all()
    if not transcripts:
        return {"data": []}
        
    results = []
    for t in transcripts:
        results.append({
            "speaker": t.speaker,
            "start": round(t.start_time, 2),
            "end": round(t.end_time, 2),
            "text": t.text
        })
        
    return {"data": results}

class RenameSpeakerRequest(BaseModel):
    old_name: str
    new_name: str

@router.put("/jobs/{job_id}/rename_speaker")
def rename_speaker(job_id: int, req: RenameSpeakerRequest, session: Session = Depends(get_session)):
    """API để đổi tên một người nói trên toàn bộ transcript của một job"""
    transcripts = session.exec(select(Transcript).where(Transcript.job_id == job_id, Transcript.speaker == req.old_name)).all()
    if not transcripts:
        raise HTTPException(status_code=404, detail="Không tìm thấy người nói này trong dữ liệu")
    for t in transcripts:
        t.speaker = req.new_name
    session.commit()
    return {"message": f"Đã đổi tên {req.old_name} thành {req.new_name}"}

class RenameJobRequest(BaseModel):
    new_filename: str

@router.put("/jobs/{job_id}/rename_job")
def rename_job(job_id: int, req: RenameJobRequest, session: Session = Depends(get_session)):
    """API để đổi tên hiển thị của cuộc họp (Job)"""
    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Không tìm thấy Job này")
    job.filename = req.new_filename
    session.commit()
    return {"message": "Đã đổi tên cuộc họp thành công", "new_filename": job.filename}

class ChatRequest(BaseModel):
    message: str

@router.post("/chat/{job_id}")
def chat_with_meeting(job_id: int, req: ChatRequest):
    """API Chat trực tiếp với Qwen sử dụng Context là Transcript và Persona là AI Career Coach"""
    from app.db.database import engine
    with Session(engine) as session:
        # 1. Load Transcript
        transcripts = session.exec(select(Transcript).where(Transcript.job_id == job_id).order_by(Transcript.start_time)).all()
        if not transcripts:
            raise HTTPException(status_code=400, detail="Chưa có dữ liệu hội thoại.")
            
        conversation = ""
        for t in transcripts:
            conversation += f"[{t.speaker}]: {t.text}\n"

    # 2. Đọc Hệ tư tưởng (System Persona) từ file ani_assistant_skill.md
    skill_file_path = os.path.join(os.path.dirname(__file__), "../../../ani_assistant_skill.md")
    system_prompt = "Bạn là AI hỗ trợ phân tích cuộc họp."
    if os.path.exists(skill_file_path):
        with open(skill_file_path, "r", encoding="utf-8") as f:
            system_prompt = f.read()

    # 3. Tạo Prompt Chat
    user_prompt = f"""Dưới đây là nội dung cuộc họp:
---
{conversation}
---
Dựa vào cuộc họp trên và vai trò của bạn, hãy trả lời câu hỏi sau của tôi:
{req.message}
"""

    # 4. Gọi Celery Task để chạy trên máy local (nơi có GPU)
    from app.worker.tasks_ai import chat_with_qwen_task
    task = chat_with_qwen_task.delay(job_id, req.message, system_prompt)
    
    # Đứng chờ (block) tối đa 120 giây để lấy kết quả
    try:
        answer = task.get(timeout=120)
        return {"data": answer}
    except Exception as e:
        print(f"[CHAT ERROR] Timeout or error: {e}")
        raise HTTPException(status_code=500, detail="Lỗi khi phản hồi Chat")

