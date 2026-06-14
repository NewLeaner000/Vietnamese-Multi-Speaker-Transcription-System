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

def run_qwen_summary_task(job_id: int):
    """
    Task chạy ngầm để gọi Llama-cpp (Qwen).
    Vì Qwen chiếm nhiều RAM, nên chạy riêng như thế này sẽ an toàn hơn
    và không làm gián đoạn FastAPI Server.
    """
    from app.db.database import engine
    from app.ai_core.pipeline_config import QWEN_GGUF_MODEL_DEFAULT
    
    with Session(engine) as session:
        # 1. Rút Transcript
        transcripts = session.exec(select(Transcript).where(Transcript.job_id == job_id).order_by(Transcript.start_time)).all()
        if not transcripts:
            return
            
        conversation = ""
        for t in transcripts:
            # Ráp lại: [unknown_1]: Chào mọi người.
            conversation += f"[{t.speaker}]: {t.text}\n"
            
        # 2. Xây dựng Prompt chuẩn MLOps
        prompt = f"""Bạn là một trợ lý AI chuyên nghiệp có nhiệm vụ tóm tắt biên bản cuộc họp.
Hãy đọc đoạn hội thoại sau và tóm tắt lại bằng tiếng Việt theo đúng 3 mục sau:
1. Khái quát cuộc họp: (Cuộc họp nói về chủ đề gì, mục đích chung là gì)
2. Các quyết định đã chốt: (Những gì mọi người đã đồng ý và chốt lại)
3. Việc cần làm của từng người: (Liệt kê rõ tên người và công việc tương ứng)

--- ĐOẠN HỘI THOẠI ---
{conversation}
"""
        try:
            import sys
            import os
            from pathlib import Path
            if sys.platform == "win32":
                torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
                if torch_lib.exists():
                    os.add_dll_directory(str(torch_lib))
                    os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")
            
            from llama_cpp import Llama
            print(f"[QWEN] Đang nạp mô hình LLM lên GPU...")
            
            # Load model (Sẽ ngốn khoảng 4-5GB VRAM)
            model = Llama(
                model_path=QWEN_GGUF_MODEL_DEFAULT,
                n_gpu_layers=-1, # Dùng tối đa GPU
                n_ctx=8192,      # Ngữ cảnh 8K token để chứa vừa file transcript dài
                verbose=False
            )
            
            print(f"[QWEN] Đang suy luận tóm tắt cho Job {job_id}...")
            response = model.create_chat_completion(
                messages=[
                    {"role": "system", "content": "Bạn là AI tóm tắt văn bản chuyên nghiệp, luôn trả lời đúng format yêu cầu."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024,
                temperature=0.1
            )
            
            result_text = response["choices"][0]["message"]["content"].strip()
            print(f"[QWEN] Suy luận thành công!")
            
            # Xóa model để giải phóng VRAM ngay lập tức!
            del model
            
            # 3. Lưu vào Database
            summary = session.exec(select(Summary).where(Summary.job_id == job_id)).first()
            if not summary:
                # Tạm thời ném toàn bộ chữ vào meeting_overview
                summary = Summary(job_id=job_id, meeting_overview=result_text)
                session.add(summary)
            else:
                summary.meeting_overview = result_text
                
            session.commit()
            
        except Exception as e:
            print(f"[QWEN ERROR] Lỗi khi chạy tóm tắt: {e}")

@router.post("/summarize/{job_id}")
def trigger_summarize(job_id: int, background_tasks: BackgroundTasks, session: Session = Depends(get_session)):
    """API Kích hoạt Qwen Tóm tắt. Trả về ngay lập tức để giao diện không bị treo."""
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
        
    background_tasks.add_task(run_qwen_summary_task, job_id)
    return {"message": "Đã ra lệnh cho Qwen tóm tắt ngầm. Vui lòng chờ vài phút, Qwen đang nạp vào GPU..."}

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
def chat_with_meeting(job_id: int, req: ChatRequest, session: Session = Depends(get_session)):
    """API Chat trực tiếp với Qwen sử dụng Context là Transcript và Persona là AI Career Coach"""
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

    # 4. Nạp Qwen và Sinh câu trả lời (Sync)
    from app.ai_core.pipeline_config import QWEN_GGUF_MODEL_DEFAULT
    try:
        import sys
        from pathlib import Path
        if sys.platform == "win32":
            torch_lib = Path(sys.prefix) / "Lib" / "site-packages" / "torch" / "lib"
            if torch_lib.exists():
                os.add_dll_directory(str(torch_lib))
                os.environ["PATH"] = str(torch_lib) + os.pathsep + os.environ.get("PATH", "")
        
        from llama_cpp import Llama
        print("[CHAT] Đang nạp Qwen để Chat...")
        model = Llama(
            model_path=QWEN_GGUF_MODEL_DEFAULT,
            n_gpu_layers=-1,
            n_ctx=8192,
            verbose=False
        )
        
        response = model.create_chat_completion(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=1024,
            temperature=0.3
        )
        
        answer = response["choices"][0]["message"]["content"].strip()
        del model # Xóa ngay lập tức để giải phóng VRAM!
        
        return {"data": answer}
        
    except Exception as e:
        print(f"[CHAT ERROR] {e}")
        raise HTTPException(status_code=500, detail=str(e))

