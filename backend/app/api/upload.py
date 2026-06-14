import os
import zipfile
from typing import List, Optional
from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException
from fastapi.responses import FileResponse, RedirectResponse
from sqlmodel import Session
from app.db.database import get_session
from app.models.job import Job, JobStatus
from app.worker.tasks import process_audio_task
from app.core.security import get_current_user
from app.core.storage import upload_file_to_supabase

router = APIRouter()

# Tạo thư mục chứa âm thanh trên ổ cứng
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@router.post("/upload")
async def upload_audio(
    file: UploadFile = File(...), 
    num_speakers: int = Form(...),
    enrollment_files: Optional[List[UploadFile]] = File(None),
    session: Session = Depends(get_session),
    current_user_id: int = Depends(get_current_user)
):
    """
    API Nhận file âm thanh từ User. Bắt buộc nhập số người nói.
    """
    # 1. Tải và lưu file xuống ổ cứng Server
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
        
    final_filename = file.filename
    
    # --- TỐI ƯU DUNG LƯỢNG: NÉN .WAV SANG .MP3 ---
    if file.filename.lower().endswith('.wav'):
        try:
            import soundfile as sf
            base_name = os.path.splitext(file.filename)[0]
            compressed_filename = f"{base_name}.mp3"
            compressed_file_path = os.path.join(UPLOAD_DIR, compressed_filename)
            
            data, samplerate = sf.read(file_path)
            sf.write(compressed_file_path, data, samplerate)
            
            os.remove(file_path)
            
            file_path = compressed_file_path
            final_filename = compressed_filename
        except Exception as e:
            print(f"Lỗi khi nén file âm thanh: {e}")
            
    # 2. Tạo Phiếu báo danh (Job) trong Database (để lấy ID trước)
    new_job = Job(
        user_id=current_user_id, 
        filename=final_filename,
        num_speakers=num_speakers,
        file_path="uploading...", # Tạm thời
        status=JobStatus.PENDING,
        has_enrollment=False
    )
    session.add(new_job)
    session.commit()
    session.refresh(new_job)

    # 3. Tải file âm thanh chính lên Supabase Storage
    try:
        remote_path = f"jobs/{new_job.id}/{final_filename}"
        public_url = upload_file_to_supabase(file_path, remote_path)
        new_job.file_path = public_url
        os.remove(file_path) # Xóa file local sau khi upload
    except Exception as e:
        session.delete(new_job)
        session.commit()
        raise HTTPException(status_code=500, detail=f"Lỗi upload file âm thanh: {str(e)}")

    # 4. Xử lý file Enrollment nếu có
    if enrollment_files and len(enrollment_files) > 0 and enrollment_files[0].filename != "":
        enrollment_dir = os.path.join(UPLOAD_DIR, f"enrollment_{new_job.id}")
        os.makedirs(enrollment_dir, exist_ok=True)
        
        # Lưu file vào thư mục local tạm thời
        for e_file in enrollment_files:
            if e_file.filename.endswith(".zip"):
                zip_path = os.path.join(enrollment_dir, e_file.filename)
                with open(zip_path, "wb") as f:
                    f.write(await e_file.read())
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(enrollment_dir)
                os.remove(zip_path) # Clean up zip after extracting
            else:
                wav_path = os.path.join(enrollment_dir, e_file.filename)
                with open(wav_path, "wb") as f:
                    f.write(await e_file.read())
        
        # Nén toàn bộ thư mục thành 1 file zip để up lên mây
        final_zip_path = os.path.join(UPLOAD_DIR, f"enrollment_{new_job.id}.zip")
        with zipfile.ZipFile(final_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(enrollment_dir):
                for file in files:
                    zipf.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), enrollment_dir))
        
        # Upload file zip lên Supabase
        upload_file_to_supabase(final_zip_path, f"jobs/{new_job.id}/enrollment.zip")
        
        # Dọn rác local
        os.remove(final_zip_path)
        import shutil
        shutil.rmtree(enrollment_dir)
        
        new_job.has_enrollment = True
        
    session.commit()
    
    # 4. Bấm chuông gọi Celery Worker gắp Job này vào hàng đợi (Redis)
    # Lệnh '.delay()' chính là chìa khóa của MLOps. Nó đẩy lệnh đi và thoát luôn, không chờ đợi.
    task = process_audio_task.delay(new_job.id)
    
    # 4. Trả kết quả về ngay lập tức cho User (Thời gian phản hồi < 0.1 giây)
    return {
        "message": "Đã đẩy file vào hàng chờ AI thành công!",
        "job_id": new_job.id,
        "celery_task_id": task.id
    }
from datetime import datetime
import shutil
from app.models.transcript import Transcript
from app.models.summary import Summary
from sqlmodel import select

@router.get("/jobs/trash")
def get_trash_jobs(session: Session = Depends(get_session), current_user_id: int = Depends(get_current_user)):
    """Lấy danh sách các file trong thùng rác"""
    jobs = session.exec(
        select(Job).where(
            Job.user_id == current_user_id,
            Job.is_deleted == True
        ).order_by(Job.deleted_at.desc())
    ).all()
    return {"data": jobs}

@router.get("/jobs/{job_id}")
def get_job_status(job_id: int, session: Session = Depends(get_session)):
    """API để Frontend (Next.js) gọi liên tục mỗi giây để vẽ Thanh Tiến Độ"""
    job = session.get(Job, job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job không tồn tại")
    
    return {
        "job_id": job.id,
        "filename": job.filename,
        "status": job.status,
        "error": job.error_message
    }

from sqlmodel import select

@router.get("/jobs")
def get_all_jobs(session: Session = Depends(get_session), current_user_id: int = Depends(get_current_user)):
    """Lấy danh sách toàn bộ Lịch sử Job của User đang đăng nhập"""
    jobs = session.exec(
        select(Job).where(
            Job.user_id == current_user_id,
            Job.is_deleted == False
        ).order_by(Job.created_at.desc())
    ).all()
    return {"data": jobs}



@router.post("/jobs/{job_id}/trash")
def move_to_trash(job_id: int, session: Session = Depends(get_session), current_user_id: int = Depends(get_current_user)):
    """Đưa 1 job vào thùng rác (Soft delete)"""
    job = session.get(Job, job_id)
    if not job or job.user_id != current_user_id:
        raise HTTPException(status_code=404, detail="Không tìm thấy file")
    
    job.is_deleted = True
    job.deleted_at = datetime.utcnow()
    session.commit()
    return {"message": "Đã đổi tên file thành công!"}

from jose import jwt, JWTError
from app.core.security import SECRET_KEY, ALGORITHM

@router.get("/jobs/{job_id}/audio")
def get_job_audio(job_id: int, token: str, session: Session = Depends(get_session)):
    """Trả về file âm thanh (hỗ trợ HTTP Range Request để Frontend có thể tua được)"""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        current_user_id = payload.get("user_id")
        if current_user_id is None:
            raise HTTPException(status_code=401, detail="Token không hợp lệ")
    except Exception:
        raise HTTPException(status_code=401, detail="Chưa đăng nhập")

    job = session.get(Job, job_id)
    if not job or job.user_id != current_user_id:
        raise HTTPException(status_code=404, detail="Không tìm thấy thông tin Job")
    
    if job.file_path.startswith("http"):
        return RedirectResponse(url=job.file_path)
    
    if not os.path.exists(job.file_path):
        raise HTTPException(status_code=404, detail="File âm thanh vật lý không tồn tại trên ổ cứng")
        
    return FileResponse(job.file_path, media_type="audio/mpeg")

@router.post("/jobs/{job_id}/restore")
def restore_job(job_id: int, session: Session = Depends(get_session), current_user_id: int = Depends(get_current_user)):
    """Khôi phục 1 job từ thùng rác"""
    job = session.get(Job, job_id)
    if not job or job.user_id != current_user_id:
        raise HTTPException(status_code=404, detail="Không tìm thấy file")
    
    job.is_deleted = False
    job.deleted_at = None
    session.commit()
    return {"message": "Đã khôi phục file"}

@router.delete("/jobs/{job_id}")
def hard_delete_job(job_id: int, session: Session = Depends(get_session), current_user_id: int = Depends(get_current_user)):
    """Xóa vĩnh viễn 1 job khỏi DB và xóa file trên ổ cứng"""
    job = session.get(Job, job_id)
    if not job or job.user_id != current_user_id:
        raise HTTPException(status_code=404, detail="Không tìm thấy file")
    
    # 1. Xóa dữ liệu liên quan (Transcript, Summary) để tránh lỗi Foreign Key
    # 1. Xóa dữ liệu liên quan (Transcript, Summary) để tránh lỗi Foreign Key
    transcripts = session.exec(select(Transcript).where(Transcript.job_id == job_id)).all()
    for t in transcripts:
        session.delete(t)
        
    summaries = session.exec(select(Summary).where(Summary.job_id == job_id)).all()
    for s in summaries:
        session.delete(s)
        
    session.commit() # Đẩy lệnh xóa con xuống DB trước
    
    # 2. Xóa Job khỏi database
    session.delete(job)
    session.commit()
    
    # 3. Xóa file vật lý (âm thanh gốc và thư mục output)
    try:
        if os.path.exists(job.file_path):
            os.remove(job.file_path)
            
        output_dir = os.path.join(os.getcwd(), "app", "ai_core", "output", f"job_{job_id}")
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
            
        enrollment_dir = os.path.join(UPLOAD_DIR, f"enrollment_{job_id}")
        if os.path.exists(enrollment_dir):
            shutil.rmtree(enrollment_dir)
    except Exception as e:
        print(f"Warning: Failed to delete physical files for job {job_id}: {e}")
        
    return {"message": "Đã xóa vĩnh viễn"}

from celery.result import AsyncResult
from app.worker.celery_app import celery_app

@router.get("/progress/{task_id}")
def get_task_progress(task_id: str):
    """
    API dùng để soi tiến độ trực tiếp từ Redis.
    Gõ cái 'celery_task_id' (chuỗi dài ngoằng) vào đây, bấm Execute liên tục để xem số % tăng dần!
    """
    task = AsyncResult(task_id, app=celery_app)
    
    if task.state == 'PROGRESS':
        # Đây chính là dữ liệu để Frontend vẽ Thanh màu xanh
        return {"status": "ĐANG CHẠY", "percent": task.info.get("percent", 0)}
    elif task.state == 'SUCCESS':
        return {"status": "HOÀN THÀNH", "percent": 100}
    elif task.state == 'PENDING':
        return {"status": "ĐANG CHỜ TRONG HÀNG ĐỢI", "percent": 0}
    else:
        return {"status": task.state, "percent": 0}
