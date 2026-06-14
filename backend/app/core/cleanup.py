import asyncio
import os
import shutil
from datetime import datetime, timedelta
from sqlmodel import Session, select
from app.db.database import engine
from app.models.job import Job
from app.models.transcript import Transcript
from app.models.summary import Summary

async def cleanup_trash_loop():
    """
    Vòng lặp chạy ngầm trong FastAPI để dọn rác tự động.
    Cứ mỗi 1 giờ sẽ thức dậy quét Database 1 lần.
    Xóa tất cả các file đã nằm trong thùng rác quá 3 ngày.
    """
    while True:
        try:
            with Session(engine) as session:
                three_days_ago = datetime.utcnow() - timedelta(days=3)
                expired_jobs = session.exec(
                    select(Job).where(
                        Job.is_deleted == True,
                        Job.deleted_at < three_days_ago
                    )
                ).all()
                
                if expired_jobs:
                    print(f" [Auto-Cleanup] Found {len(expired_jobs)} expired jobs in trash. Deleting...")
                    for job in expired_jobs:
                        # Xóa Transcript và Summary
                        transcripts = session.exec(select(Transcript).where(Transcript.job_id == job.id)).all()
                        for t in transcripts:
                            session.delete(t)
                        summary = session.exec(select(Summary).where(Summary.job_id == job.id)).first()
                        if summary:
                            session.delete(summary)
                            
                        # Xóa Job khỏi DB
                        session.delete(job)
                        
                        # Xóa file vật lý
                        try:
                            if os.path.exists(job.file_path):
                                os.remove(job.file_path)
                            output_dir = os.path.join(os.getcwd(), "app", "ai_core", "output", f"job_{job.id}")
                            if os.path.exists(output_dir):
                                shutil.rmtree(output_dir)
                            enrollment_dir = os.path.join(os.getcwd(), "uploads", f"enrollment_{job.id}")
                            if os.path.exists(enrollment_dir):
                                shutil.rmtree(enrollment_dir)
                        except Exception as e:
                            print(f"Warning: Failed to delete physical files for job {job.id}: {e}")
                    
                    session.commit()
                    print(" [Auto-Cleanup] Done.")
                    
                # 2. Tự động xoá Audio của các Job đã tạo quá 30 ngày để tiết kiệm dung lượng
                thirty_days_ago = datetime.utcnow() - timedelta(days=30)
                old_jobs = session.exec(
                    select(Job).where(
                        Job.is_deleted == False,
                        Job.created_at < thirty_days_ago
                    )
                ).all()
                
                if old_jobs:
                    deleted_count = 0
                    for job in old_jobs:
                        if job.file_path and os.path.exists(job.file_path):
                            try:
                                os.remove(job.file_path)
                                deleted_count += 1
                            except Exception as e:
                                print(f"Warning: Failed to delete audio for old job {job.id}: {e}")
                    if deleted_count > 0:
                        print(f" [Auto-Cleanup] Deleted {deleted_count} audio files older than 30 days to save space.")
                        
        except Exception as e:
            print(f" [Auto-Cleanup] Error: {e}")
            
        # Ngủ 1 giờ (3600 giây) trước khi quét lại
        await asyncio.sleep(3600)
