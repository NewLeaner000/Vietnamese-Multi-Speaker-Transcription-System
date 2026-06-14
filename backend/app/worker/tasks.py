import os
import time
from pathlib import Path
from datetime import datetime
from sqlmodel import Session
from app.worker.celery_app import celery_app
from app.db.database import engine
from app.models.job import Job, JobStatus
from app.models.user import User

# [MLOps] Import Lõi AI từ thư mục ai_core
from app.ai_core.pipeline_config import DER_SCRIPT_PATH, DER_CHECKPOINT_DEFAULT, DEFAULT_OUTPUT_DIR, ASR_CHECKPOINT_DEFAULT
from app.ai_core.der_infer_bridge import run_der_pipeline
from app.ai_core.asr_runner import prepare_engine_input, run_core_asr_engine
from app.ai_core.audio_preprocess_input import normalize_audio_to_mono16k
from app.core.storage import download_file_from_supabase
import zipfile

@celery_app.task(bind=True, name="process_audio")
def process_audio_task(self, job_id: int):
    """
    Hàm này được Celery kích hoạt để chạy AI thật sự!
    """

    with Session(engine) as session:
        job = session.get(Job, job_id)
        if not job:
            return "Job không tồn tại!"
            
        job.status = JobStatus.PROCESSING
        session.commit()
        
        try:
            self.update_state(state='PROGRESS', meta={'percent': 5, 'message': 'Bắt đầu xử lý'})
            
            # --- CHẶNG 0: Thiết lập thư mục và tải file từ Cloud ---
            self.update_state(state='PROGRESS', meta={'percent': 2, 'message': 'Đang tải file từ Đám mây (Supabase)...'})
            output_dir = DEFAULT_OUTPUT_DIR / f"job_{job_id}"
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Tải file âm thanh chính
            original_audio = output_dir / "cloud_downloaded_audio.tmp"
            remote_audio_path = f"jobs/{job_id}/{job.filename}"
            
            # Nếu URL là public (http), ta vẫn dùng Supabase Client tải bằng tên remote path cho chắc chắn
            download_file_from_supabase(remote_audio_path, str(original_audio))
            
            # --- CHẶNG 1: Tiền xử lý (Preprocess) ---
            self.update_state(state='PROGRESS', meta={'percent': 10, 'message': 'Đang chuẩn hóa âm thanh (16kHz Mono)...'})
            mono_audio_path = output_dir / "mono_16k.wav"
            normalize_audio_to_mono16k(str(original_audio), str(mono_audio_path))
            
            # --- CHẶNG 2: Nhận diện giọng nói (Diarization) ---
            self.update_state(state='PROGRESS', meta={'percent': 30, 'message': 'Đang phân biệt giọng nói (Diarization)...'})
            
            # Kiểm tra enrollment
            enrollment_dir = None
            if hasattr(job, 'has_enrollment') and job.has_enrollment:
                self.update_state(state='PROGRESS', meta={'percent': 11, 'message': 'Đang tải file mẫu (Enrollment) từ Đám mây...'})
                enrollment_dir = output_dir / f"enrollment_{job_id}"
                enrollment_dir.mkdir(parents=True, exist_ok=True)
                
                remote_zip_path = f"jobs/{job_id}/enrollment.zip"
                local_zip_path = enrollment_dir / "enrollment.zip"
                
                try:
                    download_file_from_supabase(remote_zip_path, str(local_zip_path))
                    with zipfile.ZipFile(str(local_zip_path), 'r') as zip_ref:
                        zip_ref.extractall(str(enrollment_dir))
                    local_zip_path.unlink() # Xóa file zip local
                except Exception as e:
                    print(f"Lỗi khi tải hoặc giải nén enrollment: {e}")
                    # Tiếp tục chạy mượt mà dù không có enrollment
                    pass
                    with open(f"d:/website/debug_job_{job_id}.log", "w") as f:
                        f.write(f"Enrollment dir: {enrollment_dir}\n")
                        f.write(f"Contents: {list(enrollment_dir.iterdir())}\n")
                        
                    # --- CHẶNG 1.5: Tiền xử lý Enrollment ---
                    self.update_state(state='PROGRESS', meta={'percent': 12, 'message': 'Đang chuẩn hóa âm thanh mẫu (Enrollment)...'})
                    for e_file in enrollment_dir.rglob("*"):
                        if e_file.is_file():
                            with open(f"d:/website/debug_job_{job_id}.log", "a") as f:
                                f.write(f"Processing: {e_file}\n")
                            if e_file.suffix.lower() != ".wav":
                                # Convert sang wav 16kHz Mono
                                wav_path = e_file.with_suffix(".wav")
                                try:
                                    normalize_audio_to_mono16k(str(e_file), str(wav_path))
                                    e_file.unlink() # Xóa file gốc (.m4a, .mp3...) sau khi convert
                                    with open(f"d:/website/debug_job_{job_id}.log", "a") as f:
                                        f.write(f"Converted {e_file.name} -> {wav_path.name} OK\n")
                                except Exception as e:
                                    with open(f"d:/website/debug_job_{job_id}.log", "a") as f:
                                        f.write(f"FAILED to convert {e_file.name}: {e}\n")
                            else:
                                # Nếu đã là .wav, vẫn convert để đảm bảo định dạng PCM 16kHz chuẩn (tránh lỗi float32/24-bit với scipy)
                                temp_path = e_file.with_name(e_file.stem + "_temp.wav")
                                try:
                                    e_file.rename(temp_path)
                                    normalize_audio_to_mono16k(str(temp_path), str(e_file))
                                    temp_path.unlink()
                                    with open(f"d:/website/debug_job_{job_id}.log", "a") as f:
                                        f.write(f"Re-normalized {e_file.name} OK\n")
                                except Exception as e:
                                    with open(f"d:/website/debug_job_{job_id}.log", "a") as f:
                                        f.write(f"FAILED to re-normalize {e_file.name}: {e}\n")
                    # Log final enrollment dir contents after conversion
                    print(f"[DER JOB {job_id}] Enrollment dir AFTER conversion: {list(enrollment_dir.iterdir())}")
            
            # Gọi hàm Diarization chạy ngầm bằng Python Generator (yield)
            for log in run_der_pipeline(
                audio_path=mono_audio_path,
                enrollment_dir=enrollment_dir,
                n_speakers=job.num_speakers, # Lấy linh hoạt từ Database do người dùng nhập
                checkpoint_path=Path(DER_CHECKPOINT_DEFAULT),
                output_dir=output_dir,
                script_path=DER_SCRIPT_PATH,
                segmentation_step=0.1
            ):
                # Bắn log thực tế ra màn hình Celery để debug
                print(f"[DER JOB {job_id}] {log}")
                
            rttm_path = output_dir / "hyp_low025.rttm"
            
            # --- CHẶNG 3: Bóc băng văn bản (PhoWhisper) ---
            self.update_state(state='PROGRESS', meta={'percent': 60, 'message': 'Đang bóc băng bằng AI Whisper...'})
            
            engine_input_dir = prepare_engine_input(mono_audio_path, rttm_path, output_dir)
            output_csv = output_dir / "asr_results.csv"
            
            run_core_asr_engine(
                engine_input_dir=engine_input_dir,
                model_path=Path(ASR_CHECKPOINT_DEFAULT),
                output_path=output_csv,
                use_fast_mode=True
            )
            
            # --- CHẶNG 4: Lưu kết quả (Database) ---
            self.update_state(state='PROGRESS', meta={'percent': 95, 'message': 'Đang lưu biên bản vào Database...'})
            
            import pandas as pd
            from app.models.transcript import Transcript
            
            df = pd.read_csv(output_csv)
            for _, row in df.iterrows():
                transcript = Transcript(
                    job_id=job_id,
                    start_time=float(row['start']),
                    end_time=float(row['end']),
                    speaker=str(row['speaker']),
                    text=str(row['predicted_text'])
                )
                session.add(transcript)
            
            job.status = JobStatus.COMPLETED
            session.commit()
            
            # --- CHẶNG 5: Dọn dẹp rác (Tối ưu ổ cứng) ---
            import shutil
            if output_dir.exists():
                shutil.rmtree(output_dir, ignore_errors=True)
                
            return f"Xử lý thành công! Biên bản đã lưu DB và thư mục nháp {output_dir} đã được dọn sạch."
            
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            session.commit()
            raise e
