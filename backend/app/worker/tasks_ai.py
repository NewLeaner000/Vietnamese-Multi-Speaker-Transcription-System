import os
from sqlmodel import Session, select
from app.db.database import engine
from app.models.transcript import Transcript
from app.models.summary import Summary
from app.worker.celery_app import celery_app

@celery_app.task(bind=True, name="summarize_audio")
def summarize_audio_task(self, job_id: int):
    """
    Celery task để chạy Qwen tóm tắt trên local GPU.
    """
    from app.ai_core.pipeline_config import QWEN_GGUF_MODEL_DEFAULT
    
    with Session(engine) as session:
        transcripts = session.exec(select(Transcript).where(Transcript.job_id == job_id).order_by(Transcript.start_time)).all()
        if not transcripts:
            return "Không có dữ liệu hội thoại."
            
        conversation = ""
        for t in transcripts:
            conversation += f"[{t.speaker}]: {t.text}\n"
            
        prompt = f"""Bạn là một trợ lý AI chuyên nghiệp chuyên tóm tắt biên bản cuộc họp.
Hãy đọc đoạn hội thoại sau và tóm tắt lại bằng tiếng Việt.
YÊU CẦU BẮT BUỘC: Bạn PHẢI sử dụng định dạng Markdown để trình bày bản tóm tắt thật đẹp mắt, theo đúng 3 phần sau:

### 📝 Khái quát cuộc họp
(Viết 1-2 đoạn văn tóm tắt nội dung chính và mục đích của cuộc họp)

### 🎯 Các quyết định đã chốt
(Sử dụng gạch đầu dòng `-` để liệt kê các quyết định quan trọng)

### ✅ Việc cần làm
(Sử dụng gạch đầu dòng `-` để liệt kê rõ nhiệm vụ. Tên người nhận nhiệm vụ PHẢI được **bôi đậm**)

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
            print(f"[QWEN] Đang nạp mô hình LLM lên GPU cho job {job_id}...")
            
            model = Llama(
                model_path=QWEN_GGUF_MODEL_DEFAULT,
                n_gpu_layers=-1,
                n_ctx=8192,
                verbose=False
            )
            
            response = model.create_chat_completion(
                messages=[
                    {"role": "system", "content": "Bạn là AI tóm tắt văn bản chuyên nghiệp, luôn trả lời đúng format yêu cầu."},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024,
                temperature=0.1
            )
            
            result_text = response["choices"][0]["message"]["content"].strip()
            del model # Free VRAM
            
            summary = session.exec(select(Summary).where(Summary.job_id == job_id)).first()
            if not summary:
                summary = Summary(job_id=job_id, meeting_overview=result_text)
                session.add(summary)
            else:
                summary.meeting_overview = result_text
                
            session.commit()
            return result_text
            
        except Exception as e:
            print(f"[QWEN ERROR] Lỗi khi chạy tóm tắt: {e}")
            raise e

@celery_app.task(bind=True, name="chat_with_qwen")
def chat_with_qwen_task(self, job_id: int, message: str, system_prompt: str):
    """
    Celery task để chạy Qwen trả lời chat.
    """
    from app.ai_core.pipeline_config import QWEN_GGUF_MODEL_DEFAULT
    
    with Session(engine) as session:
        transcripts = session.exec(select(Transcript).where(Transcript.job_id == job_id).order_by(Transcript.start_time)).all()
        if not transcripts:
            return "Chưa có dữ liệu hội thoại."
            
        conversation = ""
        for t in transcripts:
            conversation += f"[{t.speaker}]: {t.text}\n"

    user_prompt = f"""Dưới đây là nội dung cuộc họp:
---
{conversation}
---
Dựa vào cuộc họp trên và vai trò của bạn, hãy trả lời câu hỏi sau của tôi:
{message}
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
        del model
        
        return answer
        
    except Exception as e:
        print(f"[CHAT ERROR] {e}")
        raise e
