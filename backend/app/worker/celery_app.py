import os
from celery import Celery
import ssl
from app.core.config import settings

# Lấy URL Redis từ cấu hình chung (sẽ tự động đọc từ Upstash)
redis_url = settings.REDIS_URL

celery_app = Celery(
    "ai_worker",
    broker=redis_url,
    backend=redis_url,
    include=['app.worker.tasks', 'app.worker.tasks_ai'] # Báo cho Celery biết file nào chứa các tác vụ
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="Asia/Ho_Chi_Minh",
    enable_utc=True,
    
    # Hỗ trợ kết nối bảo mật TLS/SSL (bắt buộc cho Upstash Redis)
    broker_use_ssl={'ssl_cert_reqs': ssl.CERT_REQUIRED} if redis_url.startswith("rediss://") else None,
    redis_backend_use_ssl={'ssl_cert_reqs': ssl.CERT_REQUIRED} if redis_url.startswith("rediss://") else None,
    
    # [MLOps KNOWLEDGE] QUAN TRỌNG NHẤT: 
    # Ép Celery chỉ nhận và chạy đúng 1 task tại một thời điểm trên 1 máy tính.
    # Nếu để mặc định (chạy song song nhiều task), card RTX 4060 của bạn sẽ lập tức bị cháy VRAM (OOM)
    # khi có 2 người cùng upload file âm thanh 1 lúc!
    worker_concurrency=1,
    worker_prefetch_multiplier=1
)
