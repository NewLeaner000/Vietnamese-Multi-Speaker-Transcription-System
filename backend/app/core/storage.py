from supabase import create_client, Client
from app.core.config import settings
import uuid
import os
from pathlib import Path

# Khởi tạo client Supabase với service_role key để có toàn quyền ghi đè Storage Policies
supabase: Client = create_client(settings.SUPABASE_URL, settings.SUPABASE_KEY)
BUCKET_NAME = "ani-audio-bucket"

# Tự động tạo Bucket nếu chưa có
try:
    buckets = supabase.storage.list_buckets()
    bucket_names = [b.name for b in buckets]
    if BUCKET_NAME not in bucket_names:
        supabase.storage.create_bucket(BUCKET_NAME, options={'public': True})
        print(f"Đã tự động tạo Supabase Bucket: {BUCKET_NAME}")
except Exception as e:
    print(f"Cảnh báo khi kiểm tra Bucket: {e}")

def upload_file_to_supabase(local_file_path: str, destination_path: str) -> str:
    """
    Tải file từ máy local (hoặc backend API tạm) lên Supabase Storage.
    Trả về Public URL của file.
    """
    # Nếu file đã tồn tại trên Supabase, ta sẽ ghi đè
    with open(local_file_path, "rb") as f:
        res = supabase.storage.from_(BUCKET_NAME).upload(
            file=f,
            path=destination_path,
            file_options={"cache-control": "3600", "upsert": "true"}
        )
    
    # Lấy Public URL
    public_url = supabase.storage.from_(BUCKET_NAME).get_public_url(destination_path)
    return public_url

def download_file_from_supabase(remote_path: str, local_destination: str):
    """
    Tải file từ Supabase Storage về ổ cứng Local (dành cho AI Worker).
    """
    res = supabase.storage.from_(BUCKET_NAME).download(remote_path)
    with open(local_destination, 'wb+') as f:
        f.write(res)
    return local_destination
