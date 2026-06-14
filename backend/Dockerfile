# Sử dụng bản Python 3.10 siêu nhẹ (slim) làm nền tảng
FROM python:3.10-slim

# Thiết lập thư mục làm việc trong Docker container
WORKDIR /app

# Cài đặt các thư viện hệ thống cần thiết (để build psycopg2 và các thư viện khác)
RUN apt-get update && apt-get install -y \
    gcc \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy file requirements vào trước để tận dụng Docker Cache
COPY requirements.txt .

# Cài đặt toàn bộ thư viện Python
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ mã nguồn Backend vào trong container
COPY . .

# Nền tảng Render (và các Cloud khác) sẽ cấp phát một biến môi trường tên là PORT
# Uvicorn bắt buộc phải lắng nghe trên cổng này thay vì cổng 8000 cứng.
# Lệnh CMD này sẽ khởi động Uvicorn.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
