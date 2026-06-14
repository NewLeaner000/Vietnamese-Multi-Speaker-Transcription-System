from locust import HttpUser, task, between
import random

class ViMeetUser(HttpUser):
    # Thời gian nghỉ ngẫu nhiên giữa các thao tác (như user thật đang lướt web)
    wait_time = between(1, 5)
    
    def on_start(self):
        """Hàm này chạy ĐẦU TIÊN khi 1 User ảo được sinh ra (Mô phỏng Đăng nhập)"""
        # Tạo user random để tránh trùng lặp
        self.username = f"testuser_{random.randint(1000, 99999)}"
        self.password = "123456"
        
        # 1. Đăng ký tài khoản (Nếu đã tồn tại thì nó báo lỗi 400 nhưng kệ)
        self.client.post("/api/auth/register", json={
            "username": self.username,
            "email": f"{self.username}@test.com",
            "password": self.password
        })
        
        # 2. Đăng nhập để lấy Token
        response = self.client.post("/api/auth/login", data={
            "username": self.username,
            "password": self.password
        })
        
        if response.status_code == 200:
            self.token = response.json().get("access_token")
            # Cài đặt Header có Token cho mọi request sau này
            self.client.headers.update({"Authorization": f"Bearer {self.token}"})
        else:
            self.token = None
            print(f"Lỗi đăng nhập: {response.text}")

    @task(3) # Số 3 nghĩa là tỷ lệ gọi hàm này nhiều gấp 3 lần hàm khác
    def view_history(self):
        """Mô phỏng User vào xem lịch sử"""
        if self.token:
            self.client.get("/api/audio/jobs")

    @task(1)
    def upload_dummy_file(self):
        """Mô phỏng User upload 1 file âm thanh (Gửi file text nhưng đặt tên là .wav để ép API xử lý)"""
        if self.token:
            # Tạo một file giả để upload
            files = {
                'file': ('dummy_audio.wav', b'This is dummy audio content', 'audio/wav')
            }
            data = {
                'num_speakers': '2'
            }
            # Gửi thẳng vào FastAPI (FastAPI chỉ việc lưu DB và đẩy vào Redis, cực nhanh)
            self.client.post("/api/audio/upload", files=files, data=data)

    @task(1)
    def view_trash(self):
        """Mô phỏng User vào xem Thùng rác"""
        if self.token:
            self.client.get("/api/audio/jobs/trash")
