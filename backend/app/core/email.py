import requests
import json
from app.core.config import settings

def send_verification_email(to_email: str, code: str):
    """
    Gửi email chứa mã xác thực OTP thông qua Brevo HTTP API (Bypass cổng 587 bị chặn).
    """
    if not getattr(settings, 'BREVO_API_KEY', None):
        print("CẢNH BÁO: Chưa cấu hình BREVO_API_KEY trong .env. Bỏ qua gửi mail thực tế.")
        print(f"Mã OTP giả lập cho {to_email}: {code}")
        return

    subject = "Mã xác thực tài khoản Ani Assistant"
    body = f"""
    <html>
      <body>
        <h2>Chào mừng bạn đến với Ani Assistant</h2>
        <p>Đây là mã xác thực (OTP) để hoàn tất việc đăng ký tài khoản của bạn:</p>
        <h1 style="color: #007bff; letter-spacing: 2px;">{code}</h1>
        <p>Mã này sẽ hết hạn trong 5 phút. Vui lòng không chia sẻ mã này cho bất kỳ ai.</p>
        <br/>
        <p>Trân trọng,<br/>Đội ngũ Ani Assistant</p>
      </body>
    </html>
    """

    url = "https://api.brevo.com/v3/smtp/email"
    
    # Brevo yêu cầu email người gửi phải là email bạn đã xác minh (email đăng ký tài khoản Brevo)
    # Chúng ta sẽ tận dụng luôn biến SMTP_USER cũ trong file .env (thường là Gmail của bạn)
    sender_email = getattr(settings, 'SMTP_USER', 'noreply@ani-assistant.com')
    if not sender_email:
        sender_email = "noreply@ani-assistant.com"

    payload = json.dumps({
      "sender": {
        "name": "Ani Assistant",
        "email": sender_email
      },
      "to": [
        {
          "email": to_email
        }
      ],
      "subject": subject,
      "htmlContent": body
    })
    
    headers = {
      'accept': 'application/json',
      'api-key': settings.BREVO_API_KEY,
      'content-type': 'application/json'
    }

    try:
        response = requests.post(url, headers=headers, data=payload)
        response.raise_for_status()
        print(f"Đã gửi email OTP tới {to_email} qua Brevo API")
    except Exception as e:
        print(f"Lỗi khi gửi email qua Brevo API: {e}")
        if hasattr(e, 'response') and e.response is not None:
            print(f"Chi tiết lỗi từ Brevo: {e.response.text}")
        raise e
