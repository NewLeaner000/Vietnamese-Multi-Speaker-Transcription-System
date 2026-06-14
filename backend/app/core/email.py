import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from app.core.config import settings

def send_verification_email(to_email: str, code: str):
    """
    Gửi email chứa mã xác thực OTP.
    """
    if not settings.SMTP_USER or not settings.SMTP_PASSWORD:
        print("CẢNH BÁO: Chưa cấu hình SMTP_USER và SMTP_PASSWORD trong .env. Bỏ qua gửi mail thực tế.")
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

    msg = MIMEMultipart()
    msg['From'] = settings.SMTP_USER
    msg['To'] = to_email
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'html'))

    try:
        # Cấu hình cho Gmail hoặc các dịch vụ SMTP sử dụng TLS (port 587)
        server = smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT)
        server.starttls()
        server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
        server.sendmail(settings.SMTP_USER, to_email, msg.as_string())
        server.quit()
        print(f"Đã gửi email OTP tới {to_email}")
    except Exception as e:
        print(f"Lỗi khi gửi email: {e}")
        raise e
