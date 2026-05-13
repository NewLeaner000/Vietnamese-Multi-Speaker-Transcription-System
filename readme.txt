cấm được sửa .venv hay thư viện chỉ được sửa code
cấu hình làm code website là để chạy trên máy 4060 8gb win 11 , còn máy train test và dev hiện tại là 3090ti
thiết lập giao diện để kết nối các model để làm ra giao diện tương tác được  , 
mọi sự thay đổi hay cập nhật về code log thì cập nhật vào file này để dễ theo dõi
giao diện là dành cho người dùng nên phải có các chức năng như sau:
- upload file audio
- chọn thông để xử lý hoặc thông số mặc định và có thể chỉnh sửa
- xem kết quả xử lý và history lưu trữ các file và có thể chỉnh sửa
- có thể tải về kết quả xử lý
-  giao diện xử lí có thể hiện thời gian chờ 
các folder code code_DiariZen, code_pho_whisper, code_qwen_25_7b, code_pyannote đều chứa các best checkpoint load lên để dùng , ko dùng model base 
chỉ được sử dụng 1 venv duy nhất

---
