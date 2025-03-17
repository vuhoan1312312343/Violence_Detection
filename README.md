# HỆ THỐNG NHẬN DIỆN VÀ CẢNH BÁO HÀNH VI BẠO LỰC TRONG ĐÁM ĐÔNG

.![LOGO](https://github.com/user-attachments/assets/3228e611-f05f-44f1-9d8b-aa0a8c15ea7e)


## Giới thiệu
Hệ thống nhận diện và cảnh báo hành vi bạo lực trong đám đông được phát triển nhằm phát hiện các hành vi nguy hiểm trong thời gian thực bằng cách sử dụng laptop, camera Ezviz và phần mềm Python kết hợp với mô hình CNN 3D. Hệ thống hoạt động bằng cách thu thập hình ảnh từ camera giám sát, sau đó xử lý và phân tích dữ liệu thông qua mô hình CNN 3D để xác định các hành vi có dấu hiệu bạo lực.Toàn bộ quá trình nhận diện và cảnh báo được thực hiện trên phần mềm chạy trực tiếp trên laptop, giúp tối ưu hóa tính linh hoạt và dễ triển khai của hệ thống.

## Tính năng chính
- Nhận diện hành vi bạo lực trong thời gian thực bằng AI.
- Kết hợp camera giám sát để thu thập dữ liệu hình ảnh.
- Cảnh báo bằng âm thanh, đèn LED hoặc gửi thông báo.
- Lưu trữ dữ liệu hành vi trong MongoDB để phân tích.
- Giao diện đơn giản với Tkinter để hiển thị kết quả.

## Công nghệ sử dụng
- **Phần cứng:** Arduino, Camera giám sát
- **Ngôn ngữ lập trình:** Python
- **Mô hình AI:** CNN 3D hoặc các mô hình nhận diện hành vi
- **Cơ sở dữ liệu:** MongoDB
- **Giao diện người dùng:** Tkinter

## Cài đặt
1. Clone repository:
   ```bash
   git clone https://github.com/your-repo/violence-detection.git
   cd violence-detection
   ```
2. Cài đặt các thư viện cần thiết:
   ```bash
   pip install -r requirements.txt
   ```
3. Cấu hình MongoDB:
   - Khởi chạy MongoDB server.
   - Chỉnh sửa file cấu hình `config.py` để kết nối đến MongoDB.

4. Chạy ứng dụng:
   ```bash
   python main.py
   ```

## Đóng góp
Nếu bạn muốn đóng góp cho dự án, vui lòng fork repository và gửi pull request.

## Liên hệ
- **Đại Nam University** - [Website](https://dainam.edu.vn)
- **DNU - AIoT Lab** - [Website](https://aiotlab.dnu.edu.vn)

---
Cảm ơn bạn đã quan tâm đến dự án của chúng tôi!
