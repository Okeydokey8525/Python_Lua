🌾 Hệ Thống Chẩn Đoán 10 Loại Bệnh Lá Lúa Sử Dụng Mô Hình Hybrid YOLOv12-CBAM-CNN
Giới thiệu
Đồ án tập trung xây dựng hệ thống hỗ trợ nông dân nhận diện và chẩn đoán tự động 10 loại bệnh phổ biến trên cây lúa. Điểm nổi bật của dự án là việc kết hợp sức mạnh định vị của YOLOv12, khả năng tập trung đặc trưng của cơ chế chú ý CBAM và khả năng phân loại sâu của CNN.

Sinh viên thực hiện: 
Lê Đức Lương
Huỳnh Thái Tấn Nghĩa



Khoa: Công nghệ thông tin - HUIT

Mô hình đề xuất: Hybrid YOLOv12 + CBAM + CNN

🧪 Danh sách 10 loại bệnh mục tiêu
Hệ thống có khả năng nhận diện các nhãn sau:

Bacterial Leaf Blight (Bạc lá)

Brown Spot (Đốm nâu)

Leaf Blast (Đạo ôn lá)

Leaf Smut (Than kháng)

Tungro (Tungro)

...

🏗 Kiến trúc mô hình Hybrid
Dự án áp dụng cấu trúc đa tầng để tối ưu hóa độ chính xác mAP:

Backbone (YOLOv12): Thực hiện trích xuất đặc trưng toàn cục và khoanh vùng (Localization) các vết bệnh trên bề mặt lá.

Attention Module (CBAM): Tích hợp vào các lớp Neck của mô hình để tập trung vào các đặc trưng quan trọng theo kênh (Channel) và không gian (Spatial), giúp nhận diện tốt các vết bệnh li ti.

Classifier (CNN Custom): Tinh chỉnh các đặc trưng cuối cùng để đưa ra kết luận chính xác nhất về loại bệnh.

🛠 Công nghệ & Thư viện
Ngôn ngữ: Python 3.13

Framework: PyTorch, Ultralytics (YOLOv12)

Giao diện: Streamlit (Web App)

Xử lý dữ liệu: OpenCV, Albumentations (Augmentation)

🚀 Hướng dẫn cài đặt
1. Chuẩn bị môi trường
PowerShell
python -m venv venv
.\venv\Scripts\activate
2. Cài đặt thư viện
PowerShell
python -m pip install ultralytics streamlit opencv-python pillow

📊 Đánh giá kết quả
Hệ thống được đánh giá qua các chỉ số:

mAP@0.5: Đạt ngưỡng tối ưu cho các bệnh có hình dạng phức tạp.

Precision/Recall: Cân bằng tốt giữa việc phát hiện đúng và không bỏ sót bệnh.

Inference Time: Tối ưu hóa để chạy được trên các máy tính cấu hình trung bình (như Lenovo LOQ).

💡 Hướng phát triển
Triển khai mô hình dưới dạng Mobile App để sử dụng trực tiếp tại đồng ruộng.

Tích hợp hệ thống khuyến nghị thuốc bảo vệ thực vật tương ứng với từng loại bệnh.
