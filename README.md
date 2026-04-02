

---

# 🌾 RiceLeaf-AI: Chẩn đoán 10 Loại Bệnh Lá Lúa (YOLOv12 + Transformer)

## 📝 Giới thiệu
Đồ án tập trung xây dựng hệ thống hỗ trợ nông dân nhận diện và chẩn đoán tự động **10 loại bệnh phổ biến trên cây lúa**. 

Điểm đột phá của dự án là việc chuyển đổi từ kiến trúc CNN truyền thống sang mô hình **YOLOv12 kết hợp cơ chế Transformer (Self-Attention)**. Sự kết hợp này giúp hệ thống không chỉ định vị chính xác vết bệnh mà còn hiểu được bối cảnh toàn cục của lá lúa, giúp phân biệt các loại bệnh có hình thái tương đồng một cách vượt trội.

* **Sinh viên thực hiện:** Lê Đức Lương & Huỳnh Thái Tấn Nghĩa
* **Khoa:** Công nghệ thông tin - HUIT
* **Mô hình đề xuất:** YOLOv12 (Attention-centric Architecture)

---

## 🧪 Danh sách 10 loại bệnh mục tiêu
Hệ thống được huấn luyện để nhận diện chính xác 10 nhãn (Classes):
1.  **Bacterial Leaf Blight** (Bạc lá)
2.  **Brown Spot** (Đốm nâu)
3.  **Leaf Blast** (Đạo ôn lá)
4.  **Leaf Smut** (Than kháng)
5.  **Tungro** (Bệnh virus Tungro)
6.  **Leaf Scald** (Thối thân)
7.  **Narrow Brown Spot** (Đốm nâu hẹp)
8.  **Neck Blast** (Đạo ôn cổ bông)
9.  **Rice Hispa** (Sâu gai)
10. **Sheath Blight** (Khô vằn)

---

## 🏗 Kiến trúc mô hình (Transformer + YOLOv12)
Dự án áp dụng triết lý thiết kế hiện đại, loại bỏ các thành phần CNN rời rạc để tối ưu hóa độ trễ:

* **Backbone (YOLOv12):** Tích hợp các khối **Transformer/Attention Blocks** giúp trích xuất đặc trưng với tầm nhìn toàn cục (Global Context).
* **Attention Mechanism:** Sử dụng cơ chế **Self-Attention** để tập trung vào các chi tiết vết bệnh nhỏ và phức tạp mà CNN truyền thống thường bỏ sót.
* **End-to-End Regression:** Biến bài toán phát hiện thành một bài toán hồi quy duy nhất, tối ưu hóa tốc độ xử lý cho các thiết bị thực tế.

---

## 🛠 Công nghệ & Thư viện
* **Ngôn ngữ:** Python 3.13+
* **Framework:** PyTorch, Ultralytics (YOLOv12)
* **Giao diện:** Streamlit (Web App)
* **Kỹ thuật:** Transformer Integration, Image Augmentation, Edge AI Optimization.

---

## 🚀 Hướng dẫn cài đặt

### 1. Chuẩn bị môi trường
```powershell
# Tạo môi trường ảo
python -m venv .venv
# Kích hoạt môi trường
.\.venv\Scripts\activate
```

### 2. Cài đặt thư viện
```powershell
python -m pip install --upgrade pip
python -m pip install ultralytics streamlit opencv-python pillow
```

---

## 📊 Đánh giá kết quả
Mô hình YOLOv12 + Transformer đạt được các chỉ số ấn tượng (SOTA):
* **mAP@0.5:** Tăng đáng kể so với phiên bản YOLOv8/v9 nhờ hiểu sâu bối cảnh ảnh.
* **FPS (Frames Per Second):** Duy trì tốc độ thời gian thực (>30 FPS) trên các dòng laptop gaming tầm trung (như Lenovo LOQ).
* **Inference Time:** Giảm thiểu độ trễ, phù hợp cho việc triển khai trên Drone hoặc thiết bị di động.

---

## 💡 Hướng phát triển
* Huấn luyện lại với bộ dữ liệu đặc thù tại các vùng canh tác lúa tại Việt Nam.
* Tích hợp hệ thống chuyên gia khuyến nghị thuốc bảo vệ thực vật tương ứng.
* Triển khai ứng dụng di động nhận diện trực tiếp qua Camera điện thoại.

---

### 🛡 Giấy phép
Dự án được phát triển dưới dạng mã nguồn mở nhằm phục vụ mục đích học tập và hỗ trợ cộng đồng nông nghiệp.

---
