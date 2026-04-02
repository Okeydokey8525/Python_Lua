# AGENTS.md

## 1. Project Overview (Mô tả project)

Dự án này xây dựng một hệ thống **web chẩn đoán ảnh** sử dụng **YOLOv12 + Transformer** với luồng xử lý end-to-end như sau:

- **Người dùng upload ảnh lên web**
  - Người dùng truy cập giao diện web (Flask/FastAPI + frontend đơn giản).
  - Ảnh được gửi từ trình duyệt lên backend thông qua API endpoint.

- **Hệ thống chẩn đoán ảnh bằng YOLOv12 + Transformer**
  - Backend nhận ảnh, tiền xử lý (resize, normalize, chuyển định dạng màu nếu cần).
  - Mô hình YOLOv12 thực hiện phát hiện đối tượng/vùng nghi ngờ.
  - Transformer xử lý ngữ cảnh hoặc tăng chất lượng suy luận (ví dụ: phân loại tinh chỉnh, tổng hợp đặc trưng, hoặc giải thích kết quả).
  - Kết quả được hậu xử lý thành dạng dễ đọc (label, confidence, bounding box, kết luận).

- **Trả kết quả chẩn đoán về giao diện web**
  - Backend phản hồi kết quả dưới dạng JSON hoặc render trực tiếp HTML.
  - Web hiển thị ảnh đầu vào kèm kết quả chẩn đoán (nhãn, độ tin cậy, vùng phát hiện).

### Mục tiêu và lợi ích

- Tạo pipeline chẩn đoán ảnh tự động, dễ dùng qua giao diện web.
- Giúp rút ngắn thời gian phân tích ảnh và giảm thao tác thủ công.
- Tăng khả năng mở rộng: dễ tích hợp thêm model, API, hoặc dashboard theo nhu cầu thực tế.

---

## 2. Tech Stack (Công nghệ sử dụng)

- **Python** (ngôn ngữ chính cho backend, xử lý dữ liệu và model inference)
- **YOLOv12 + Transformer** (mô hình AI cho detection + hiểu ngữ cảnh)
- **Flask hoặc FastAPI** (xây dựng web server/API)
- **Thư viện liên quan**:
  - `torch` / `torchvision`
  - `opencv-python` (OpenCV)
  - `Pillow`
  - `numpy`
  - `pydantic` (đặc biệt hữu ích nếu dùng FastAPI)
  - Các thư viện hỗ trợ khác tùy theo triển khai thực tế

---

## 3. Rules (Quy tắc coding)

1. **Không sửa file data**
   - Không chỉnh sửa trực tiếp dữ liệu gốc trong thư mục `/data`.
   - Nếu cần biến đổi dữ liệu, tạo file/phiên bản mới ở thư mục phù hợp.

2. **Code sạch, readable, dễ maintain (Clean Code)**
   - Tách module rõ ràng: web, model, utils, config.
   - Đặt tên biến/hàm có ý nghĩa, tránh viết tắt khó hiểu.
   - Ưu tiên hàm ngắn gọn, một nhiệm vụ chính.

3. **Tuân thủ chuẩn Python (PEP8)**
   - Format code nhất quán, dễ đọc.
   - Khuyến nghị dùng `black`, `isort`, `flake8` hoặc `ruff`.

4. **Có test coverage cơ bản**
   - Tối thiểu có test cho các luồng chính: load model, infer ảnh mẫu, API upload/predict.
   - Ưu tiên dùng `pytest`.

5. **Không xóa hoặc phá hỏng file cũ**
   - Mọi thay đổi cần đảm bảo backward compatibility ở mức hợp lý.
   - Nếu cần refactor lớn, thực hiện theo từng bước và có kiểm thử đi kèm.

---

## 4. How to Run (Cách chạy project)

### Cấu trúc thư mục gợi ý

```text
/project
  /data        # chứa ảnh hoặc dataset
  /model       # chứa YOLOv12 + transformer model
  run_web.py   # file chạy giao diện web
  run_model.py # file chạy model chẩn đoán
```

### Lệnh cơ bản

```bash
python run_model.py   # Chạy model YOLOv12 + transformer
python run_web.py     # Chạy web server
```

### Ví dụ luồng sử dụng

1. Khởi động web server:

```bash
python run_web.py
```

2. Mở giao diện web (ví dụ):
   - `http://127.0.0.1:5000` (Flask)
   - hoặc `http://127.0.0.1:8000` (FastAPI)

3. Upload file ảnh `test.jpg`.
4. Hệ thống trả kết quả, ví dụ:
   - Label: `abnormal_region`
   - Confidence: `0.93`
   - Bounding box: `[x1, y1, x2, y2]`

---

## 5. Agent Behavior (Cách AI nên hành xử)

- AI **có thể tự sửa bug** trong code khi xác định rõ nguyên nhân.
- AI **không được xóa hoặc làm hỏng file cũ**.
- AI nên **đưa gợi ý tối ưu code** (cấu trúc, hiệu năng, readability), đồng thời kiểm tra lỗi syntax/performance cơ bản.
- Nếu AI **không chắc chắn** về thay đổi có thể ảnh hưởng logic nghiệp vụ, **hãy hỏi người dùng trước khi chỉnh sửa**.
- Khi chỉnh sửa nhiều file, AI nên mô tả rõ phạm vi ảnh hưởng và lý do thay đổi.
