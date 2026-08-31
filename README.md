# student grade predict

Dự án học tập, thực hành quy trình hoàn chỉnh khi huấn luyện mô hình học máy: từ xử lý dữ liệu, huấn luyện, đến đánh giá kết quả. Dữ liệu sử dụng là bộ `student_performance_dataset`, với 4 mô hình chính được cài đặt: **Decision Tree**, **Random Forest**, **AdaBoost**, **LogisticRegression**

## Cấu trúc thư mục

| Thư mục | Vai trò |
|---|---|
| [`data/`](./data/README.md) | Chứa dữ liệu gốc và dữ liệu đã qua xử lý |
| [`notebooks/`](./notebooks/README.md) | Các notebook thử nghiệm: xử lý dữ liệu, huấn luyện, tinh chỉnh tham số |
| [`src/`](./src/README.md) | Mã nguồn chính của dự án (đóng gói thành module) |

## Cách chạy
1. `pip install -r requirements.txt`
2. Chạy `notebooks/preprocessing.ipynb` để tạo data/raw, data/processed, data/filtered
3. Chạy `notebooks/hyperparameter_tuning.ipynb` (tùy chọn) để tìm lại tham số tối ưu
4. Chạy `notebooks/main.ipynb` để train và đánh giá 4 model trên 3 bộ dữ liệu

Chi tiết từng thư mục xem trong file `README.md` tương ứng bên trong.
