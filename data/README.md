# data/

Chứa toàn bộ dữ liệu dùng trong dự án.

## Nội dung

- **`student_performance_dataset.csv`** — Bộ dữ liệu gốc, chưa qua xử lý hay tách tập.
- **`raw/`** — Dữ liệu thô sau khi được tách thành các tập (chưa xử lý):
  - `train.csv` — Tập huấn luyện
  - `val.csv` — Tập validation
  - `test.csv` — Tập kiểm tra
  - `.gitkeep` — File giữ chỗ để Git không bỏ qua thư mục rỗng
- **`processed/`** — Dữ liệu đã qua tiền xử lý (làm sạch, biến đổi đặc trưng...), tương ứng cấu trúc với `raw/`:
  - `train.csv`, `val.csv`, `test.csv`
  - `.gitkeep`
- **`filtered/`** — Dữ liệu qua giảm chiều bằng phương pháp boruta, tương ứng cấu trúc với `raw/`:
  - `train.csv`, `val.csv`, `test.csv`
  - `.gitkeep`
