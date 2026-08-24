import pandas as pd


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tạo các feature tương tác (interaction) giữa các biến tương quan mạnh với target.

    Các feature mới:
    - study_efficiency: study_time_hours * attendance_percent
    - study_time_x_previous_grade: study_time_hours * previous_grade
    - attendance_x_sleep: attendance_percent * sleep_hours

    Lưu ý: hàm KHÔNG sửa df gốc, trả về bản copy đã thêm cột.
    """
    required_cols = ["study_time_hours", "attendance_percent", "previous_grade", "sleep_hours"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Thiếu cột bắt buộc để tạo interaction feature: {missing}")

    df = df.copy()
    df["study_efficiency"] = df["study_time_hours"] * df["attendance_percent"]
    df["study_time_x_previous_grade"] = df["study_time_hours"] * df["previous_grade"]
    df["attendance_x_sleep"] = df["attendance_percent"] * df["sleep_hours"]
    return df


def bin_sleep_hours(
    df: pd.DataFrame,
    col: str = "sleep_hours",
    low_thresh: float = 6.0,
    high_thresh: float = 8.0,
    drop_original: bool = False,
) -> pd.DataFrame:
    """Rời rạc hóa sleep_hours thành 3 nhóm: thiếu ngủ / đủ / dư.

    Nhóm được tạo dựa trên giả thuyết quan hệ dạng chữ U giữa giờ ngủ
    và kết quả học tập (ngủ quá ít hoặc quá nhiều đều không tốt).

    Kết quả: thêm cột categorical f"{col}_group" với 3 giá trị:
    "thieu_ngu" (< low_thresh), "du" (low_thresh - high_thresh), "du_thua" (> high_thresh)
    """
    if col not in df.columns:
        raise ValueError(f"Không tìm thấy cột '{col}' trong dataframe")

    df = df.copy()
    bins = [-float("inf"), low_thresh, high_thresh, float("inf")]
    labels = [-1, 0, 1]
    df[f"{col}_group"] = pd.cut(df[col], bins=bins, labels=labels, right=False)

    if drop_original:
        df = df.drop(columns=[col])

    return df


def engineer_features(
    df: pd.DataFrame,
    add_interactions: bool = True,
    add_sleep_bins: bool = True,
) -> pd.DataFrame:
    """Hàm tổng hợp: chạy toàn bộ pipeline feature engineering theo thứ tự.

    Dùng hàm này trong notebook/pipeline chính thay vì gọi từng hàm lẻ,
    để đảm bảo thứ tự xử lý nhất quán.
    """
    result = df.copy()

    if add_interactions:
        result = add_interaction_features(result)

    if add_sleep_bins:
        result = bin_sleep_hours(result)

    return result