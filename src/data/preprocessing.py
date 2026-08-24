import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TEST_SIZE, VAL_SIZE


def drop_unused_columns(df: pd.DataFrame, columns: list = None) -> pd.DataFrame:
    """Loại bỏ các cột không dùng để train (không mang tính dự đoán)."""
    if columns is None:
        columns = ["student_id"]
    return df.drop(columns=columns)


def encode_binary_columns(df: pd.DataFrame, binary_map: dict = None) -> pd.DataFrame:
    """Chuyển các cột nhị phân sang 0/1.

    LƯU Ý: kiểm tra df[col].unique() trước để đảm bảo khớp chính xác chuỗi
    gốc (hoa/thường, khoảng trắng) — map() sai chính tả sẽ âm thầm ra NaN.
    """
    df = df.copy()
    if binary_map is None:
        binary_map = {
            "gender": {"Male": 0, "Female": 1},
            "internet_access": {"No": 0, "Yes": 1},
            "extracurricular_activities": {"No": 0, "Yes": 1},
            "part_time_job": {"No": 0, "Yes": 1},
        }
    for col, mapping in binary_map.items():
        df[col] = df[col].map(mapping)
    return df


def encode_ordinal_columns(df: pd.DataFrame, missing_strategy: str = "flag",
                             ordinal_map: dict = None) -> pd.DataFrame:
    """
    missing_strategy:
        "flag"  -> gán -1, thêm cột binary is_missing_parental_edu(Tree)
        "mode"  -> fill bằng giá trị phổ biến nhất(KNN)
    """
    df = df.copy()
    if ordinal_map is None:
        ordinal_map = {
            "parental_education": {"High School": 0, "Bachelors": 1, "Masters": 2, "PhD": 3},
            "final_grade": {"F": 0, "D": 1, "C": 2, "B": 3, "A": 4},
        }
    for col, mapping in ordinal_map.items():
        df[col] = df[col].map(mapping)

    if "parental_education" in df.columns:
        if missing_strategy == "flag":
            df["parental_education"] = df["parental_education"].fillna(-1).astype(int)
        elif missing_strategy == "mode":
            mode_val = df["parental_education"].mode()[0]
            df["parental_education"] = df["parental_education"].fillna(mode_val).astype(int)

    return df

# Split and save data

def split_train_val_test(
    df: pd.DataFrame,
    folder_path,
    test_size: float = TEST_SIZE,
    val_size: float = VAL_SIZE,
    target_col: str = "final_grade",
    random_state: int = RANDOM_STATE,
    stratify: bool = True,
) -> None:
    """Chia dữ liệu thành 3 tập train/val/test và lưu ra CSV trong folder_path.

    Cách chia: tách test trước (test_size trên toàn bộ df), sau đó tách val
    từ phần còn lại (train+val). val_size được hiểu là tỉ lệ trên TOÀN BỘ
    df ban đầu (giống ý nghĩa TEST_SIZE trong config), không phải tỉ lệ trên
    phần còn lại sau khi trừ test — để 2 tham số dễ diễn giải cùng nhau
    (vd: TEST_SIZE=0.2, VAL_SIZE=0.1 -> train=70%, val=10%, test=20%).

    Params:
        df: dataframe đã qua feature engineering.
        folder_path: thư mục lưu file, mặc định DATA_PROCESSED_DIR.
        test_size: tỉ lệ tập test trên toàn bộ dữ liệu, mặc định config.TEST_SIZE.
        val_size: tỉ lệ tập val trên toàn bộ dữ liệu, mặc định config.VAL_SIZE.
        target_col: cột target, dùng để stratify.
        random_state: seed để tái lập kết quả.
        stratify: nếu True và target_col là biến rời rạc, giữ tỉ lệ phân bố
                  target đồng đều giữa 3 tập.

    Output: không trả về gì, lưu "train.csv", "val.csv", "test.csv" vào folder_path.
    """
    if test_size + val_size >= 1.0:
        raise ValueError(f"test_size + val_size phải nhỏ hơn 1.0, hiện tại = {test_size + val_size}")

    folder_path.mkdir(parents=True, exist_ok=True)

    def _get_stratify_col(data: pd.DataFrame):
        if stratify and target_col in data.columns and data[target_col].nunique() <= 20:
            return data[target_col]
        return None

    # Bước 1: tách test ra khỏi toàn bộ dữ liệu
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=_get_stratify_col(df),
    )

    # Bước 2: tách val từ phần còn lại (train_val_df)
    # quy đổi val_size (tỉ lệ trên toàn bộ df) sang tỉ lệ trên train_val_df
    relative_val_size = val_size / (1.0 - test_size)

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=relative_val_size,
        random_state=random_state,
        stratify=_get_stratify_col(train_val_df),
    )

    train_path = folder_path / "train.csv"
    val_path = folder_path / "val.csv"
    test_path = folder_path / "test.csv"

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)

    print(f"Đã lưu {len(train_df)} dòng train ({len(train_df)/len(df):.1%}) vào: {train_path}")
    print(f"Đã lưu {len(val_df)} dòng val ({len(val_df)/len(df):.1%}) vào: {val_path}")
    print(f"Đã lưu {len(test_df)} dòng test ({len(test_df)/len(df):.1%}) vào: {test_path}")