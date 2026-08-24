import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from boruta import BorutaPy
from src.config import RANDOM_STATE

def boruta_filt(
        df: pd.DataFrame,
        target_col = "final_grade",
        max_depth: int = 5
)->pd.DataFrame:
    X = df.drop(columns=[target_col])
    y = df[target_col]

    feature_names = X.columns.to_list()
    #Khởi tạo mô hình để random forest để lọc

    rf = RandomForestClassifier(
        n_jobs= -1,
        max_depth=5,
        random_state=RANDOM_STATE,
        class_weight="balanced"
    )

    boruta_selector = BorutaPy(
        estimator=rf,
        random_state=RANDOM_STATE,
        verbose=0
    )

    boruta_selector.fit(X.to_numpy(), y.to_numpy())

    selected_features = [
        name for name, keep in zip(feature_names, boruta_selector.support_) if keep
    ]
 
    if not selected_features:
        raise RuntimeError(
            "Boruta không chọn được đặc trưng nào quan trọng. "
            "Hãy kiểm tra lại dữ liệu đầu vào hoặc tham số của mô hình."
        )
 
    reduced_df = df[selected_features + [target_col]].copy()
 
    return reduced_df