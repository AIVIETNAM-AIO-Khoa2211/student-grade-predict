"""Khởi tạo và train mô hình Logistic Regression (linear classifier)."""
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

from src.config import RANDOM_STATE


def evaluate_val_logistic_regression(X_train, y_train, X_val, y_val, verbose=True, **params):
    # Chuẩn hóa dữ liệu — bắt buộc với model tuyến tính, khác với model dạng cây
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)  # chỉ transform, không fit lại trên val

    # Huấn luyện Logistic Regression
    lr_model = LogisticRegression(random_state=RANDOM_STATE, **params, tol=0.002)
    lr_model.fit(X_train_scaled, y_train)

    # Dự đoán và đánh giá trên tập val
    lr_pred = lr_model.predict(X_val_scaled)
    lr_accuracy = accuracy_score(y_val, lr_pred)

    if verbose:
        print(f"Độ chính xác Logistic Regression: {lr_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_val, lr_pred, zero_division=0))

    return lr_model, lr_accuracy

def concat_and_train_logistic(X_train, y_train, X_val, y_val, verbose=True, **params):
    # 1. Gộp DataFrame trước khi scale
    X_combined_raw = pd.concat([X_train, X_val], ignore_index=True)
    y_combined = pd.concat([y_train, y_val], ignore_index=True)

    # 2. Fit và transform trên toàn bộ dữ liệu đã gộp
    scaler = StandardScaler()
    X_combined_scaled = scaler.fit_transform(X_combined_raw)

    # 3. Train mô hình
    model = LogisticRegression(random_state=RANDOM_STATE, **params, tol=0.002)
    model.fit(X_combined_scaled, y_combined)

    if verbose:
        print(f"Đã train Logistic Regression trên {len(X_combined_scaled)} mẫu (train + val).")

    return model, scaler