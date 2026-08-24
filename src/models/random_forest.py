"""Khởi tạo và train mô hình Random Forest."""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.config import RANDOM_STATE

def evaluate_val_random_forest(X_train, y_train, X_val, y_val, isprint = True, **params):
    # Huấn luyện Random Forest
    rf_model = RandomForestClassifier(random_state=RANDOM_STATE, **params)
    rf_model.fit(X_train, y_train)

    # Dự đoán và đánh giá trên tập val
    rf_pred = rf_model.predict(X_val)
    rf_accuracy = accuracy_score(y_val, rf_pred)
    if isprint:
        print(f"Độ chính xác Random Forest: {rf_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_val, rf_pred, zero_division=0))
    return rf_model, rf_accuracy


def concat_and_train_RF(X_train, y_train, X_val, y_val, isprint=True, **params):
    X_combined = pd.concat([X_train, X_val], ignore_index=True)
    y_combined = pd.concat([y_train, y_val], ignore_index=True)

    model = RandomForestClassifier(random_state=RANDOM_STATE, **params)
    model.fit(X_combined, y_combined)

    if isprint:
        print(f"Đã train Random Forest trên {len(X_combined)} mẫu (train + val).")

    return model
