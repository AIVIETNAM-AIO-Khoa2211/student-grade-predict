"""Khởi tạo và train mô hình Decision Tree."""
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.config import RANDOM_STATE


def evaluate_val_decision_tree(X_train, y_train, X_val, y_val, isprint = True, **params):
    # Huấn luyện Decision Tree
    dt_model = DecisionTreeClassifier(random_state=RANDOM_STATE, **params)
    dt_model.fit(X_train, y_train)

    # Dự đoán và đánh giá trên tập val
    dt_pred = dt_model.predict(X_val)
    dt_accuracy = accuracy_score(y_val, dt_pred)
    if isprint:
        print(f"Độ chính xác Decision Tree: {dt_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_val, dt_pred, zero_division=0))
    return dt_model, dt_accuracy


def concat_and_train_DT(X_train, y_train, X_val, y_val, isprint=True, **params):
    X_combined = pd.concat([X_train, X_val], ignore_index=True)
    y_combined = pd.concat([y_train, y_val], ignore_index=True)

    model = DecisionTreeClassifier(random_state=RANDOM_STATE, **params)
    model.fit(X_combined, y_combined)

    if isprint:
        print(f"Đã train Decision Tree trên {len(X_combined)} mẫu (train + val).")

    return model