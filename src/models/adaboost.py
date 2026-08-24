"""Khởi tạo và train mô hình AdaBoost."""
import pandas as pd
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

from src.config import RANDOM_STATE

def evaluate_val_adaboost(X_train, y_train, X_val, y_val, base_max_depth=1, isprint = True, **params):
    # Huấn luyện AdaBoost với base estimator là decision stump (mặc định max_depth=1)
    base_estimator = DecisionTreeClassifier(max_depth=base_max_depth, random_state=RANDOM_STATE)
    ada_model = AdaBoostClassifier(estimator=base_estimator, random_state=RANDOM_STATE, **params)
    ada_model.fit(X_train, y_train)

    # Dự đoán và đánh giá trên tập val
    ada_pred = ada_model.predict(X_val)
    ada_accuracy = accuracy_score(y_val, ada_pred)
    if isprint:
        print(f"Độ chính xác AdaBoost: {ada_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_val, ada_pred, zero_division=0))
    return ada_model, ada_accuracy

def concat_and_train_AB(X_train, y_train, X_val, y_val,base_max_depth=1, isprint=True, **params):
    X_combined = pd.concat([X_train, X_val], ignore_index=True)
    y_combined = pd.concat([y_train, y_val], ignore_index=True)

    base_estimator = DecisionTreeClassifier(max_depth=base_max_depth, random_state=RANDOM_STATE)
    model = AdaBoostClassifier(estimator=base_estimator, random_state=RANDOM_STATE, **params)
    model.fit(X_combined, y_combined)

    if isprint:
        print(f"Đã train AdaBoost (base_max_depth={base_max_depth}) trên {len(X_combined)} mẫu (train + val).")

    return model