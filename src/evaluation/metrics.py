"""Hàm tính accuracy, F1, confusion matrix..."""
from sklearn.metrics import accuracy_score, classification_report

def evaluate_test(model, X_test, y_test, model_name="Model"):
    # Dự đoán và đánh giá trên tập test
    test_pred = model.predict(X_test)
    test_accuracy = accuracy_score(y_test, test_pred)
    print(f"Độ chính xác {model_name} trên tập test: {test_accuracy:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, test_pred, zero_division=0))
    return test_accuracy