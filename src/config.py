from pathlib import Path

# Đường dẫn gốc của project
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DATA_RAW_DIR = ROOT_DIR / "data" / "raw"
DATA_PROCESSED_DIR = ROOT_DIR / "data" / "processed"
DATA_FILTERED_DIR = ROOT_DIR / "data" / "filtered"
RESULTS_DIR = ROOT_DIR / "results"

# Tham số chung
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.1

# Tham số tối ưu cho từng model (tìm được qua hyperparameter_tuning.ipynb, đánh giá trên tập val)

DECISION_TREE_PARAMS = {
    "max_depth": 7,
    "min_samples_leaf": 1,
    "min_samples_split": 5,
}

RANDOM_FOREST_PARAMS = {
    "n_estimators": 100,
    "max_depth": 2,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
}

ADABOOST_PARAMS = {
    "n_estimators": 150,
    "learning_rate": 0.5,
    "base_max_depth": 2,
}

DECISION_TREE_FE_PARAMS = {
    "max_depth": 7,
    "min_samples_leaf": 3,
    "min_samples_split": 2,
}

RANDOM_FOREST_FE_PARAMS = {
    "n_estimators": 100,
    "max_depth": 8,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
}

ADABOOST_FE_PARAMS = {
    "n_estimators": 150,
    "learning_rate": 1,
    "base_max_depth": 3,
}
DECISION_TREE_FI_PARAMS = {
    "max_depth": 7,
    "min_samples_leaf": 3,
    "min_samples_split": 2,
}

RANDOM_FOREST_FI_PARAMS = {
    "n_estimators": 100,
    "max_depth": 8,
    "min_samples_leaf": 1,
    "max_features": "sqrt",
}

ADABOOST_FI_PARAMS = {
    "n_estimators": 150,
    "learning_rate": 1,
    "base_max_depth": 3,
}

LOG_REG_PARAM = {
    "C": 0.1,
    "l1_ratio": 0.5,
    "max_iter": 1000,
    "solver": "saga" 
}

