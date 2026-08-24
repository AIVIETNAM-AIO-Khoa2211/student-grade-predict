"""Hàm đọc dữ liệu."""
import pandas as pd


def load_dataset(file_path: str):
    df = pd.read_csv(file_path)
    print(df.info())
    return df

def read_csv(file_path, target ="final_grade"):
    df = pd.read_csv(file_path)
    print(df.info())

    X = df.drop(target, axis=1)
    y = df[target] 
    print(y.value_counts())

    print("Shape df: ", df.shape)
    print("Shape X: ", X.shape)
    print("Shape y: ", y.shape)

    return X, y



