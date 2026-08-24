"""Hàm trực quan hóa dữ liệu."""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def heatmap(df: pd.DataFrame, cols: list = None, figsize: tuple = (10, 8)) -> None:
    """Vẽ heatmap ma trận tương quan (correlation matrix) giữa các feature.

    Mục đích: phát hiện các cặp feature tương quan cao (đa cộng tuyến),
    hoặc feature tương quan quá cao với target -> nghi ngờ DATA LEAKAGE.
    """
    data = df[cols] if cols is not None else df.select_dtypes(include="number")
    corr = data.corr()

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1, vmax=1,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Ma trận tương quan giữa các feature")
    plt.tight_layout()
    plt.show()


def plot_distribution(df: pd.DataFrame, cols: list = None, n_cols: int = 3, figsize_per_plot: tuple = (4, 3)) -> None:
    """Vẽ phân bố (histogram) của từng feature dạng lưới subplot.

    Mục đích: kiểm tra hình dạng phân phối (lệch, outlier, đa đỉnh...)
    trước khi quyết định scaling / transform cho model.
    """
    cols = cols if cols is not None else df.select_dtypes(include="number").columns.tolist()
    n = len(cols)
    n_rows = -(-n // n_cols)  # ceil division

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(figsize_per_plot[0] * n_cols, figsize_per_plot[1] * n_rows),
    )
    axes = axes.flatten() if n > 1 else [axes]

    for i, col in enumerate(cols):
        sns.histplot(df[col], kde=True, ax=axes[i])
        axes[i].set_title(f"Phân bố {col}")
        axes[i].set_xlabel(col)

    # ẩn các subplot thừa nếu số feature không chia hết cho n_cols
    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()