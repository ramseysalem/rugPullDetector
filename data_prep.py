"""
data_prep.py
------------
Loads and preprocesses the rug pull dataset for model training/evaluation.

Dataset: Dataset_v1.9.csv
  - 18,296 rows  (16,462 rug pulls, 1,834 normal)
  - 18 numeric features + 1 id column + 1 Label column
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


DATASET_PATH = "Dataset_v1.9.csv"

FEATURE_COLUMNS = [
    "mint_count_per_week",
    "burn_count_per_week",
    "mint_ratio",
    "swap_ratio",
    "burn_ratio",
    "mint_mean_period",
    "swap_mean_period",
    "burn_mean_period",
    "swap_in_per_week",
    "swap_out_per_week",
    "swap_rate",
    "lp_avg",
    "lp_std",
    "lp_creator_holding_ratio",
    "lp_lock_ratio",
    "token_burn_ratio",
    "token_creator_holding_ratio",
    "number_of_token_creation_of_creator",
]


def load_raw() -> pd.DataFrame:
    """Return the full dataset with id, Label, and all features."""
    df = pd.read_csv(DATASET_PATH)
    return df


def load_data(
    test_size: float = 0.2,
    random_state: int = 42,
    normalize: bool = True,
) -> tuple:
    """
    Load, clean, optionally normalize, and split the dataset.

    Returns
    -------
    X_train, X_test, y_train, y_test  (numpy arrays)
    """
    df = pd.read_csv(DATASET_PATH)

    # Drop non-feature columns
    df = df.drop(columns=["id"])
    df = df.dropna(how="any", axis=0)

    # Label: TRUE = rug pull (1), FALSE = normal (0)
    y = df["Label"].map({True: 1, False: 0, "True": 1, "False": 0}).astype(int).values
    X = df[FEATURE_COLUMNS].values

    if normalize:
        scaler = MinMaxScaler()
        X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def load_normalized_with_scaler(
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Same as load_data but also returns the fitted scaler — useful when you
    need to transform new live data the same way as training data.

    Returns
    -------
    X_train, X_test, y_train, y_test, scaler
    """
    df = pd.read_csv(DATASET_PATH)
    df = df.drop(columns=["id"])
    df = df.dropna(how="any", axis=0)

    y = df["Label"].map({True: 1, False: 0, "True": 1, "False": 0}).astype(int).values
    X = df[FEATURE_COLUMNS].values

    scaler = MinMaxScaler()
    X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, scaler


def class_distribution() -> dict:
    """Print and return the class distribution of the dataset."""
    df = pd.read_csv(DATASET_PATH)
    counts = df["Label"].value_counts()
    dist = {
        "rug_pull (TRUE)": int(counts.get(True, counts.get("True", 0))),
        "normal (FALSE)": int(counts.get(False, counts.get("False", 0))),
        "total": len(df),
    }
    print(f"  Rug pull  : {dist['rug_pull (TRUE)']} ({dist['rug_pull (TRUE)']/dist['total']*100:.1f}%)")
    print(f"  Normal    : {dist['normal (FALSE)']} ({dist['normal (FALSE)']/dist['total']*100:.1f}%)")
    print(f"  Total     : {dist['total']}")
    return dist


if __name__ == "__main__":
    print("=== Dataset overview ===")
    class_distribution()

    print("\n=== Train/test split (80/20, normalized) ===")
    X_train, X_test, y_train, y_test = load_data()
    print(f"  X_train: {X_train.shape}  y_train: {y_train.shape}")
    print(f"  X_test : {X_test.shape}   y_test : {y_test.shape}")

    print("\n=== Feature columns ===")
    for i, col in enumerate(FEATURE_COLUMNS):
        print(f"  [{i:02d}] {col}")
