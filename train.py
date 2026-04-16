"""
train.py
--------
Full training pipeline for rug pull detection.

Dataset: Dataset_v1.9.csv  (18,296 tokens — 90% rug pulls, 10% normal)
Models : Logistic Regression → Random Forest → XGBoost (primary)
Output : models/xgboost_best.pkl  + models/scaler.pkl
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

DATASET_PATH = "Dataset_v1.9.csv"
MODELS_DIR = "models"
RANDOM_STATE = 42

# ── Original 18 features ────────────────────────────────────────────────────
BASE_FEATURES = [
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


# ── Feature engineering ──────────────────────────────────────────────────────

def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute 8 derived features that capture rug pull mechanics more directly
    than the raw components alone.

    unlocked_creator_lp   — creator holds LP AND it is not locked.
                            This is the most direct mechanism of a rug pull.
    burn_to_mint_ratio    — how many LP burns happen per mint event.
                            High ratio → liquidity is being pulled out faster
                            than it is being added.
    total_swap_volume     — combined buy + sell activity per week.
                            Very low activity is a red flag (abandoned token).
    creator_total_exposure— creator's combined stake across token supply + LP.
                            High exposure → one actor controls the exit.
    sell_dominance        — fraction of swaps that are sell-side.
                            Near 1.0 → everyone is trying to exit.
    swap_timing_delay     — how much later swaps occur relative to mints
                            (normalised). Large gap = pump-and-dump timing.
    lp_exposure_risk      — hard binary: creator holds >50% of unlocked LP.
                            The clearest single-feature rug pull indicator.
    honeypot_signal       — buy-side swaps per unit of sell-side activity.
                            Very high → sells are blocked (honeypot).
    """
    df = df.copy()

    df["unlocked_creator_lp"] = (
        df["lp_creator_holding_ratio"] * (1.0 - df["lp_lock_ratio"].clip(0, 1))
    )

    df["burn_to_mint_ratio"] = (
        df["burn_count_per_week"] / (df["mint_count_per_week"] + 1e-6)
    )

    df["total_swap_volume"] = df["swap_in_per_week"] + df["swap_out_per_week"]

    df["creator_total_exposure"] = (
        df["token_creator_holding_ratio"] + df["lp_creator_holding_ratio"]
    ).clip(0, 1)

    total_swaps = df["swap_in_per_week"] + df["swap_out_per_week"] + 1e-6
    df["sell_dominance"] = df["swap_out_per_week"] / total_swaps

    # How much later do swaps start compared to first mints (both normalised 0-1)?
    df["swap_timing_delay"] = (
        df["swap_mean_period"] - df["mint_mean_period"]
    ).clip(-1, 1)

    # 1 if creator controls >50% of unlocked LP, 0 otherwise
    df["lp_exposure_risk"] = (
        (df["lp_creator_holding_ratio"] > 50)
        & (df["lp_lock_ratio"] < 0.1)
    ).astype(float)

    # Buy swaps per sell swap — very high values indicate sell is blocked
    df["honeypot_signal"] = (
        df["swap_in_per_week"] / (df["swap_out_per_week"] + 1e-6)
    ).clip(0, 100)

    return df


DERIVED_FEATURES = [
    "unlocked_creator_lp",
    "burn_to_mint_ratio",
    "total_swap_volume",
    "creator_total_exposure",
    "sell_dominance",
    "swap_timing_delay",
    "lp_exposure_risk",
    "honeypot_signal",
]

ALL_FEATURES = BASE_FEATURES + DERIVED_FEATURES


# ── Data loading ─────────────────────────────────────────────────────────────

def load_dataset() -> tuple[pd.DataFrame, np.ndarray, np.ndarray, list[str]]:
    """
    Returns (df_features, X, y, feature_names).
    X is raw (unscaled) so tree models can use it directly.
    """
    df = pd.read_csv(DATASET_PATH)

    # TRUE = rug pull = 1,  FALSE = normal = 0
    y = (
        df["Label"]
        .map({True: 1, False: 0, "True": 1, "False": 0})
        .astype(int)
        .values
    )

    df = add_derived_features(df)
    X = df[ALL_FEATURES].values

    print(f"Dataset loaded: {len(df):,} rows")
    print(f"  Rug pull (1): {y.sum():,}  ({y.mean():.1%})")
    print(f"  Normal   (0): {(y == 0).sum():,}  ({1 - y.mean():.1%})")
    print(f"  Features    : {len(ALL_FEATURES)}")

    return df[ALL_FEATURES], X, y, ALL_FEATURES


# ── Evaluation helper ────────────────────────────────────────────────────────

def _print_metrics(name: str, y_true, y_pred, y_prob) -> dict:
    auc = roc_auc_score(y_true, y_prob)
    f1 = f1_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    print(f"\n{'─'*55}")
    print(f"  {name}")
    print(f"{'─'*55}")
    print(classification_report(y_true, y_pred, target_names=["normal", "rug_pull"]))
    print(f"  AUC-ROC : {auc:.4f}")
    return {"model": name, "precision": prec, "recall": rec, "f1": f1, "auc": auc}


# ── Training ─────────────────────────────────────────────────────────────────

def train_all():
    os.makedirs(MODELS_DIR, exist_ok=True)

    df_feat, X, y, feature_names = load_dataset()

    # Stratified 80/20 split — preserve the 90/10 imbalance ratio
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # Scale (needed for logistic regression; harmless for trees)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_raw)
    X_test_scaled = scaler.transform(X_test_raw)

    # SMOTE — oversample the minority class (normal tokens) on training set only
    smote = SMOTE(random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)
    print(f"\nAfter SMOTE → training set: {len(X_train_bal):,} samples "
          f"({(y_train_bal==1).sum():,} rug / {(y_train_bal==0).sum():,} normal)")

    results = []

    # ── 1. Logistic Regression (simple baseline) ──────────────────────────
    print("\n[1/3] Training Logistic Regression …")
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    lr.fit(X_train_bal, y_train_bal)
    results.append(_print_metrics(
        "Logistic Regression",
        y_test,
        lr.predict(X_test_scaled),
        lr.predict_proba(X_test_scaled)[:, 1],
    ))

    # ── 2. Random Forest (strong baseline) ───────────────────────────────
    print("\n[2/3] Training Random Forest …")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",   # extra guard beyond SMOTE
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    rf.fit(X_train_bal, y_train_bal)
    results.append(_print_metrics(
        "Random Forest",
        y_test,
        rf.predict(X_test_scaled),
        rf.predict_proba(X_test_scaled)[:, 1],
    ))

    # ── 3. XGBoost — hyperparameter search then final fit ─────────────────
    print("\n[3/3] Tuning XGBoost (GridSearchCV, 5-fold) …")

    # scale_pos_weight = negative / positive in the *original* training set
    # (before SMOTE — we let SMOTE handle balance and keep this at 1 for
    # the balanced set, or set it to counteract residual bias)
    neg_orig = (y_train == 0).sum()
    pos_orig = (y_train == 1).sum()
    spw = neg_orig / pos_orig  # ~0.11 here (minority is normal)

    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }

    base_xgb = XGBClassifier(
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(
        base_xgb,
        param_grid,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
        verbose=1,
    )
    grid.fit(X_train_bal, y_train_bal)

    print(f"\n  Best params : {grid.best_params_}")
    print(f"  Best CV F1  : {grid.best_score_:.4f}")

    best_xgb = grid.best_estimator_
    xgb_pred = best_xgb.predict(X_test_scaled)
    xgb_prob = best_xgb.predict_proba(X_test_scaled)[:, 1]
    results.append(_print_metrics("XGBoost (tuned)", y_test, xgb_pred, xgb_prob))

    # ── Model comparison table ────────────────────────────────────────────
    print("\n" + "═" * 65)
    print(f"  {'Model':<28} {'Precision':>9} {'Recall':>8} {'F1':>8} {'AUC':>8}")
    print("═" * 65)
    for r in results:
        print(f"  {r['model']:<28} {r['precision']:>9.4f} {r['recall']:>8.4f} "
              f"{r['f1']:>8.4f} {r['auc']:>8.4f}")
    print("═" * 65)

    # ── Save artefacts ────────────────────────────────────────────────────
    with open(os.path.join(MODELS_DIR, "xgboost_best.pkl"), "wb") as f:
        pickle.dump(best_xgb, f)
    with open(os.path.join(MODELS_DIR, "random_forest.pkl"), "wb") as f:
        pickle.dump(rf, f)
    with open(os.path.join(MODELS_DIR, "logistic_regression.pkl"), "wb") as f:
        pickle.dump(lr, f)
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # Save feature names for evaluation script
    with open(os.path.join(MODELS_DIR, "feature_names.pkl"), "wb") as f:
        pickle.dump(feature_names, f)

    print(f"\nModels saved to ./{MODELS_DIR}/")

    return {
        "models": {"lr": lr, "rf": rf, "xgb": best_xgb},
        "scaler": scaler,
        "test_data": (X_test_scaled, y_test),
        "feature_names": feature_names,
        "results": results,
    }


if __name__ == "__main__":
    train_all()
