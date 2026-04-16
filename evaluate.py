"""
evaluate.py
-----------
Generate all evaluation plots for the rug pull detection models.

Requires models/ directory to be populated by train.py first.

Outputs (saved to plots/):
  confusion_matrix_<model>.png
  roc_curves.png
  precision_recall_curves.png
  feature_importance_xgb.png
  feature_importance_rf.png
  correlation_matrix.png
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score,
)

from train import load_dataset, RANDOM_STATE
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

MODELS_DIR = "models"
PLOTS_DIR = "plots"
CLASS_NAMES = ["Normal", "Rug Pull"]


def _load_artefacts():
    def _load(name):
        with open(os.path.join(MODELS_DIR, name), "rb") as f:
            return pickle.load(f)

    return {
        "xgb": _load("xgboost_best.pkl"),
        "rf": _load("random_forest.pkl"),
        "lr": _load("logistic_regression.pkl"),
        "scaler": _load("scaler.pkl"),
        "feature_names": _load("feature_names.pkl"),
    }


def _get_test_data(scaler):
    _, X, y, _ = load_dataset()
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    return scaler.transform(X_test), y_test


# ── Individual confusion matrices ────────────────────────────────────────────

def plot_confusion_matrices(models: dict, X_test, y_test):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    titles = {
        "lr": "Logistic Regression",
        "rf": "Random Forest",
        "xgb": "XGBoost (tuned)",
    }
    for ax, (key, model) in zip(axes, models.items()):
        cm = confusion_matrix(y_test, model.predict(X_test))
        disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(titles[key], fontsize=13, fontweight="bold")

    fig.suptitle("Confusion Matrices — Test Set", fontsize=15, y=1.02)
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


# ── ROC curves (all models on one chart) ────────────────────────────────────

def plot_roc_curves(models: dict, X_test, y_test):
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"lr": "#e74c3c", "rf": "#2ecc71", "xgb": "#3498db"}
    labels = {
        "lr": "Logistic Regression",
        "rf": "Random Forest",
        "xgb": "XGBoost (tuned)",
    }
    for key, model in models.items():
        prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[key], lw=2,
                label=f"{labels[key]}  (AUC = {roc_auc:.4f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1.2, label="Random classifier")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves — Rug Pull Detection", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    path = os.path.join(PLOTS_DIR, "roc_curves.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── Precision-Recall curves ──────────────────────────────────────────────────

def plot_pr_curves(models: dict, X_test, y_test):
    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"lr": "#e74c3c", "rf": "#2ecc71", "xgb": "#3498db"}
    labels = {
        "lr": "Logistic Regression",
        "rf": "Random Forest",
        "xgb": "XGBoost (tuned)",
    }
    baseline = y_test.mean()
    ax.axhline(y=baseline, color="k", linestyle="--", lw=1.2,
               label=f"Baseline (prevalence = {baseline:.2f})")

    for key, model in models.items():
        prob = model.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, prob)
        ap = average_precision_score(y_test, prob)
        ax.plot(rec, prec, color=colors[key], lw=2,
                label=f"{labels[key]}  (AP = {ap:.4f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — Rug Pull Detection", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    path = os.path.join(PLOTS_DIR, "precision_recall_curves.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── XGBoost feature importance ───────────────────────────────────────────────

def plot_feature_importance_xgb(model, feature_names: list):
    importances = model.feature_importances_
    indices = np.argsort(importances)  # ascending → best at bottom in barh

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#e74c3c" if importances[i] >= np.percentile(importances, 80)
              else "#95a5a6" for i in indices]
    ax.barh(range(len(indices)), importances[indices], color=colors, edgecolor="white")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=10)
    ax.set_xlabel("Feature Importance (gain)", fontsize=12)
    ax.set_title("XGBoost Feature Importance — All Features", fontsize=14, fontweight="bold")
    ax.axvline(x=np.percentile(importances, 80), color="#e74c3c",
               linestyle="--", lw=1, alpha=0.6, label="Top-20% threshold")
    ax.legend(fontsize=10)
    ax.grid(axis="x", alpha=0.3)

    path = os.path.join(PLOTS_DIR, "feature_importance_xgb.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── Random Forest feature importance ────────────────────────────────────────

def plot_feature_importance_rf(model, feature_names: list):
    importances = model.feature_importances_
    indices = np.argsort(importances)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(range(len(indices)), importances[indices], color="#2ecc71", edgecolor="white")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=10)
    ax.set_xlabel("Feature Importance (mean decrease in impurity)", fontsize=12)
    ax.set_title("Random Forest Feature Importance", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    path = os.path.join(PLOTS_DIR, "feature_importance_rf.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── Feature correlation heatmap ──────────────────────────────────────────────

def plot_correlation_matrix(feature_names: list):
    from train import add_derived_features
    df = pd.read_csv("Dataset_v1.9.csv")
    df = add_derived_features(df)
    corr = df[feature_names].corr()

    fig, ax = plt.subplots(figsize=(14, 12))
    mask = np.triu(np.ones_like(corr, dtype=bool))  # upper triangle
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
        annot_kws={"size": 7},
    )
    ax.set_title("Feature Correlation Matrix", fontsize=14, fontweight="bold")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=8)

    path = os.path.join(PLOTS_DIR, "correlation_matrix.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── Feature distribution by class ───────────────────────────────────────────

def plot_feature_distributions(feature_names: list, top_n: int = 6):
    """Plot the 6 most important features, colored by class label."""
    from train import add_derived_features
    df = pd.read_csv("Dataset_v1.9.csv")
    df = add_derived_features(df)
    df["label"] = df["Label"].map({True: 1, False: 0, "True": 1, "False": 0})

    # Use the features that differ most between classes (by mean difference)
    diffs = {}
    for col in feature_names:
        mu_rug = df.loc[df["label"] == 1, col].median()
        mu_norm = df.loc[df["label"] == 0, col].median()
        diffs[col] = abs(mu_rug - mu_norm) / (df[col].std() + 1e-9)
    top_feats = sorted(diffs, key=diffs.get, reverse=True)[:top_n]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for ax, feat in zip(axes, top_feats):
        rug = df.loc[df["label"] == 1, feat].clip(
            df[feat].quantile(0.01), df[feat].quantile(0.99)
        )
        norm = df.loc[df["label"] == 0, feat].clip(
            df[feat].quantile(0.01), df[feat].quantile(0.99)
        )
        ax.hist(rug, bins=50, alpha=0.6, color="#e74c3c", label="Rug pull", density=True)
        ax.hist(norm, bins=50, alpha=0.6, color="#3498db", label="Normal", density=True)
        ax.set_title(feat, fontsize=11)
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.3)

    fig.suptitle("Top Feature Distributions by Class", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(PLOTS_DIR, "feature_distributions.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"  Saved: {path}")


# ── Main ─────────────────────────────────────────────────────────────────────

def run_all():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    print("Loading saved models …")
    artefacts = _load_artefacts()
    models = {k: artefacts[k] for k in ("lr", "rf", "xgb")}
    scaler = artefacts["scaler"]
    feature_names = artefacts["feature_names"]

    print("Reconstructing test set …")
    X_test, y_test = _get_test_data(scaler)

    print("\nGenerating plots …")
    plot_confusion_matrices(models, X_test, y_test)
    plot_roc_curves(models, X_test, y_test)
    plot_pr_curves(models, X_test, y_test)
    plot_feature_importance_xgb(artefacts["xgb"], feature_names)
    plot_feature_importance_rf(artefacts["rf"], feature_names)
    plot_correlation_matrix(feature_names)
    plot_feature_distributions(feature_names)

    print(f"\nAll plots saved to ./{PLOTS_DIR}/")


if __name__ == "__main__":
    run_all()
