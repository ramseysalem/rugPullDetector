"""
predict.py
----------
Score a live Uniswap V2 token pair for rug pull risk.

The live prediction is explained with **SHAP** (TreeExplainer) — the "Top Risk
Factors" section reports the five features with the largest absolute SHAP
contribution to *this* prediction, along with each contribution's sign so the
user can see whether the feature pushed the model toward RUG or SAFE.

Model load order: prefers `models/xgboost_calibrated.pkl` (produced on the
rigor track) and falls back to `models/xgboost_best.pkl`.

Usage:
    python predict.py --pair 0xPAIR_ADDRESS --token 0xTOKEN_ADDRESS --eth_index 1

Arguments:
    --pair       Uniswap V2 pair contract address
    --token      ERC-20 token address (the non-WETH side)
    --eth_index  0 if token0 is WETH, 1 if token1 is WETH
"""

import argparse
import os
import pickle
import sys
import numpy as np
import pandas as pd
import shap

from features import extract_features
from train import add_derived_features


MODELS_DIR = "models"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def _load(name: str):
    """
    Load a model artefact by filename.  For the model itself, prefer the
    calibrated XGBoost (rigor track) and fall back to xgboost_best.pkl.

    Any other artefact (scaler, feature_names) loads by exact name.
    """
    if name in ("xgboost.pkl", "model.pkl", "xgboost_best.pkl"):
        calibrated = os.path.join(MODELS_DIR, "xgboost_calibrated.pkl")
        best = os.path.join(MODELS_DIR, "xgboost_best.pkl")
        if os.path.exists(calibrated):
            return _load_pickle(calibrated)
        return _load_pickle(best)
    return _load_pickle(os.path.join(MODELS_DIR, name))


def _tree_model(model):
    """SHAP TreeExplainer needs the raw boosted-tree estimator."""
    if type(model).__name__ == "CalibratedClassifierCV":
        return model.calibrated_classifiers_[0].estimator
    return model


def _risk_tier(prob: float) -> str:
    if prob >= 0.80:
        return "HIGH"
    if prob >= 0.50:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# Module-level model + explainer cache
# ---------------------------------------------------------------------------

def _init_artefacts():
    """
    Load model, scaler, feature names, and a SHAP TreeExplainer once at
    module load.  The explainer is the expensive object — building it per
    prediction would re-walk the entire forest.
    """
    try:
        model = _load("xgboost_best.pkl")
        scaler = _load("scaler.pkl")
        feature_names = _load("feature_names.pkl")
    except FileNotFoundError:
        return None, None, None, None
    explainer = shap.TreeExplainer(_tree_model(model))
    return model, scaler, feature_names, explainer


_MODEL, _SCALER, _FEATURE_NAMES, _EXPLAINER = _init_artefacts()


# ---------------------------------------------------------------------------
# SHAP-based top features
# ---------------------------------------------------------------------------

def _top_features(explainer, feature_names: list, X_scaled, raw_values,
                  n: int = 5) -> list:
    """
    Return the top n features by absolute SHAP contribution for the single
    live prediction.

    Each entry is (feature_name, raw_value, scaled_value, shap_value).
    A positive shap_value pushes the model toward RUG; negative toward SAFE.
    """
    shap_vals = np.asarray(explainer.shap_values(X_scaled))
    # TreeExplainer on a binary XGBClassifier returns a 2D array
    # (n_samples, n_features); some shap versions return a list of two arrays
    # (one per class) — collapse either case to a 1-D vector for this sample.
    if shap_vals.ndim == 3:
        shap_vals = shap_vals[1]  # positive (rug) class
    sv = shap_vals[0]

    order = np.argsort(np.abs(sv))[::-1][:n]
    out = []
    for i in order:
        out.append((feature_names[i], float(raw_values[i]),
                    float(X_scaled[0, i]), float(sv[i])))
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def predict(pair_id: str, token_id: str, eth_index: int):
    if _MODEL is None:
        print("ERROR: models/ directory not found. Run train.py first.")
        sys.exit(1)

    print(f"\nFetching on-chain data for pair {pair_id} ...")

    try:
        raw = extract_features(pair_id, token_id, eth_index)
    except Exception as e:
        print(f"ERROR: Could not extract features — {e}")
        sys.exit(1)

    df = pd.DataFrame([raw])
    df = add_derived_features(df)

    raw_values = df[_FEATURE_NAMES].values[0]
    X = _SCALER.transform(df[_FEATURE_NAMES].values)
    prob = _MODEL.predict_proba(X)[0][1]
    label = _MODEL.predict(X)[0]
    tier = _risk_tier(prob)

    # ── Output ────────────────────────────────────────────────────────────
    print()
    print("=" * 50)
    print(f"  Token   : {token_id}")
    print(f"  Pair    : {pair_id}")
    print("=" * 50)
    print(f"  Rug Pull Probability : {prob:.1%}")
    print(f"  Prediction           : {'RUG PULL' if label == 1 else 'NORMAL'}")
    print(f"  Risk Tier            : {tier}")
    print("=" * 50)

    print("\n  Top Risk Factors:")
    top = _top_features(_EXPLAINER, _FEATURE_NAMES, X, raw_values)
    for i, (name, raw_val, _scaled, shap_val) in enumerate(top, 1):
        direction = "pushes toward RUG" if shap_val > 0 else "pushes toward SAFE"
        print(
            f"    {i}. {name:<40} value: {raw_val:>10.4f}   "
            f"SHAP: {shap_val:+.4f}   ({direction})"
        )

    print()
    return prob


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rug pull risk scorer")
    parser.add_argument("--pair",      required=True, help="Uniswap V2 pair address")
    parser.add_argument("--token",     required=True, help="ERC-20 token address")
    parser.add_argument("--eth_index", required=True, type=int, choices=[0, 1],
                        help="0 if token0 is WETH, 1 if token1 is WETH")
    args = parser.parse_args()

    predict(args.pair, args.token, args.eth_index)
