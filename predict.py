"""
predict.py
----------
Score a live Uniswap V2 token pair for rug pull risk.

Usage:
    python predict.py --pair 0xPAIR_ADDRESS --token 0xTOKEN_ADDRESS --eth_index 1

Arguments:
    --pair       Uniswap V2 pair contract address
    --token      ERC-20 token address (the non-WETH side)
    --eth_index  0 if token0 is WETH, 1 if token1 is WETH
"""

import argparse
import pickle
import sys
import pandas as pd

from features import extract_features
from train import add_derived_features


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load(name: str):
    with open(f"models/{name}", "rb") as f:
        return pickle.load(f)


def _risk_tier(prob: float) -> str:
    if prob >= 0.80:
        return "HIGH"
    if prob >= 0.50:
        return "MEDIUM"
    return "LOW"


def _top_features(model, feature_names: list, X_scaled, n: int = 5) -> list:
    """Return the top n features driving this prediction, with their values."""
    importances = model.feature_importances_
    # Weight importance by the scaled feature value so high-value features rank higher
    weighted = importances * abs(X_scaled[0])
    ranked = sorted(zip(feature_names, weighted, X_scaled[0]), key=lambda x: x[1], reverse=True)
    return ranked[:n]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def predict(pair_id: str, token_id: str, eth_index: int):
    # Load artefacts
    try:
        model        = _load("xgboost_best.pkl")
        scaler       = _load("scaler.pkl")
        feature_names = _load("feature_names.pkl")
    except FileNotFoundError:
        print("ERROR: models/ directory not found. Run train.py first.")
        sys.exit(1)

    print(f"\nFetching on-chain data for pair {pair_id} ...")

    # Extract raw features
    try:
        raw = extract_features(pair_id, token_id, eth_index)
    except Exception as e:
        print(f"ERROR: Could not extract features — {e}")
        sys.exit(1)

    # Add derived features
    df = pd.DataFrame([raw])
    df = add_derived_features(df)

    # Scale and predict
    X = scaler.transform(df[feature_names].values)
    prob  = model.predict_proba(X)[0][1]
    label = model.predict(X)[0]
    tier  = _risk_tier(prob)

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
    top = _top_features(model, feature_names, X)
    for i, (name, _, value) in enumerate(top, 1):
        print(f"    {i}. {name:<40} value: {value:.4f}")

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
