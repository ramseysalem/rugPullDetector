"""
shap_analysis.py
----------------
Compute SHAP attributions for the production rug-pull model on the held-out
test set.

Outputs:
  plots/shap_summary.png            — bee-swarm summary across all features
  plots/shap_waterfall_example.png  — per-sample waterfall for the highest
                                      predicted rug probability in the test set
  shap_summary.md                   — top-5 feature ranking + interpretations,
                                      compared against gain-based importance
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

from sklearn.model_selection import train_test_split

from train import load_dataset, RANDOM_STATE


warnings.filterwarnings("ignore")

MODELS_DIR = "models"
PLOTS_DIR = "plots"
SAMPLE_LIMIT = 1000  # stratified sample size if full test set is too slow


# ── Loading ──────────────────────────────────────────────────────────────────

def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_model_with_fallback() -> tuple:
    calibrated_path = os.path.join(MODELS_DIR, "xgboost_calibrated.pkl")
    best_path = os.path.join(MODELS_DIR, "xgboost_best.pkl")
    if os.path.exists(calibrated_path):
        print(f"Loaded model: {calibrated_path}")
        return _load_pickle(calibrated_path), "xgboost_calibrated.pkl"
    print(f"Loaded model: {best_path}  (calibrated model not present)")
    return _load_pickle(best_path), "xgboost_best.pkl"


def get_tree_model(model):
    """
    SHAP's TreeExplainer needs the raw boosted-tree estimator.  If the loaded
    model is a CalibratedClassifierCV wrapper, peel it open.
    """
    cls = type(model).__name__
    if cls == "CalibratedClassifierCV":
        inner = model.calibrated_classifiers_[0].estimator
        print(f"Unwrapped CalibratedClassifierCV → {type(inner).__name__}")
        return inner
    return model


# ── SHAP computation ─────────────────────────────────────────────────────────

def maybe_subsample(X: np.ndarray, y: np.ndarray, limit: int,
                    rng_seed: int = RANDOM_STATE) -> tuple:
    """Stratified subsample if X is too large to explain quickly."""
    if len(X) <= limit:
        return X, y, np.arange(len(X))
    rng = np.random.default_rng(rng_seed)
    idx = []
    for cls in (0, 1):
        cls_idx = np.where(y == cls)[0]
        n_take = int(round(limit * len(cls_idx) / len(X)))
        n_take = max(n_take, 1)
        idx.extend(rng.choice(cls_idx, size=n_take, replace=False))
    idx = np.array(sorted(idx))
    return X[idx], y[idx], idx


# ── Markdown summary ─────────────────────────────────────────────────────────

INTERPRETATION = {
    "unlocked_creator_lp":
        "Creator-held LP that is *not* locked — direct rug-pull capability; high values push toward RUG.",
    "lp_creator_holding_ratio":
        "Share of LP tokens held by the creator; concentrated creator LP pushes toward RUG.",
    "lp_lock_ratio":
        "Fraction of LP tokens parked in a known locker; high values push toward SAFE.",
    "token_creator_holding_ratio":
        "Share of token supply held by the creator; concentration pushes toward RUG.",
    "creator_total_exposure":
        "Combined token + LP creator stake; high single-actor exposure pushes toward RUG.",
    "lp_exposure_risk":
        "Binary flag for >50% unlocked creator LP; firing pushes toward RUG.",
    "burn_to_mint_ratio":
        "LP burns per mint; high values mean liquidity is leaving faster than entering and push toward RUG.",
    "sell_dominance":
        "Sell-side share of swaps; near 1.0 means everyone is exiting and pushes toward RUG.",
    "swap_timing_delay":
        "Gap between mean swap time and mean mint time; large gaps fit pump-and-dump and push toward RUG.",
    "honeypot_signal":
        "Buy-swaps per sell-swap; very high values suggest sells are blocked (honeypot) and push toward RUG.",
    "total_swap_volume":
        "Combined buy+sell swaps per week; very low activity fits abandoned tokens and pushes toward RUG.",
    "mint_count_per_week":
        "LP add events per week; abnormal patterns (very low or very spiky) push toward RUG.",
    "burn_count_per_week":
        "LP remove events per week; high values push toward RUG.",
    "swap_in_per_week":
        "Buy-side swaps per week; low values push toward RUG.",
    "swap_out_per_week":
        "Sell-side swaps per week; high values push toward RUG.",
    "swap_rate":
        "Buy/sell pressure ratio; very low values push toward RUG.",
    "mint_ratio":
        "Mints as a share of all events; abnormal share pushes toward RUG.",
    "swap_ratio":
        "Swaps as a share of all events; very low values fit abandoned tokens.",
    "burn_ratio":
        "Burns as a share of all events; high values push toward RUG.",
    "mint_mean_period":
        "Average normalised time of mints; mints clustered late push toward RUG.",
    "swap_mean_period":
        "Average normalised time of swaps; swaps clustered late push toward RUG.",
    "burn_mean_period":
        "Average normalised time of burns; burns clustered late push toward RUG (exit).",
    "lp_avg":
        "Average LP share among significant holders; depends on the rest of the distribution.",
    "lp_std":
        "Spread of LP shares across holders; high spread can indicate concentration.",
    "token_burn_ratio":
        "Share of token supply sent to burn addresses; high values push toward SAFE.",
    "number_of_token_creation_of_creator":
        "Count of ERC-20s this creator has deployed; serial deployers push toward RUG.",
}


def write_markdown(top5_shap: pd.DataFrame, gain_rank: pd.DataFrame,
                   model_label: str, n_used: int, total_test: int,
                   out_path: str) -> None:
    lines = []
    lines.append("# SHAP Feature Attribution — Top Drivers of Rug-Pull Predictions\n")
    lines.append(
        f"Model used: **{model_label}**.  "
        f"SHAP values computed on **{n_used:,}** test-set rows "
        f"(of {total_test:,} total, "
        f"{'stratified subsample' if n_used < total_test else 'full test set'}) "
        f"using `shap.TreeExplainer` on the raw boosted-tree estimator.\n"
    )
    lines.append("## Top 5 features by mean |SHAP value|\n")
    lines.append("| Rank | Feature | Mean &#124;SHAP&#124; | Mean SHAP (signed) | Direction → RUG when… |")
    lines.append("|---|---|---|---|---|")
    for i, row in top5_shap.iterrows():
        feat = row["feature"]
        interp = INTERPRETATION.get(feat, "Pushes toward RUG when elevated.")
        direction = "feature value rises" if row["mean_signed_shap"] > 0 else "feature value falls"
        lines.append(
            f"| {i+1} | `{feat}` | {row['mean_abs_shap']:.4f} "
            f"| {row['mean_signed_shap']:+.4f} | {direction} — {interp} |"
        )
    lines.append("\n## Comparison vs gain-based importance\n")

    # Show side-by-side ranking of top 10 by each metric
    shap_top = (
        gain_rank["shap_rank_val"].sort_values(ascending=False).head(10).index.tolist()
    )
    gain_top = (
        gain_rank["gain_rank_val"].sort_values(ascending=False).head(10).index.tolist()
    )

    lines.append("| Rank | By mean &#124;SHAP&#124; | By gain (XGBoost importance) |")
    lines.append("|---|---|---|")
    for i in range(10):
        a = shap_top[i] if i < len(shap_top) else ""
        b = gain_top[i] if i < len(gain_top) else ""
        lines.append(f"| {i+1} | `{a}` | `{b}` |")

    overlap_top5 = set(shap_top[:5]) & set(gain_top[:5])
    spearman = gain_rank[["shap_rank_val", "gain_rank_val"]].corr(method="spearman").iloc[0, 1]
    lines.append(
        f"\nTop-5 overlap: **{len(overlap_top5)} of 5** features appear in both rankings.  "
        f"Spearman rank correlation across all features: **{spearman:.3f}**.\n"
    )
    if len(overlap_top5) >= 4:
        lines.append(
            "The two rankings agree closely — SHAP confirms the gain-based view of which "
            "features drive predictions, while adding direction (sign) information that gain "
            "cannot express.\n"
        )
    elif len(overlap_top5) >= 2:
        lines.append(
            "The rankings broadly agree on the heavy hitters but reorder mid-ranked features.  "
            "Gain measures *split frequency × loss reduction* during training, while mean |SHAP| "
            "measures *attribution magnitude on the test set* — they will diverge whenever a "
            "feature is split on often but contributes little to individual predictions, or "
            "vice versa.\n"
        )
    else:
        lines.append(
            "The rankings disagree substantially.  Gain reflects training-time split usefulness; "
            "SHAP reflects test-time attribution per prediction.  Trust SHAP for explaining what "
            "the model is doing on real inputs — it is what the production `predict.py` now reports.\n"
        )

    lines.append("## Interpretation of top 5\n")
    for i, row in top5_shap.iterrows():
        feat = row["feature"]
        interp = INTERPRETATION.get(feat, "Pushes toward RUG when elevated.")
        lines.append(f"- **`{feat}`** — {interp}")
    lines.append("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    model, model_label = load_model_with_fallback()
    scaler = _load_pickle(os.path.join(MODELS_DIR, "scaler.pkl"))
    feature_names = _load_pickle(os.path.join(MODELS_DIR, "feature_names.pkl"))

    _, X, y, _ = load_dataset()
    _, X_test_raw, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    X_test = scaler.transform(X_test_raw)
    n_total = len(X_test)

    tree_model = get_tree_model(model)

    # Highest-probability sample is picked from the FULL test set, then we use
    # the (possibly subsampled) X for the bee-swarm summary.
    full_probs = tree_model.predict_proba(X_test)[:, 1]
    top_idx = int(np.argmax(full_probs))

    X_used, y_used, _ = maybe_subsample(X_test, y_test, SAMPLE_LIMIT)
    print(f"Computing SHAP on {len(X_used):,} rows "
          f"({'subsample' if len(X_used) < n_total else 'full test set'})…")

    explainer = shap.TreeExplainer(tree_model)
    shap_values = explainer.shap_values(X_used)

    # ── Bee-swarm summary ────────────────────────────────────────────────
    plt.figure()
    shap.summary_plot(
        shap_values, X_used, feature_names=feature_names,
        plot_type="dot", show=False,
    )
    summary_path = os.path.join(PLOTS_DIR, "shap_summary.png")
    plt.savefig(summary_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {summary_path}")

    # ── Waterfall for highest-probability sample ────────────────────────
    sv_top = explainer.shap_values(X_test[top_idx:top_idx + 1])
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = float(np.array(base).ravel()[0])

    waterfall_explanation = shap.Explanation(
        values=sv_top[0],
        base_values=base,
        data=X_test[top_idx],
        feature_names=feature_names,
    )
    plt.figure()
    shap.plots.waterfall(waterfall_explanation, show=False, max_display=12)
    waterfall_path = os.path.join(PLOTS_DIR, "shap_waterfall_example.png")
    plt.savefig(waterfall_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {waterfall_path}  (rug-prob = {full_probs[top_idx]:.4f})")

    # ── Rankings (SHAP vs gain) ──────────────────────────────────────────
    mean_abs = np.abs(shap_values).mean(axis=0)
    mean_signed = shap_values.mean(axis=0)
    shap_df = pd.DataFrame({
        "feature": feature_names,
        "mean_abs_shap": mean_abs,
        "mean_signed_shap": mean_signed,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    top5 = shap_df.head(5)
    print("\n  Top 5 features by mean |SHAP|:")
    for i, r in top5.iterrows():
        print(f"    {i+1}. {r['feature']:<40} |SHAP|={r['mean_abs_shap']:.4f}  "
              f"signed={r['mean_signed_shap']:+.4f}")

    # Compare against gain-based importance (model.feature_importances_)
    gain_imp = tree_model.feature_importances_
    rank_df = pd.DataFrame({"feature": feature_names,
                            "shap_rank_val": mean_abs,
                            "gain_rank_val": gain_imp})
    rank_df["shap_rank"] = rank_df["shap_rank_val"].rank(ascending=False)
    rank_df["gain_rank"] = rank_df["gain_rank_val"].rank(ascending=False)
    rank_df = rank_df.sort_values("shap_rank").set_index("feature")

    md_path = "shap_summary.md"
    write_markdown(top5, rank_df, model_label, len(X_used), n_total, md_path)
    print(f"\n  Markdown: {md_path}")

    # Final stdout finding
    top_share = top5.iloc[0]["mean_abs_shap"] / mean_abs.sum()
    print(
        f"\n  Top SHAP driver: {top5.iloc[0]['feature']} "
        f"(~{top_share:.1%} of total mean|SHAP| attribution)."
    )


if __name__ == "__main__":
    main()
