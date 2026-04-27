"""
threshold_analysis.py
---------------------
Sweep decision thresholds on the held-out test set and recommend an operating
point under an explicit cost model.

For each threshold t in [0.05, 0.95]:
  - rug-class precision / recall / F1
  - normal-class precision / recall / F1
  - false-positive rate  (legit tokens flagged as rugs)
  - false-negative rate  (rugs that slip through)
  - expected user loss   (FN = $100, FP = $5, correct = $0)

Outputs:
  plots/threshold_analysis.png  — 2-panel figure (curves + expected loss)
  threshold_summary.md          — metrics table + recommendation + caveats
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

from train import load_dataset, RANDOM_STATE


MODELS_DIR = "models"
PLOTS_DIR = "plots"

# Cost model (USD per token outcome)
COST_FN = 100.0   # missed rug — user loses their stake
COST_FP = 5.0     # legit token flagged — user misses an opportunity
COST_TP = 0.0
COST_TN = 0.0


# ── Model loading with calibrated fallback ───────────────────────────────────

def _load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_model_with_fallback() -> tuple:
    """
    Prefer the calibrated model produced on the rigor track; fall back to the
    original tuned XGBoost.
    """
    calibrated_path = os.path.join(MODELS_DIR, "xgboost_calibrated.pkl")
    best_path = os.path.join(MODELS_DIR, "xgboost_best.pkl")

    if os.path.exists(calibrated_path):
        print(f"Loaded model: {calibrated_path}")
        return _load_pickle(calibrated_path), "xgboost_calibrated.pkl"
    print(f"Loaded model: {best_path}  (calibrated model not present)")
    return _load_pickle(best_path), "xgboost_best.pkl"


# ── Threshold sweep ──────────────────────────────────────────────────────────

def sweep_thresholds(y_true: np.ndarray, y_prob: np.ndarray,
                     thresholds: np.ndarray) -> pd.DataFrame:
    rows = []
    n_total = len(y_true)
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        # rug = positive class
        rug_prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        rug_rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
        rug_f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)

        # normal = negative class
        norm_prec = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
        norm_rec = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
        norm_f1 = f1_score(y_true, y_pred, pos_label=0, zero_division=0)

        n_neg = tn + fp
        n_pos = tp + fn
        fpr = fp / n_neg if n_neg else 0.0
        fnr = fn / n_pos if n_pos else 0.0

        expected_loss = (fn * COST_FN + fp * COST_FP) / n_total

        rows.append({
            "threshold": t,
            "rug_precision": rug_prec,
            "rug_recall": rug_rec,
            "rug_f1": rug_f1,
            "normal_precision": norm_prec,
            "normal_recall": norm_rec,
            "normal_f1": norm_f1,
            "fpr": fpr,
            "fnr": fnr,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "expected_loss_per_token": expected_loss,
        })
    return pd.DataFrame(rows)


# ── Recommendations ──────────────────────────────────────────────────────────

def pick_recommendations(df: pd.DataFrame) -> dict:
    loss_row = df.loc[df["expected_loss_per_token"].idxmin()]
    f1_row = df.loc[df["rug_f1"].idxmax()]

    high_prec_mask = df["normal_precision"] >= 0.95
    if high_prec_mask.any():
        high_prec_row = df[high_prec_mask].sort_values("threshold").iloc[0]
        high_prec_met = True
    else:
        # No threshold in the swept range meets the 0.95 bar — fall back to
        # the threshold that maximises normal-class precision and flag this
        # explicitly downstream.
        high_prec_row = df.loc[df["normal_precision"].idxmax()]
        high_prec_met = False

    return {
        "loss_optimal": loss_row,
        "f1_optimal": f1_row,
        "high_precision": high_prec_row,
        "_high_prec_met": high_prec_met,
    }


# ── Plot ─────────────────────────────────────────────────────────────────────

def make_plot(df: pd.DataFrame, recs: dict, out_path: str) -> None:
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: rug F1, normal precision, FPR
    axL.plot(df["threshold"], df["rug_f1"],
             color="#3498db", lw=2, label="Rug-class F1")
    axL.plot(df["threshold"], df["normal_precision"],
             color="#2ecc71", lw=2, label="Normal-class precision")
    axL.plot(df["threshold"], df["fpr"],
             color="#e74c3c", lw=2, label="False positive rate")
    axL.set_xlabel("Decision threshold")
    axL.set_ylabel("Metric value")
    axL.set_title("Classification metrics vs threshold", fontweight="bold")
    axL.set_ylim(-0.02, 1.02)
    axL.grid(alpha=0.3)
    axL.legend(loc="center right", fontsize=10)

    # Right: expected loss with three vertical markers
    axR.plot(df["threshold"], df["expected_loss_per_token"],
             color="#8e44ad", lw=2.2, label="Expected loss / token")
    axR.set_xlabel("Decision threshold")
    axR.set_ylabel(f"Expected loss per token (USD; FN=${COST_FN:.0f}, FP=${COST_FP:.0f})")
    axR.set_title("Expected user loss vs threshold", fontweight="bold")
    axR.grid(alpha=0.3)

    colors = {"loss_optimal": "#8e44ad", "f1_optimal": "#3498db",
              "high_precision": "#2ecc71"}
    labels = {"loss_optimal": "Loss-optimal",
              "f1_optimal": "F1-optimal",
              "high_precision": "High-precision (normal≥0.95)"}
    for key in ("loss_optimal", "f1_optimal", "high_precision"):
        row = recs[key]
        axR.axvline(row["threshold"], color=colors[key], linestyle="--",
                    lw=1.5, alpha=0.85,
                    label=f"{labels[key]} t={row['threshold']:.2f}")
    axR.legend(fontsize=9, loc="upper center")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# ── Markdown summary ─────────────────────────────────────────────────────────

def make_markdown(df: pd.DataFrame, recs: dict, model_label: str,
                  baseline_t: float, n_test: int, out_path: str) -> None:
    def row_md(name: str, r: pd.Series) -> str:
        return (
            f"| {name} "
            f"| {r['threshold']:.2f} "
            f"| {r['rug_precision']:.4f} "
            f"| {r['rug_recall']:.4f} "
            f"| {r['rug_f1']:.4f} "
            f"| {r['normal_precision']:.4f} "
            f"| {r['normal_recall']:.4f} "
            f"| {r['normal_f1']:.4f} "
            f"| {r['fpr']:.4f} "
            f"| {r['fnr']:.4f} "
            f"| ${r['expected_loss_per_token']:.2f} |"
        )

    baseline_idx = (df["threshold"] - baseline_t).abs().idxmin()
    baseline_row = df.loc[baseline_idx]
    loss_opt = recs["loss_optimal"]
    delta = baseline_row["expected_loss_per_token"] - loss_opt["expected_loss_per_token"]

    lines = []
    lines.append("# Threshold Analysis — Operating Point Recommendation\n")
    lines.append(
        f"Model used: **{model_label}**.  "
        f"Test set reconstructed from `train.load_dataset()` with an 80/20 stratified split "
        f"(`random_state=42`) and the saved `MinMaxScaler`.  "
        f"Test size: **{n_test:,} tokens**.\n"
    )

    lines.append("## Cost model assumptions\n")
    lines.append(
        "We score each test token under a simple, explicit per-token cost model:\n\n"
        f"- **False negative** (real rug, model says safe): the user invests and is rugged → loss = **${COST_FN:.0f}**.\n"
        f"- **False positive** (legit token, model says rug): the user skips a real opportunity → opportunity cost = **${COST_FP:.0f}**.\n"
        "- **True positive / true negative**: $0.\n\n"
        "Expected loss per token = (FN × $100 + FP × $5) / N_test.  "
        "This is a deliberately rough proxy — real-world losses depend on stake size, win-rates "
        "on the legit side, and whether the user shorts vs. avoids flagged tokens.  "
        f"The 20:1 FN/FP cost ratio is the load-bearing assumption; results are explored under "
        "alternative ratios in the caveat section below.\n"
    )

    high_prec_met = recs.get("_high_prec_met", True)
    high_prec_label = (
        "High-precision (normal≥0.95)" if high_prec_met
        else "High-precision (best attained, target 0.95 not met)"
    )

    lines.append("## Metrics at recommended thresholds\n")
    lines.append(
        "| Recommendation | t | Rug Prec | Rug Rec | Rug F1 | Norm Prec | Norm Rec | Norm F1 | FPR | FNR | E[loss]/token |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|"
    )
    lines.append(row_md("Loss-optimal",   recs["loss_optimal"]))
    lines.append(row_md("F1-optimal",     recs["f1_optimal"]))
    lines.append(row_md(high_prec_label,  recs["high_precision"]))
    lines.append(row_md(f"Default t≈{baseline_t:.2f}", baseline_row))

    lines.append("\n## Recommendation\n")
    lines.append(
        f"Adopt the **loss-optimal threshold t = {loss_opt['threshold']:.2f}** for "
        f"production.  Under the cost model above, expected loss drops from "
        f"**${baseline_row['expected_loss_per_token']:.2f}/token** at the default "
        f"t={baseline_t:.2f} to **${loss_opt['expected_loss_per_token']:.2f}/token** "
        f"(Δ ≈ ${delta:+.2f}/token).  At this operating point the model catches "
        f"{loss_opt['rug_recall']*100:.1f}% of rugs while flagging only "
        f"{loss_opt['fpr']*100:.1f}% of legit tokens.\n"
    )
    hp = recs["high_precision"]
    if high_prec_met:
        lines.append(
            f"The **F1-optimal** threshold (t = {recs['f1_optimal']['threshold']:.2f}) "
            f"maximises rug-class F1 but ignores asymmetric costs.  The "
            f"**high-precision** threshold (t = {hp['threshold']:.2f}) is the smallest "
            f"threshold at which normal-class precision (purity of the safe-list) reaches "
            f"0.95 — appropriate for a UX where the cost of telling a user *“safe”* "
            f"about an actual rug is unacceptably high.\n"
        )
    else:
        lines.append(
            f"The **F1-optimal** threshold (t = {recs['f1_optimal']['threshold']:.2f}) "
            f"maximises rug-class F1 but ignores asymmetric costs.  The "
            f"**high-precision target was not met**: no threshold in the swept range "
            f"[0.05, 0.95] achieved normal-class precision ≥ 0.95 — the best attained value "
            f"is **{hp['normal_precision']:.4f}** at t = {hp['threshold']:.2f}.  "
            f"Reaching 0.95 would require either a more aggressive threshold below 0.05 "
            f"(driving normal-class precision up by shrinking the safe-list further) or a "
            f"better-calibrated model.  Treat the row above as the closest available "
            f"approximation, not a strict guarantee.\n"
        )

    lines.append("## Caveats — sensitivity to cost assumptions\n")
    lines.append(
        "The recommendation moves materially under different cost ratios:\n\n"
        "- If **FN cost falls** (e.g. user only deposits small amounts, or mostly shorts), "
        "the optimal threshold *rises* — it becomes worth tolerating a few missed rugs to "
        "avoid annoying false alarms.\n"
        "- If **FN cost rises** (concentrated stakes, no exit options), the optimal "
        "threshold *falls* — flag aggressively, accept more FPs.\n"
        "- If **FP cost rises** (premium recommendation product where false alarms erode "
        "trust), the high-precision operating point is the right anchor instead.\n\n"
        "The 90/10 class imbalance amplifies this: with ~90% rugs in the test set, even a "
        "small FNR translates into many missed rugs in absolute terms, which is why the "
        "loss-optimal threshold tends to sit below the F1-optimal one whenever FN/FP > ~5×.\n"
    )

    with open(out_path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    model, model_label = load_model_with_fallback()
    scaler = _load_pickle(os.path.join(MODELS_DIR, "scaler.pkl"))

    _, X, y, _ = load_dataset()
    _, X_test_raw, _, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    X_test = scaler.transform(X_test_raw)

    y_prob = model.predict_proba(X_test)[:, 1]

    thresholds = np.linspace(0.05, 0.95, 19)
    df = sweep_thresholds(y_test, y_prob, thresholds)

    recs = pick_recommendations(df)

    plot_path = os.path.join(PLOTS_DIR, "threshold_analysis.png")
    md_path = "threshold_summary.md"
    make_plot(df, recs, plot_path)
    make_markdown(df, recs, model_label,
                  baseline_t=0.50, n_test=len(y_test), out_path=md_path)

    print("\n" + "=" * 70)
    print("  Threshold analysis — recommendations on test set "
          f"(N={len(y_test):,})")
    print("=" * 70)
    for name in ("loss_optimal", "f1_optimal", "high_precision"):
        row = recs[name]
        print(f"\n  {name}:")
        print(f"    threshold        = {row['threshold']:.2f}")
        print(f"    rug F1           = {row['rug_f1']:.4f}")
        print(f"    rug precision    = {row['rug_precision']:.4f}")
        print(f"    rug recall       = {row['rug_recall']:.4f}")
        print(f"    normal precision = {row['normal_precision']:.4f}")
        print(f"    FPR              = {row['fpr']:.4f}")
        print(f"    FNR              = {row['fnr']:.4f}")
        print(f"    E[loss]/token    = ${row['expected_loss_per_token']:.2f}")

    print(f"\n  Plot:    {plot_path}")
    print(f"  Summary: {md_path}")


if __name__ == "__main__":
    main()
