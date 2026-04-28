"""
calibration_analysis.py
-----------------------
Probability calibration for the rug pull XGBoost classifier.

Tree-based models tend to be miscalibrated: a "70% rug pull probability" output
does not necessarily correspond to a 70% real rate. This matters for the
Track-B threshold-cost analysis, which assumes the probabilities are
trustworthy.

Procedure:
  1. Reconstruct the same train/test split as `train.py` (`random_state=42`).
  2. Score the saved `xgboost_best.pkl` on the test set → uncalibrated probs.
  3. Re-split the *training* set into fit (80%) + calibration (20%).
     Retrain a fresh XGBoost on the fit-set with the same hyperparameters as
     the saved model. Wrap with `CalibratedClassifierCV(cv='prefit',
     method='isotonic')` fit on the calibration set.
  4. Score the wrapped model on the same test set → calibrated probs.
  5. Report Brier score and Expected Calibration Error (ECE, 10 equal-width
     bins) for both, plot a reliability diagram, and save the calibrated
     model.

Outputs:
  plots/calibration_curve.png
  models/xgboost_calibrated.pkl
  calibration_summary.md
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

try:
    # sklearn ≥ 1.6 — replaces the deprecated `cv='prefit'` flag
    from sklearn.frozen import FrozenEstimator
    _HAS_FROZEN = True
except ImportError:
    FrozenEstimator = None
    _HAS_FROZEN = False

from train import load_dataset

RANDOM_STATE = 42
MODELS_DIR = "models"
PLOTS_DIR = "plots"
PLOT_PATH = os.path.join(PLOTS_DIR, "calibration_curve.png")
CALIBRATED_MODEL_PATH = os.path.join(MODELS_DIR, "xgboost_calibrated.pkl")
SUMMARY_PATH = "calibration_summary.md"


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """ECE with equal-width bins on [0, 1]."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    # right=False so that 1.0 falls into the last bin
    bin_idx = np.clip(np.digitize(y_prob, bins[1:-1], right=False), 0, n_bins - 1)

    ece = 0.0
    n = len(y_true)
    for b in range(n_bins):
        mask = bin_idx == b
        if not mask.any():
            continue
        bin_acc = y_true[mask].mean()
        bin_conf = y_prob[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def _load_artifacts():
    with open(os.path.join(MODELS_DIR, "xgboost_best.pkl"), "rb") as f:
        xgb_model = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
        scaler = pickle.load(f)
    with open(os.path.join(MODELS_DIR, "feature_names.pkl"), "rb") as f:
        feature_names = pickle.load(f)
    return xgb_model, scaler, feature_names


def _xgb_from_saved(saved_xgb):
    """Build a fresh XGBClassifier mirroring the saved model's hyperparameters."""
    p = saved_xgb.get_params()
    keep = {
        "n_estimators",
        "max_depth",
        "learning_rate",
        "subsample",
        "colsample_bytree",
        "min_child_weight",
        "gamma",
        "reg_alpha",
        "reg_lambda",
        "scale_pos_weight",
        "booster",
        "objective",
        "eval_metric",
    }
    kwargs = {k: v for k, v in p.items() if k in keep and v is not None}
    kwargs.setdefault("eval_metric", "logloss")
    kwargs["random_state"] = RANDOM_STATE
    kwargs["n_jobs"] = -1
    return XGBClassifier(**kwargs), kwargs


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)

    xgb_model, scaler, feature_names = _load_artifacts()
    print("Loaded saved artefacts.")
    print(f"  feature_names: {len(feature_names)} features")

    # 1. Reconstruct the same split as train.py
    df_feat, X, y, names = load_dataset()
    assert names == feature_names, "feature ordering drift between train.py and saved pickle"

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    X_test_scaled = scaler.transform(X_test_raw)
    X_train_scaled = scaler.transform(X_train_raw)

    # 2. Uncalibrated probabilities from the saved model
    p_uncal = xgb_model.predict_proba(X_test_scaled)[:, 1]
    brier_uncal = brier_score_loss(y_test, p_uncal)
    ece_uncal = expected_calibration_error(y_test, p_uncal, n_bins=10)
    print(f"\nUncalibrated XGBoost (saved model):")
    print(f"  Brier score : {brier_uncal:.4f}")
    print(f"  ECE (10 bin): {ece_uncal:.4f}")

    # 3. Build a calibrated wrapper.
    # Re-split the *original, unbalanced* training set into fit (80%) and
    # calibration (20%). Retrain a fresh XGBoost on the fit-set with the same
    # hyperparameters and SMOTE-balance the fit-set the same way train.py does.
    fresh_xgb, used_kwargs = _xgb_from_saved(xgb_model)
    print("\nBuilding calibrated wrapper.")
    print(f"  Hyperparameters in use: "
          + ", ".join(f"{k}={v}" for k, v in used_kwargs.items()))

    X_fit_raw, X_calib_raw, y_fit, y_calib = train_test_split(
        X_train_raw, y_train,
        test_size=0.2, stratify=y_train, random_state=RANDOM_STATE,
    )
    X_fit_scaled = scaler.transform(X_fit_raw)
    X_calib_scaled = scaler.transform(X_calib_raw)

    smote = SMOTE(random_state=RANDOM_STATE)
    X_fit_bal, y_fit_bal = smote.fit_resample(X_fit_scaled, y_fit)
    print(f"  Fit-set after SMOTE  : {len(X_fit_bal):,} samples")
    print(f"  Calibration set      : {len(X_calib_scaled):,} samples (untouched)")

    fresh_xgb.fit(X_fit_bal, y_fit_bal)

    if _HAS_FROZEN:
        # sklearn ≥ 1.6: cv='prefit' was removed; wrap with FrozenEstimator.
        calibrated = CalibratedClassifierCV(
            estimator=FrozenEstimator(fresh_xgb), method="isotonic",
        )
    else:
        calibrated = CalibratedClassifierCV(
            estimator=fresh_xgb, cv="prefit", method="isotonic",
        )
    calibrated.fit(X_calib_scaled, y_calib)

    # 4. Calibrated probs on the same test set
    p_cal = calibrated.predict_proba(X_test_scaled)[:, 1]
    brier_cal = brier_score_loss(y_test, p_cal)
    ece_cal = expected_calibration_error(y_test, p_cal, n_bins=10)
    print(f"\nCalibrated XGBoost (isotonic, prefit):")
    print(f"  Brier score : {brier_cal:.4f}")
    print(f"  ECE (10 bin): {ece_cal:.4f}")

    # 5. Save calibrated model
    with open(CALIBRATED_MODEL_PATH, "wb") as f:
        pickle.dump(calibrated, f)
    print(f"\nCalibrated model saved: {CALIBRATED_MODEL_PATH}")

    # 6. Reliability diagram
    frac_pos_uncal, mean_pred_uncal = calibration_curve(
        y_test, p_uncal, n_bins=10, strategy="uniform"
    )
    frac_pos_cal, mean_pred_cal = calibration_curve(
        y_test, p_cal, n_bins=10, strategy="uniform"
    )

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.0, label="Perfectly calibrated")
    ax.plot(
        mean_pred_uncal, frac_pos_uncal, "o-",
        color="#DD8452", linewidth=2,
        label=f"Uncalibrated XGBoost  (Brier={brier_uncal:.4f}, ECE={ece_uncal:.4f})",
    )
    ax.plot(
        mean_pred_cal, frac_pos_cal, "s-",
        color="#4C72B0", linewidth=2,
        label=f"Calibrated  (isotonic) (Brier={brier_cal:.4f}, ECE={ece_cal:.4f})",
    )
    ax.set_xlabel("Mean predicted rug-pull probability (per bin)")
    ax.set_ylabel("Empirical fraction of rug pulls (per bin)")
    ax.set_title("Reliability Diagram — XGBoost Rug Pull Probabilities")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.legend(loc="lower right")
    ax.grid(linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved: {PLOT_PATH}")

    # 7. Markdown summary
    # Direction: compare mean predicted vs mean actual on the test set.
    mean_pred_full = float(np.mean(p_uncal))
    mean_actual = float(np.mean(y_test))
    if mean_pred_full > mean_actual + 0.005:
        direction = "overconfident (predicting higher rug rates than actually occur)"
    elif mean_pred_full < mean_actual - 0.005:
        direction = "underconfident (predicting lower rug rates than actually occur)"
    else:
        direction = "approximately neutral on average — but bin-level miscalibration can still exist"

    brier_delta = brier_uncal - brier_cal
    ece_delta = ece_uncal - ece_cal
    if ece_delta > 0:
        recommendation = (
            f"**Recommendation: use `models/xgboost_calibrated.pkl` for any "
            f"production scoring or threshold/cost analysis that depends on "
            f"the numeric probability value.** Isotonic calibration reduces "
            f"ECE by {ece_delta:+.4f} and Brier score by {brier_delta:+.4f}. "
            f"For binary yes/no classification at the default 0.5 cutoff the "
            f"two are nearly identical, so existing accuracy/F1 reporting can "
            f"keep using `xgboost_best.pkl`. The probability column on `predict.py` "
            f"output and any expected-cost tradeoff should use the calibrated "
            f"model."
        )
    else:
        recommendation = (
            f"**Recommendation: keep using `models/xgboost_best.pkl`.** The "
            f"saved XGBoost is already well-calibrated — isotonic calibration "
            f"changed ECE by {ece_delta:+.4f} and Brier by {brier_delta:+.4f}, "
            f"i.e. essentially no improvement. The calibrated model is saved "
            f"for completeness but offers no production benefit."
        )

    md = []
    md.append("# Probability Calibration — Rug Pull XGBoost\n")
    md.append(
        "**Question.** When the saved XGBoost model says \"this token has a "
        "70% rug-pull probability,\" is that actually a 70% rate? Tree-based "
        "ensembles are usually pulled toward 0 and 1 by their averaging "
        "behavior, so the raw `predict_proba` output is often miscalibrated. "
        "We need this to be reliable because the threshold/cost work on "
        "Track B treats those probabilities as decision inputs.\n"
    )
    md.append(
        "**Method.** Reuse the same train/test split as `train.py` "
        "(80/20 stratified, `random_state=42`). Score the saved "
        "`xgboost_best.pkl` on the held-out test set to get *uncalibrated* "
        "probabilities. Then re-split the original training set 80/20 into a "
        "fit-set and a calibration-set, retrain a fresh XGBoost on the "
        "SMOTE-balanced fit-set with identical hyperparameters, and wrap with "
        "`CalibratedClassifierCV(cv='prefit', method='isotonic')` fit on the "
        "untouched calibration-set. Score this wrapper on the same test set "
        "to get *calibrated* probabilities.\n"
    )
    md.append("## Metrics on the test set\n")
    md.append("| Model | Brier score | ECE (10 bins) | Mean predicted P(rug) | Mean actual P(rug) |")
    md.append("|---|---:|---:|---:|---:|")
    md.append(
        f"| Uncalibrated `xgboost_best.pkl` | {brier_uncal:.4f} | {ece_uncal:.4f} | "
        f"{mean_pred_full:.4f} | {mean_actual:.4f} |"
    )
    md.append(
        f"| Calibrated (isotonic, prefit)  | {brier_cal:.4f} | {ece_cal:.4f} | "
        f"{float(np.mean(p_cal)):.4f} | {mean_actual:.4f} |"
    )
    md.append(
        "\nLower is better for both Brier and ECE. Brier ∈ [0, 1] is mean "
        "squared error of the probability; ECE is the weighted gap between "
        "average predicted probability and empirical rate inside each "
        "confidence bin.\n"
    )
    md.append("## Direction of miscalibration\n")
    md.append(
        f"On the test set, the saved XGBoost is **{direction}**. The "
        f"reliability diagram (`plots/calibration_curve.png`) shows the "
        f"per-bin pattern: bins where the orange (uncalibrated) curve sits "
        f"*above* the diagonal are over-predictions of rug risk; bins where "
        f"it sits *below* the diagonal are under-predictions.\n"
    )
    md.append("## Recommendation\n")
    md.append(recommendation)
    md.append("")
    md.append(
        "**Note for Track B (threshold/cost analysis).** Threshold sweeps and "
        "expected-cost framings should be run against the calibrated "
        "probabilities — otherwise the cost of \"flag at p ≥ 0.7\" is being "
        "computed against a number that does not actually correspond to a "
        "70% rate.\n"
    )
    md.append("## Reproducibility\n")
    md.append(
        "Run `python calibration_analysis.py`. The script does not modify "
        "`xgboost_best.pkl` or any other existing artefact; it produces only "
        "the new files listed at the top of this document.\n"
    )

    with open(SUMMARY_PATH, "w") as f:
        f.write("\n".join(md))
    print(f"Summary saved: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
