# Probability Calibration — Rug Pull XGBoost

**Question.** When the saved XGBoost model says "this token has a 70% rug-pull probability," is that actually a 70% rate? Tree-based ensembles are usually pulled toward 0 and 1 by their averaging behavior, so the raw `predict_proba` output is often miscalibrated. We need this to be reliable because the threshold/cost work on Track B treats those probabilities as decision inputs.

**Method.** Reuse the same train/test split as `train.py` (80/20 stratified, `random_state=42`). Score the saved `xgboost_best.pkl` on the held-out test set to get *uncalibrated* probabilities. Then re-split the original training set 80/20 into a fit-set and a calibration-set, retrain a fresh XGBoost on the SMOTE-balanced fit-set with identical hyperparameters, and wrap with `CalibratedClassifierCV(cv='prefit', method='isotonic')` fit on the untouched calibration-set. Score this wrapper on the same test set to get *calibrated* probabilities.

## Metrics on the test set

| Model | Brier score | ECE (10 bins) | Mean predicted P(rug) | Mean actual P(rug) |
|---|---:|---:|---:|---:|
| Uncalibrated `xgboost_best.pkl` | 0.0197 | 0.0189 | 0.8913 | 0.8997 |
| Calibrated (isotonic, prefit)  | 0.0178 | 0.0051 | 0.9011 | 0.8997 |

Lower is better for both Brier and ECE. Brier ∈ [0, 1] is mean squared error of the probability; ECE is the weighted gap between average predicted probability and empirical rate inside each confidence bin.

## Direction of miscalibration

On the test set, the saved XGBoost is **underconfident (predicting lower rug rates than actually occur)**. The reliability diagram (`plots/calibration_curve.png`) shows the per-bin pattern: bins where the orange (uncalibrated) curve sits *above* the diagonal are over-predictions of rug risk; bins where it sits *below* the diagonal are under-predictions.

## Recommendation

**Recommendation: use `models/xgboost_calibrated.pkl` for any production scoring or threshold/cost analysis that depends on the numeric probability value.** Isotonic calibration reduces ECE by +0.0138 and Brier score by +0.0019. For binary yes/no classification at the default 0.5 cutoff the two are nearly identical, so existing accuracy/F1 reporting can keep using `xgboost_best.pkl`. The probability column on `predict.py` output and any expected-cost tradeoff should use the calibrated model.

**Note for Track B (threshold/cost analysis).** Threshold sweeps and expected-cost framings should be run against the calibrated probabilities — otherwise the cost of "flag at p ≥ 0.7" is being computed against a number that does not actually correspond to a 70% rate.

## Reproducibility

Run `python calibration_analysis.py`. The script does not modify `xgboost_best.pkl` or any other existing artefact; it produces only the new files listed at the top of this document.
