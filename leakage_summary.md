# Temporal Feature Ablation — Rug Pull Detector

**Question.** The deck claims this model *predicts* rug pulls before users invest. Is that true, or does the headline F1 = 0.987 come from end-of-life features (burn timing, swap-mean-period) that effectively leak the label?

**Method.** Features are partitioned into three tiers by *when* they become knowable in a token's lifecycle. We retrain XGBoost three times — EARLY only, EARLY+MID, and the full feature set — using the same pipeline as `train.py` (80/20 stratified split, MinMaxScaler, SMOTE on the training set only, `random_state=42`). Hyperparameters are fixed at `n_estimators=300, max_depth=6, learning_rate=0.1` for comparability across variants.

## Feature tiers

**EARLY** (knowable shortly after pool creation, pre-trading):

- `lp_creator_holding_ratio`
- `lp_lock_ratio`
- `lp_avg`
- `lp_std`
- `token_creator_holding_ratio`
- `token_burn_ratio`
- `number_of_token_creation_of_creator`
- `unlocked_creator_lp`
- `creator_total_exposure`
- `lp_exposure_risk`

**MID** (requires some trading history, no end-of-life signal):

- `mint_count_per_week`
- `burn_count_per_week`
- `swap_in_per_week`
- `swap_out_per_week`
- `swap_rate`
- `mint_ratio`
- `swap_ratio`
- `burn_ratio`
- `total_swap_volume`
- `sell_dominance`
- `honeypot_signal`

**LATE** (encodes end-of-life behavior — likely label leakage):

- `mint_mean_period`
- `swap_mean_period`
- `burn_mean_period`
- `swap_timing_delay`
- `burn_to_mint_ratio`

## Test-set metrics

| Variant | # feat | Normal P | Normal R | Normal F1 | Rug P | Rug R | Rug F1 | AUC | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| EARLY | 10 | 0.7248 | 0.8828 | 0.7961 | 0.9866 | 0.9626 | 0.9745 | 0.9531 | 324 | 43 | 123 | 3170 |
| EARLY+MID | 21 | 0.8489 | 0.9183 | 0.8822 | 0.9908 | 0.9818 | 0.9863 | 0.9868 | 337 | 30 | 60 | 3233 |
| FULL | 26 | 0.8496 | 0.9237 | 0.8851 | 0.9914 | 0.9818 | 0.9866 | 0.9881 | 339 | 28 | 60 | 3233 |

## Interpretation

**The model is predictive but needs trading history.** EARLY features alone are not sufficient — adding MID-tier trading activity (swap volumes, mint/burn counts, sell dominance) recovers performance without any end-of-life timing signals. This means the system can flag rugs after a token is live but before liquidity is pulled — still useful, but not at-pool-creation prediction.

Going from EARLY → FULL changes normal-class precision by +0.1248 and rug-class F1 by +0.0121. The size of these deltas is the empirical answer to the leakage question: small deltas mean the late-life signal adds little; large deltas mean the headline number depends on it.

**The cleanest leakage check is EARLY+MID vs FULL.** Adding the five LATE-tier features on top of EARLY+MID changes normal-class precision by +0.0008, rug-class F1 by +0.0003, and AUC by +0.0014. If LATE features were leaking the label, this delta would be large; it is essentially zero. The headline numbers do **not** depend on end-of-life timing signals — a rug-class F1 in the same ballpark is reachable from features available before the rug occurs.

Investors care most about the **normal-class precision** column — when the system says "safe," how often is it actually safe. That is the metric a sharp reviewer should anchor on, not the rug-class F1, because the dataset is 90% rugs and any dumb classifier hits high rug-side numbers by default.

**Reproducibility.** Run `python leakage_analysis.py`. All randomness is seeded with `random_state=42`; rerunning gives identical numbers.
