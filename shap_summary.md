# SHAP Feature Attribution — Top Drivers of Rug-Pull Predictions

Model used: **xgboost_best.pkl**.  SHAP values computed on **1,000** test-set rows (of 3,660 total, stratified subsample) using `shap.TreeExplainer` on the raw boosted-tree estimator.

## Top 5 features by mean |SHAP value|

| Rank | Feature | Mean &#124;SHAP&#124; | Mean SHAP (signed) | Direction → RUG when… |
|---|---|---|---|---|
| 1 | `mint_count_per_week` | 2.7274 | +2.0671 | feature value rises — LP add events per week; abnormal patterns (very low or very spiky) push toward RUG. |
| 2 | `lp_creator_holding_ratio` | 1.4523 | +0.9627 | feature value rises — Share of LP tokens held by the creator; concentrated creator LP pushes toward RUG. |
| 3 | `sell_dominance` | 1.1136 | +0.9209 | feature value rises — Sell-side share of swaps; near 1.0 means everyone is exiting and pushes toward RUG. |
| 4 | `creator_total_exposure` | 0.9221 | +0.6341 | feature value rises — Combined token + LP creator stake; high single-actor exposure pushes toward RUG. |
| 5 | `mint_mean_period` | 0.6425 | +0.4183 | feature value rises — Average normalised time of mints; mints clustered late push toward RUG. |

## Comparison vs gain-based importance

| Rank | By mean &#124;SHAP&#124; | By gain (XGBoost importance) |
|---|---|---|
| 1 | `mint_count_per_week` | `unlocked_creator_lp` |
| 2 | `lp_creator_holding_ratio` | `lp_creator_holding_ratio` |
| 3 | `sell_dominance` | `creator_total_exposure` |
| 4 | `creator_total_exposure` | `mint_count_per_week` |
| 5 | `mint_mean_period` | `lp_lock_ratio` |
| 6 | `unlocked_creator_lp` | `lp_avg` |
| 7 | `token_burn_ratio` | `sell_dominance` |
| 8 | `lp_lock_ratio` | `number_of_token_creation_of_creator` |
| 9 | `swap_out_per_week` | `mint_mean_period` |
| 10 | `token_creator_holding_ratio` | `honeypot_signal` |

Top-5 overlap: **3 of 5** features appear in both rankings.  Spearman rank correlation across all features: **0.570**.

The rankings broadly agree on the heavy hitters but reorder mid-ranked features.  Gain measures *split frequency × loss reduction* during training, while mean |SHAP| measures *attribution magnitude on the test set* — they will diverge whenever a feature is split on often but contributes little to individual predictions, or vice versa.

## Interpretation of top 5

- **`mint_count_per_week`** — LP add events per week; abnormal patterns (very low or very spiky) push toward RUG.
- **`lp_creator_holding_ratio`** — Share of LP tokens held by the creator; concentrated creator LP pushes toward RUG.
- **`sell_dominance`** — Sell-side share of swaps; near 1.0 means everyone is exiting and pushes toward RUG.
- **`creator_total_exposure`** — Combined token + LP creator stake; high single-actor exposure pushes toward RUG.
- **`mint_mean_period`** — Average normalised time of mints; mints clustered late push toward RUG.

