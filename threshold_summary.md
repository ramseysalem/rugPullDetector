# Threshold Analysis — Operating Point Recommendation

Model used: **xgboost_best.pkl**.  Test set reconstructed from `train.load_dataset()` with an 80/20 stratified split (`random_state=42`) and the saved `MinMaxScaler`.  Test size: **3,660 tokens**.

## Cost model assumptions

We score each test token under a simple, explicit per-token cost model:

- **False negative** (real rug, model says safe): the user invests and is rugged → loss = **$100**.
- **False positive** (legit token, model says rug): the user skips a real opportunity → opportunity cost = **$5**.
- **True positive / true negative**: $0.

Expected loss per token = (FN × $100 + FP × $5) / N_test.  This is a deliberately rough proxy — real-world losses depend on stake size, win-rates on the legit side, and whether the user shorts vs. avoids flagged tokens.  The 20:1 FN/FP cost ratio is the load-bearing assumption; results are explored under alternative ratios in the caveat section below.

## Metrics at recommended thresholds

| Recommendation | t | Rug Prec | Rug Rec | Rug F1 | Norm Prec | Norm Rec | Norm F1 | FPR | FNR | E[loss]/token |
|---|---|---|---|---|---|---|---|---|---|---|
| Loss-optimal | 0.05 | 0.9840 | 0.9924 | 0.9882 | 0.9263 | 0.8556 | 0.8895 | 0.1444 | 0.0076 | $0.76 |
| F1-optimal | 0.20 | 0.9885 | 0.9900 | 0.9892 | 0.9088 | 0.8965 | 0.9026 | 0.1035 | 0.0100 | $0.95 |
| High-precision (best attained, target 0.95 not met) | 0.05 | 0.9840 | 0.9924 | 0.9882 | 0.9263 | 0.8556 | 0.8895 | 0.1444 | 0.0076 | $0.76 |
| Default t≈0.50 | 0.50 | 0.9908 | 0.9827 | 0.9867 | 0.8553 | 0.9183 | 0.8857 | 0.0817 | 0.0173 | $1.60 |

## Recommendation

Adopt the **loss-optimal threshold t = 0.05** for production.  Under the cost model above, expected loss drops from **$1.60/token** at the default t=0.50 to **$0.76/token** (Δ ≈ $+0.84/token).  At this operating point the model catches 99.2% of rugs while flagging only 14.4% of legit tokens.

The **F1-optimal** threshold (t = 0.20) maximises rug-class F1 but ignores asymmetric costs.  The **high-precision target was not met**: no threshold in the swept range [0.05, 0.95] achieved normal-class precision ≥ 0.95 — the best attained value is **0.9263** at t = 0.05.  Reaching 0.95 would require either a more aggressive threshold below 0.05 (driving normal-class precision up by shrinking the safe-list further) or a better-calibrated model.  Treat the row above as the closest available approximation, not a strict guarantee.

## Caveats — sensitivity to cost assumptions

The recommendation moves materially under different cost ratios:

- If **FN cost falls** (e.g. user only deposits small amounts, or mostly shorts), the optimal threshold *rises* — it becomes worth tolerating a few missed rugs to avoid annoying false alarms.
- If **FN cost rises** (concentrated stakes, no exit options), the optimal threshold *falls* — flag aggressively, accept more FPs.
- If **FP cost rises** (premium recommendation product where false alarms erode trust), the high-precision operating point is the right anchor instead.

The 90/10 class imbalance amplifies this: with ~90% rugs in the test set, even a small FNR translates into many missed rugs in absolute terms, which is why the loss-optimal threshold tends to sit below the F1-optimal one whenever FN/FP > ~5×.

