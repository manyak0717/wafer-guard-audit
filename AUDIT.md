# WAFER-GUARD audit

**Verdict: no-go.** The notebook's 0.99 F1 isn't real. Once I fixed the leaks and the broken loss, F1 on the newest wafers is **0.16**, nowhere near the 0.90 gate. Every number below comes from `wafer_guard_fixed.ipynb` (seed 1337).

## The three silent bugs

**1. Data leakage.** This was really three leaks. SMOTE ran on the whole dataset *before* the split, so the test set was full of fake wafers blended from training ones. The split was random and ignored timestamps, so the model trained on October and got tested on July. And the imputer and scaler were fitted on every row, test wafers included.

How bad was it? The old test set was 50% defective (real life: 6.6%), and 274 of its 293 "defects" were synthetic. Only 19 were real. That's how the original got ROC-AUC 1.000 and missed nothing.

The fix: sort by time, split 70/15/15, fit all preprocessing on the training window only, and drop SMOTE in favour of `pos_weight` (13.1) in the loss.

**2. Silent broadcasting in the loss.** The inputs are `[64, 1, 590]` and the reconstructions come out `[64, 590]`. Subtract one from the other and PyTorch quietly builds `[64, 64, 590]`, comparing every rebuilt wafer to every wafer in the batch. There's no error, just useless gradients. The decoder's best move is to output the average wafer, and that's what it did. Reconstruction loss sat at 0.677 for all 40 epochs, which matches the 0.6775 you get by always predicting the mean.

The fix: compare against `batch_x.flatten(1)` and assert both shapes. Reconstruction loss now falls to 0.56, below the 0.99 baseline.

**3. The legacy station.** 146 wafers go through Probe Station PS-L. On those, `sensor_0` is a 0.0 flag and `sensor_1` holds error codes (100, 200 or 300), not measurements. The original scaled them like real readings. A code of 100 sits about 30σ below the normal `sensor_1` level of ~2,500, which skews that channel for every wafer. None of the 146 are defective either (0% vs 7.3% for the rest).

The fix: an `is_legacy` flag, one-hot `code_100/200/300` columns (not scaled), and `sensor_0`/`sensor_1` set to missing for those rows, kept out of the scaler, then masked to 0. I didn't invent readings for them.

## The real numbers

The test window is 5-17 Oct: 236 wafers with 9 defects (3.8%). The threshold (0.0097) was picked on validation and applied once.

| | Original | Repaired |
|---|---|---|
| Test prevalence | 50.0% | 3.8% |
| F1 | 0.9915 | **0.1633** |
| Precision / Recall | 0.983 / 1.000 | 0.100 / 0.444 |
| ROC-AUC | 1.000 | 0.7044 |
| PR-AUC (AP) | 1.000 | 0.1017 (random guess: 0.0381) |
| TN / FP / FN / TP | 288 / 5 / 0 / 293 | 191 / 36 / 5 / 4 |

I lean on PR-AUC because with defects under 7% of wafers, ROC-AUC flatters the model. The model does beat random guessing (0.10 vs 0.04), but it catches 4 of 9 defects at the cost of 36 false alarms. That isn't deployable.

## Smaller things I fixed

- The decision threshold was a fixed 0.5 with no validation set. It's now chosen on validation (val F1 0.1882).
- There was no early stopping. Training loss hit 0.0009 while validation AP sat near 0.064 (validation prevalence is 7.2%), so the model was memorising. Epoch 3 was best (val AP 0.0978), so that's the one kept.
- 122 constant columns are dropped.
- The release cell saved the model even when the gate failed. It now saves only on a pass, and this time it correctly wrote nothing.

## What still worries me

Nine test defects is very few. One wafer moves recall by 0.11, so treat 0.16 as a rough estimate. The defect rate also drifts (7.1% in train, 7.2% in validation, 3.8% in test), so a model trained on July-September may already be stale. And since all 146 legacy wafers passed, the model can learn "legacy means pass", which is only safe if PS-L really is lower-risk. Someone at the fab should confirm that. Finally, 78 training defects against 472 features and 317,465 parameters, on one seed, invites overfitting.

## Recommendation

No-go for Friday. Before anyone tries again: a simple logistic regression or tree baseline to beat, retraining on a rolling window, more labelled defects, and a longer test period.
