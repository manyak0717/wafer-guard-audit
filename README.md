# wafer-guard-audit

My audit of the Fab-7 wafer defect detector (SECOM data, a supervised autoencoder in PyTorch).

The original notebook claimed F1 ≥ 0.90. That number wasn't real it came from data leakage, a loss function that silently broadcast the wrong shapes, and legacy station error codes being treated as sensor readings. after being fixed properly, F1 on the newest wafers is 0.16 actually, so my call is no-go. the complete explanation is in AUDIT.md

## What's here

 `wafer_guard_fixed.ipynb`: the repaired notebook saved with its outputs
 `AUDIT.md`: how I fixed the issues in the original notebook and the real numbers. 

## To run it

You need Python 3.10+ and a few libraries:
pip install torch scikit-learn imbalanced-learn pandas numpy joblib

Download `secom_telemetry_modified.csv` from the task's Drive folder (I haven't included it here) and put it next to the notebook. On Google Colab there's nothing to install: upload the CSV when the first cell asks, then use Runtime → Run all. It takes a minute or two on CPU. The random seed is 1337, set near the top, if you want to change it.

## How to check it worked

The final cell should print an F1 of 0.1633 and a confusion matrix of TN=191, FP=36, FN=5, TP=4, matching AUDIT.md. The later decimals can shift a little between machines.

The training loop also has `assert` lines that compare the tensor shapes in both losses. If someone reintroduces the broadcasting bug, the notebook will stop and complain instead of quietly training on garbage.

The appendix at the bottom rebuilds the original buggy pipeline on purpose. It's there to reproduce the evidence I cite in the audit (the synthetic test defects, the `(64, 64, 590)` shape), not as part of the fix.

## Libraries

PyTorch for the model, scikit-learn for preprocessing and metrics, pandas and numpy for data. imbalanced-learn is only used in the appendix, to reproduce the original SMOTE leak.
