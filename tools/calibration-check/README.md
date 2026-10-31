# calibcheck — measure and fix classifier calibration

A small, dependency-light Python library + CLI that answers one question:
**when my classifier says "90 % sure", is it right 90 % of the time?**

It works on predicted probabilities only (a CSV or NumPy arrays), so it fits
any model: scikit-learn, XGBoost/LightGBM, PyTorch, or a Kaggle submission
pipeline. It reports calibration metrics, draws a reliability diagram, and
can fit a post-hoc calibrator on a held-out split and show before/after.

## Why it is useful

Accuracy and AUC do not tell you whether probabilities can be trusted. Badly
calibrated probabilities break anything that uses them as numbers: decision
thresholds, expected-cost calculations, ensembling/stacking, log-loss
leaderboards, and "abstain when unsure" logic. `calibcheck` makes that visible
in one command and fixes it without retraining the model.

## What is included

| Area | Functions / classes |
|---|---|
| Metrics | `expected_calibration_error` (uniform or equal-mass bins), `maximum_calibration_error`, `brier_score` (binary and multi-class), `log_loss`, `accuracy`, `brier_decomposition` (Murphy: reliability − resolution + uncertainty), `reliability_bins`, `calibration_report` |
| Calibrators | `TemperatureScaling` (binary + multi-class, keeps argmax), `PlattScaling` (binary), `IsotonicCalibrator` (binary, one-vs-rest for multi-class) |
| Plot | `calibcheck.plot.reliability_diagram` (curves + log-scale bin counts) |
| CLI | `python -m calibcheck preds.csv ...` → `report.json`, `bins.csv`, `reliability.png` |

## Install

```bash
cd tools/calibration-check
pip install -r requirements.txt
```

## Use

### Python

```python
from calibcheck import calibration_report, TemperatureScaling

print(calibration_report(y_test, model.predict_proba(X_test)).to_dict())

ts = TemperatureScaling().fit(y_cal, model.predict_proba(X_cal))   # held-out split
p_test = ts.transform(model.predict_proba(X_test))
print("T =", ts.temperature_)
```

Binary problems can pass a 1-D array of p(class 1) or an `(n, 2)` array.
Labels must be integers `0..K-1`.

### CLI

```bash
# just measure
python -m calibcheck preds.csv --label-col label --prob-cols p0,p1,p2 --out-dir report

# measure + fit temperature scaling on 50 % of the rows, evaluate on the other 50 %
python -m calibcheck examples/results/demo_predictions_digits_gnb.csv \
    --calibrate temperature --out-dir examples/results/cli_demo
```

Output of the second command (real run, included in `examples/results/cli_demo/`):

```
rows=359 classes=10 accuracy=0.8217
ECE=0.1692  MCE=0.5222  Brier=0.3402  NLL=3.5384
after temperature: ECE=0.0174  MCE=0.1628  Brier=0.2928  NLL=0.7374
```

Gaussian Naive Bayes on the digits data is ~99 % "confident" on average but only
82 % accurate; the fitted temperature (T ≈ 6.7) removes most of that gap without
changing a single prediction.

## Benchmark on real data

`python examples/benchmark.py` (≈ 12 s on CPU) trains four common
scikit-learn models on two real datasets, splits each 60/20/20 into
train / calibration / test (stratified), fits each calibrator on the
calibration split and measures **on the test split only**. Means over 5
seeds (± std for ECE), 10 uniform bins:

| dataset | model | calibrator | accuracy | ECE | Brier | NLL |
|---|---|---|---|---|---|---|
| breast_cancer | GaussianNB | uncalibrated | 0.942 | 0.059 ± 0.018 | 0.054 | 0.670 |
| breast_cancer | GaussianNB | temperature | 0.942 | 0.034 ± 0.020 | 0.045 | 0.166 |
| breast_cancer | GaussianNB | platt | 0.933 | 0.054 ± 0.022 | 0.053 | 0.232 |
| breast_cancer | GaussianNB | isotonic | 0.926 | 0.052 ± 0.033 | 0.057 | 0.656 |
| breast_cancer | LogisticRegression | uncalibrated | 0.979 | 0.018 ± 0.002 | 0.016 | 0.056 |
| breast_cancer | LogisticRegression | temperature | 0.979 | 0.025 ± 0.010 | 0.017 | 0.063 |
| breast_cancer | LogisticRegression | platt | 0.977 | 0.017 ± 0.008 | 0.018 | 0.066 |
| breast_cancer | LogisticRegression | isotonic | 0.977 | 0.022 ± 0.011 | 0.019 | 0.192 |
| breast_cancer | RandomForest | uncalibrated | 0.967 | 0.041 ± 0.017 | 0.029 | 0.192 |
| breast_cancer | RandomForest | temperature | 0.967 | 0.056 ± 0.034 | 0.032 | 0.201 |
| breast_cancer | RandomForest | platt | 0.965 | 0.042 ± 0.027 | 0.033 | 0.194 |
| breast_cancer | RandomForest | isotonic | 0.958 | 0.035 ± 0.018 | 0.033 | 0.414 |
| breast_cancer | MLP | uncalibrated | 0.974 | 0.019 ± 0.006 | 0.018 | 0.065 |
| breast_cancer | MLP | temperature | 0.974 | 0.024 ± 0.014 | 0.018 | 0.067 |
| breast_cancer | MLP | platt | 0.970 | 0.022 ± 0.004 | 0.021 | 0.073 |
| breast_cancer | MLP | isotonic | 0.968 | 0.021 ± 0.013 | 0.023 | 0.203 |
| digits | GaussianNB | uncalibrated | 0.842 | 0.147 ± 0.023 | 0.299 | 2.593 |
| digits | GaussianNB | temperature | 0.842 | 0.037 ± 0.015 | 0.250 | 0.596 |
| digits | GaussianNB | isotonic | 0.885 | 0.036 ± 0.012 | 0.193 | 0.765 |
| digits | LogisticRegression | uncalibrated | 0.964 | 0.020 ± 0.003 | 0.055 | 0.114 |
| digits | LogisticRegression | temperature | 0.964 | 0.018 ± 0.003 | 0.054 | 0.111 |
| digits | LogisticRegression | isotonic | 0.963 | 0.023 ± 0.005 | 0.060 | 0.497 |
| digits | RandomForest | uncalibrated | 0.976 | 0.223 ± 0.007 | 0.122 | 0.332 |
| digits | RandomForest | temperature | 0.976 | 0.015 ± 0.005 | 0.036 | 0.073 |
| digits | RandomForest | isotonic | 0.978 | 0.015 ± 0.004 | 0.040 | 0.287 |
| digits | MLP | uncalibrated | 0.971 | 0.018 ± 0.005 | 0.041 | 0.100 |
| digits | MLP | temperature | 0.971 | 0.019 ± 0.006 | 0.041 | 0.098 |
| digits | MLP | isotonic | 0.970 | 0.020 ± 0.006 | 0.048 | 0.396 |

Files: `examples/results/benchmark_summary.csv` (mean/std), `benchmark_raw.csv`
(every seed), and reliability diagrams such as
`examples/results/reliability_digits_RandomForest.png`.

### What the numbers say

* **Random forest on digits is strongly under-confident** (ECE 0.223):
  averaging 200 trees pulls probabilities towards the middle. Temperature
  scaling cuts ECE to 0.015 and log-loss from 0.332 to 0.073 with identical
  accuracy.
* **Gaussian Naive Bayes is badly over-confident** on both datasets
  (digits ECE 0.147, NLL 2.59). Temperature scaling brings NLL down to 0.60.
* **Logistic regression and the MLP are already well calibrated**; calibrators
  change little and can make things slightly worse. Calibrate only when the
  diagnostics say so.
* **Isotonic regression often hurts log-loss** here: with only 114–360
  calibration rows it outputs exact 0/1 probabilities for some regions, which
  log-loss punishes heavily (we clip at 1e-12). Prefer temperature/Platt on
  small calibration sets.
* On the tiny breast-cancer test split (114 rows) differences between
  calibrators are within one standard deviation across seeds — treat them as
  noise, not a ranking.

## Data sources and licenses

Both datasets ship inside scikit-learn (`sklearn.datasets`), so nothing is
downloaded:

* **Breast Cancer Wisconsin (Diagnostic)** — Wolberg, Mangasarian, Street &
  Street, UCI Machine Learning Repository,
  <https://doi.org/10.24432/C5DW2B>, license **CC BY 4.0**.
* **Optical Recognition of Handwritten Digits** (8×8 version) — Alpaydin &
  Kaynak, UCI Machine Learning Repository,
  <https://doi.org/10.24432/C50P49>, license **CC BY 4.0**.

`examples/results/demo_predictions_digits_gnb.csv` contains model
predictions derived from the digits data (no raw images).

## Limitations (honest)

* ECE depends on the number and type of bins and is noisy on small test sets;
  the CLI also reports equal-mass ECE (`ece_quantile`) so you can compare.
  ECE/MCE use top-label (confidence) calibration, not full class-wise calibration.
* Calibrators must be fitted on data the model was **not** trained on.
  The CLI's `--calibrate` mode does a random split of the given file; if your
  rows are time-ordered, split by time yourself and use the Python API.
* `brier_decomposition` is binary-only and binned, so it reports a small
  `within_bin` residual to keep the identity exact.
* Benchmark covers two small, clean, tabular/image datasets; results on large
  or shifted data can differ. No claim is made beyond what is in the table.

## Tests

```bash
python -m pytest -q tests
```

15 tests: hand-computed ECE/Brier values, the Murphy identity, synthetic
over-confident data where temperature scaling must recover T ≈ 3, Platt and
isotonic on a skewed binary problem, input validation, and CLI end-to-end runs.
(The synthetic data is used only inside tests.)

## References

* Guo, Pleiss, Sun & Weinberger (2017). *On Calibration of Modern Neural Networks.* ICML.
* Platt (1999). *Probabilistic outputs for support vector machines.*
* Zadrozny & Elkan (2002). *Transforming classifier scores into accurate multiclass probability estimates.* KDD.
* Murphy (1973). *A new vector partition of the probability score.* J. Appl. Meteor.

## ملخص بالعربي

**calibcheck** أداة بايثون صغيرة تقيس هل احتمالات نموذج التصنيف موثوقة: لما
النموذج يقول "متأكد ٩٠٪" هل فعلاً يصيب ٩٠٪ من المرات؟ تحسب مقاييس المعايرة
(ECE وBrier وlog-loss)، ترسم مخطط الموثوقية، وتصلح الاحتمالات بعد التدريب
(Temperature / Platt / Isotonic) بدون إعادة تدريب النموذج. على بيانات digits
الحقيقية خفّضت خطأ المعايرة لنموذج Random Forest من 0.223 إلى 0.015 بدون
ما تتغير الدقة.
