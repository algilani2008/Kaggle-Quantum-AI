# Leakage audit: Titanic (random 75/25 split, no injected leak)

Train rows: 981, test rows: 328, features checked: 3, task: classification.

| Severity | Count |
|---|---|
| high | 0 |
| medium | 1 |
| low | 1 |
| info | 2 |

- **[MEDIUM] cross_duplicates**: 286 of 328 test rows (87.2%) have exact copies in train. Only 31% of train rows are distinct, so repeats are expected for this data; this is a weaker signal.
- **[LOW] train_duplicates**: 69.0% of train rows repeat an earlier row on all features. With so few distinct rows this is probably natural to the data, not a pipeline bug.
- **[INFO] conflicting_labels**: 68 groups of identical feature rows carry different targets (label noise or missing features).
- **[INFO] adversarial_validation**: A model separates train from test rows with AUC 0.513 (0.5 = indistinguishable, z = 0.7). Most shifted columns: age (0.51), passengerClass (0.50), sex (0.48).

Single-column out-of-fold AUC against `survived`:

- sex: 0.736
- age: 0.593
- passengerClass: 0.668
