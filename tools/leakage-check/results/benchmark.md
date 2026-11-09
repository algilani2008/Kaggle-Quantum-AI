# leakcheck benchmark

200 audits on 4 scikit-learn datasets x 5 scenarios x 10 seeds, total 210 s on CPU. Seeds 100-109.

Thresholds were tuned while developing on seeds 0-9; the numbers below use fresh seeds
that were not looked at during tuning.

Detection rate = share of runs where the expected check reported medium or high.
False-alarm rate = share of runs with a medium/high finding from any *other* check.

## Overall

| Scenario | Detection | False alarms | Runs |
|---|---|---|---|
| clean | - | 0% | 40 |
| target_proxy | 100% | 0% | 40 |
| sorted_id | 100% | 0% | 40 |
| test_copies | 100% | 0% | 40 |
| shifted_split | 100% | 0% | 40 |

## By dataset

| Scenario | Dataset | Detection | False alarms | Mean s/audit |
|---|---|---|---|---|
| clean | breast_cancer | - | 0% | 0.98 |
| target_proxy | breast_cancer | 100% | 0% | 0.95 |
| sorted_id | breast_cancer | 100% | 0% | 1.02 |
| test_copies | breast_cancer | 100% | 0% | 1.05 |
| shifted_split | breast_cancer | 100% | 0% | 0.65 |
| clean | wine | - | 0% | 0.36 |
| target_proxy | wine | 100% | 0% | 0.37 |
| sorted_id | wine | 100% | 0% | 0.38 |
| test_copies | wine | 100% | 0% | 0.38 |
| shifted_split | wine | 100% | 0% | 0.33 |
| clean | digits | - | 0% | 2.52 |
| target_proxy | digits | 100% | 0% | 2.56 |
| sorted_id | digits | 100% | 0% | 2.57 |
| test_copies | digits | 100% | 0% | 2.54 |
| shifted_split | digits | 100% | 0% | 2.21 |
| clean | diabetes | - | 0% | 0.42 |
| target_proxy | diabetes | 100% | 0% | 0.47 |
| sorted_id | diabetes | 100% | 0% | 0.44 |
| test_copies | diabetes | 100% | 0% | 0.44 |
| shifted_split | diabetes | 100% | 0% | 0.33 |

## Misses and false alarms

None.
