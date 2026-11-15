"""Train, tune and evaluate CO2 forecasters on the Mauna Loa monthly record.

    python train.py                    # downloads data, writes results/
    python train.py --data file.csv    # use a local copy of the CSV

Protocol (all splits are time-ordered, nothing is shuffled):
  train      : 1958-03 .. 2019-12   models are fitted here
  validation : 2020-01 .. 2022-12   with train, used for time-series cross-validation:
               59 forecast origins (every 6 months, 1990-12 .. 2019-12), each forecasting
               36 months ahead; hyper-parameters minimise the mean error over all origins
  test       : 2023-01 .. latest    touched once, after refitting on train+validation
Then the chosen model is refitted on everything and forecasts the next 24 months,
with 90% intervals built from its own past forecast errors (rolling backtest).
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from co2fc import data
from co2fc.evaluate import (by_horizon_year, coverage, forecast_from, interval_offsets,
                            rolling_errors, scores, split, tune_rolling)

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN_END, VAL_END = "2019-12", "2022-12"
CV_H = 36
SPACES = {
    "seasonal_naive": {},
    "seasonal_naive_drift": {"drift_years": [1, 3, 5, 10]},
    "harmonic_trend": {"degree": [1, 2, 3], "harmonics": [1, 2, 3, 4],
                       "window_years": [10, 15, 20, 30, None], "anchor_months": [0, 6, 12, 24]},
}
LEARNED = "harmonic_trend"
FUTURE_H = 24
BACKTEST_FROM = "1990-12"


def main(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", help="local CSV (same format as the GitHub mirror)")
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--no-plots", action="store_true")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    df = data.load(args.data, cache=os.path.join(HERE, ".cache", "co2-mm-mlo.csv"))
    y = df["co2"]
    train, val, test = split(y, TRAIN_END, VAL_END)
    trainval = pd.concat([train, val])
    print(f"data: {y.index[0]} .. {y.index[-1]} ({len(y)} months); "
          f"train {len(train)}, val {len(val)}, test {len(test)}")

    # 1) tune every model by rolling-origin cross-validation on train+val
    chosen, tuning_rows, backtest = {}, [], {}
    for name, space in SPACES.items():
        params, rows = tune_rolling(trainval, name, space, CV_H, BACKTEST_FROM)
        chosen[name] = params
        backtest[name] = min(r["cv_mae"] for r in rows)
        tuning_rows += [{"model": name, **r} for r in rows]
        print(f"tuned {name:22s} -> {params}  (CV MAE {backtest[name]:.3f}, "
              f"{rows[0]['origins']} origins)")
    pd.DataFrame(tuning_rows).to_csv(os.path.join(args.out, "tuning.csv"), index=False)

    # 2) test: refit on train+val, forecast the whole test window once
    test_pred = {n: forecast_from(trainval, n, p, len(test)) for n, p in chosen.items()}
    test_scores = {n: scores(test.values, p) for n, p in test_pred.items()}
    val_scores = {n: scores(val.values, forecast_from(train, n, chosen[n], len(val)))
                  for n in chosen}

    # 3) 90% intervals for the learned model from its own past errors (train+val only)
    errs = rolling_errors(trainval, LEARNED, chosen[LEARNED], len(test), BACKTEST_FROM)
    lo_off, hi_off = interval_offsets(errs, 0.9)
    lo, hi = test_pred[LEARNED] + lo_off, test_pred[LEARNED] + hi_off
    test_cov = coverage(test.values, lo, hi)

    tp = pd.DataFrame({"month": test.index.astype(str), "actual": test.values,
                       **{n: np.round(p, 3) for n, p in test_pred.items()},
                       f"{LEARNED}_lo90": np.round(lo, 3), f"{LEARNED}_hi90": np.round(hi, 3)})
    tp.to_csv(os.path.join(args.out, "test_predictions.csv"), index=False)

    # 4) final model on all data, next 24 months
    future = forecast_from(y, LEARNED, chosen[LEARNED], FUTURE_H)
    ferrs = rolling_errors(y, LEARNED, chosen[LEARNED], FUTURE_H, BACKTEST_FROM)
    flo, fhi = interval_offsets(ferrs, 0.9)
    fidx = pd.period_range(y.index[-1] + 1, periods=FUTURE_H, freq="M")
    fc = pd.DataFrame({"month": fidx.astype(str), "forecast": np.round(future, 2),
                       "lo90": np.round(future + flo, 2), "hi90": np.round(future + fhi, 2)})
    fc.to_csv(os.path.join(args.out, "forecast_next24.csv"), index=False)

    best_base = min((n for n in chosen if n != LEARNED), key=lambda n: test_scores[n]["mae"])
    gain = 1 - test_scores[LEARNED]["mae"] / test_scores[best_base]["mae"]
    peak = fc.loc[fc["forecast"].idxmax()]
    report = {
        "data": {"first": str(y.index[0]), "last": str(y.index[-1]), "months": len(y),
                 "noaa_interpolated_months": int(df["interpolated"].sum()),
                 "source": data.DATA_URL},
        "split": {"train": [str(train.index[0]), str(train.index[-1])],
                  "validation": [str(val.index[0]), str(val.index[-1])],
                  "test": [str(test.index[0]), str(test.index[-1])]},
        "chosen_params": chosen,
        "validation": val_scores,
        "test": test_scores,
        "cv_mae_36m": backtest,
        "test_mae_by_horizon_year": {n: by_horizon_year(test.values, p) for n, p in test_pred.items()},
        "best_baseline": best_base,
        "learned_vs_best_baseline_mae_reduction_pct": round(100 * gain, 1),
        "test_interval90_coverage": round(test_cov, 3),
        "backtest_origins": {"test_horizon": int(errs.shape[0]), "future_horizon": int(ferrs.shape[0])},
        "forecast_peak": {"month": peak["month"], "ppm": float(peak["forecast"]),
                          "lo90": float(peak["lo90"]), "hi90": float(peak["hi90"])},
    }
    with open(os.path.join(args.out, "metrics.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    print("\nTEST (2023-01 onward, one forecast from 2022-12):")
    for n, s in test_scores.items():
        print(f"  {n:22s} MAE {s['mae']:.3f} ppm  RMSE {s['rmse']:.3f}  MAPE {s['mape_pct']:.3f}%")
    print("CROSS-VALIDATION (train+val, 36-month horizon), MAE of the tuned models:")
    for n, v in backtest.items():
        print(f"  {n:22s} {v:.3f} ppm")
    print(f"  learned model vs best baseline ({best_base}): {100 * gain:+.1f}% lower MAE")
    print(f"  90% interval coverage on test: {test_cov:.1%}")
    print(f"  forecast peak in next 24 months: {peak['month']} {peak['forecast']} ppm "
          f"[{peak['lo90']}, {peak['hi90']}]")

    if not args.no_plots:
        from plots import make_plots
        make_plots(y, tp, fc, report, args.out)
    return report


if __name__ == "__main__":
    main()
