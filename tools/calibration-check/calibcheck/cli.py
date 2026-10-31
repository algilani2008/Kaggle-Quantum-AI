"""Command-line interface.

Example
-------
python -m calibcheck preds.csv --label-col label --prob-cols p0,p1,p2 \
    --out-dir report --calibrate temperature
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

from .calibrators import CALIBRATORS
from .metrics import calibration_report, reliability_bins, brier_decomposition
from .plot import reliability_diagram


def _load(path, label_col, prob_cols):
    df = pd.read_csv(path)
    if label_col not in df.columns:
        raise SystemExit(f"label column '{label_col}' not found; columns: {list(df.columns)}")
    if prob_cols:
        cols = [c.strip() for c in prob_cols.split(",") if c.strip()]
    else:
        cols = [c for c in df.columns if c != label_col and c.lower().startswith("p")]
    missing = [c for c in cols if c not in df.columns]
    if missing or not cols:
        raise SystemExit(f"probability columns missing or empty: {missing or cols}")
    y = df[label_col].to_numpy()
    p = df[cols].to_numpy(dtype=float)
    if p.shape[1] == 1:
        p = p[:, 0]
    return y, p


def main(argv=None):
    ap = argparse.ArgumentParser(prog="calibcheck", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("csv", help="CSV with one label column and probability column(s)")
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--prob-cols", default=None,
                    help="comma-separated probability columns (one column = binary p(class 1)); "
                         "default: every column starting with 'p'")
    ap.add_argument("--bins", type=int, default=15)
    ap.add_argument("--out-dir", default="calibcheck_report")
    ap.add_argument("--calibrate", choices=sorted(CALIBRATORS), default=None,
                    help="fit a calibrator on a random part of the file and evaluate on the rest")
    ap.add_argument("--cal-frac", type=float, default=0.5,
                    help="fraction of rows used to fit the calibrator (default 0.5)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    y, p = _load(a.csv, a.label_col, a.prob_cols)
    os.makedirs(a.out_dir, exist_ok=True)
    out = {"input": os.path.basename(a.csv)}
    curves = []

    if a.calibrate:
        if not 0.05 <= a.cal_frac <= 0.95:
            raise SystemExit("--cal-frac must be between 0.05 and 0.95")
        rng = np.random.default_rng(a.seed)
        idx = rng.permutation(len(y))
        n_cal = int(round(a.cal_frac * len(y)))
        cal, ev = idx[:n_cal], idx[n_cal:]
        cal_model = CALIBRATORS[a.calibrate]().fit(y[cal], p[cal])
        p_after = cal_model.transform(p[ev])
        out["evaluated_on"] = f"{len(ev)} held-out rows ({len(cal)} rows used to fit '{a.calibrate}')"
        out["before"] = calibration_report(y[ev], p[ev], a.bins).to_dict()
        out["after"] = calibration_report(y[ev], p_after, a.bins).to_dict()
        if hasattr(cal_model, "temperature_"):
            out["temperature"] = cal_model.temperature_
        curves = [("before", y[ev], p[ev]), (f"after {a.calibrate}", y[ev], p_after)]
        bins_rows = reliability_bins(y[ev], p_after, a.bins).to_rows()
    else:
        out["before"] = calibration_report(y, p, a.bins).to_dict()
        curves = [("model", y, p)]
        bins_rows = reliability_bins(y, p, a.bins).to_rows()
        if np.asarray(p).ndim == 1 or np.asarray(p).shape[1] == 2:
            out["brier_decomposition"] = brier_decomposition(y, p, min(a.bins, 10))

    pd.DataFrame(bins_rows).to_csv(os.path.join(a.out_dir, "bins.csv"), index=False)
    with open(os.path.join(a.out_dir, "report.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    reliability_diagram(curves, os.path.join(a.out_dir, "reliability.png"), a.bins)

    b = out["before"]
    print(f"rows={b['n_samples']} classes={b['n_classes']} accuracy={b['accuracy']:.4f}")
    print(f"ECE={b['ece']:.4f}  MCE={b['mce']:.4f}  Brier={b['brier']:.4f}  NLL={b['log_loss']:.4f}")
    if "after" in out:
        c = out["after"]
        print(f"after {a.calibrate}: ECE={c['ece']:.4f}  MCE={c['mce']:.4f}  "
              f"Brier={c['brier']:.4f}  NLL={c['log_loss']:.4f}")
    print(f"wrote {a.out_dir}/report.json, bins.csv, reliability.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
