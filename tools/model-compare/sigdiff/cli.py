"""Command line interface.

Examples
--------
Compare two prediction columns of a CSV::

    python -m sigdiff compare preds.csv --target y --a logreg --b gbm --metric auc

Compare several models against a baseline (Holm-corrected)::

    python -m sigdiff compare preds.csv --target y --a logreg --b gbm,rf,knn --metric accuracy

Compare cross-validation fold scores (one row per fold)::

    python -m sigdiff cv folds.csv --a logreg --b gbm --n-train 900 --n-test 100
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

from .compare import compare, compare_many
from .cv import corrected_resampled_ttest, five_by_two_cv_ttest, naive_ttest
from .metrics import METRICS


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sigdiff", description="Is model B really better than model A?")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compare", help="compare predictions on the same test set")
    c.add_argument("csv")
    c.add_argument("--target", required=True, help="column with true labels / values")
    c.add_argument("--a", required=True, help="baseline prediction column")
    c.add_argument("--b", required=True, help="challenger column(s), comma-separated")
    c.add_argument("--metric", default="accuracy", choices=sorted(METRICS))
    c.add_argument("--method", default="bootstrap", choices=["bootstrap", "permutation", "mcnemar"])
    c.add_argument("--groups", default=None, help="optional column of cluster ids (resample whole clusters)")
    c.add_argument("--alpha", type=float, default=0.05)
    c.add_argument("--n-resamples", type=int, default=2000)
    c.add_argument("--seed", type=int, default=0)
    c.add_argument("--json", action="store_true", help="print JSON instead of text")

    v = sub.add_parser("cv", help="compare per-fold cross-validation scores")
    v.add_argument("csv", help="one row per fold, same fold order for both models")
    v.add_argument("--a", required=True)
    v.add_argument("--b", required=True)
    v.add_argument("--n-train", type=int, help="rows in one training fold (corrected t-test)")
    v.add_argument("--n-test", type=int, help="rows in one test fold (corrected t-test)")
    v.add_argument("--five-by-two", action="store_true", help="the 10 rows are a 5x2cv run (rep-major order)")
    v.add_argument("--alpha", type=float, default=0.05)
    v.add_argument("--json", action="store_true")
    return p


def _cmd_compare(args) -> int:
    df = pd.read_csv(args.csv)
    challengers = [s.strip() for s in args.b.split(",") if s.strip()]
    missing = [c for c in [args.target, args.a, *challengers] + ([args.groups] if args.groups else []) if c not in df]
    if missing:
        print(f"error: columns not found: {missing}", file=sys.stderr)
        return 2
    y = df[args.target].to_numpy()
    groups = df[args.groups].to_numpy() if args.groups else None
    kw = dict(method=args.method, alpha=args.alpha, n_resamples=args.n_resamples, groups=groups, seed=args.seed)
    if len(challengers) == 1:
        res = [compare(y, df[args.a].to_numpy(), df[challengers[0]].to_numpy(), args.metric,
                       name_a=args.a, name_b=challengers[0], **kw)]
    else:
        preds = {args.a: df[args.a].to_numpy(), **{c: df[c].to_numpy() for c in challengers}}
        res = compare_many(y, preds, args.a, args.metric, **kw)
    if args.json:
        print(json.dumps([r.to_dict() for r in res], indent=2))
    else:
        print("\n\n".join(r.summary() for r in res))
    return 0


def _cmd_cv(args) -> int:
    df = pd.read_csv(args.csv)
    a, b = df[args.a].to_numpy(dtype=float), df[args.b].to_numpy(dtype=float)
    out = {"naive_ttest": naive_ttest(a, b, args.alpha)}
    if args.five_by_two:
        out["five_by_two_cv_ttest"] = five_by_two_cv_ttest(a.reshape(5, 2), b.reshape(5, 2), args.alpha)
    if args.n_train and args.n_test:
        out["corrected_resampled_ttest"] = corrected_resampled_ttest(a, b, args.n_train, args.n_test, args.alpha)
    if args.json:
        print(json.dumps({k: vars(r) for k, r in out.items()}, indent=2))
    else:
        print(f"mean fold difference ({args.b} - {args.a}): {np.mean(b - a):+.5f} over {a.size} folds")
        for k, r in out.items():
            note = "  (over-confident, reference only)" if k == "naive_ttest" else ""
            print(f"  {k:28s} p={r.p_value:.4g}  CI [{r.ci_low:+.5f}, {r.ci_high:+.5f}]{note}")
        if "corrected_resampled_ttest" not in out and not args.five_by_two:
            print("  tip: pass --n-train/--n-test for the corrected t-test")
    return 0


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    return _cmd_compare(args) if args.cmd == "compare" else _cmd_cv(args)


if __name__ == "__main__":
    raise SystemExit(main())
