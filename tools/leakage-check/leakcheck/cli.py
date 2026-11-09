"""Command line interface: python -m leakcheck train.csv --test test.csv --target y"""
from __future__ import annotations

import argparse
import sys

import pandas as pd

from .core import SEVERITIES, audit


def _read(path: str) -> pd.DataFrame:
    if path.endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="leakcheck",
                                description="Audit a train/test pair for data leakage.")
    p.add_argument("train", help="train CSV or Parquet file")
    p.add_argument("--test", help="test CSV or Parquet file (target column optional)")
    p.add_argument("--target", help="label column in train")
    p.add_argument("--task", default="auto", choices=["auto", "classification", "regression"])
    p.add_argument("--group", help="group id column (e.g. patient, user, store)")
    p.add_argument("--time", help="timestamp column")
    p.add_argument("--ignore", nargs="*", default=[], help="columns to skip")
    p.add_argument("--max-rows", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--md", help="write a Markdown report here")
    p.add_argument("--json", help="write a JSON report here")
    p.add_argument("--fail-on", choices=SEVERITIES[:3],
                   help="exit with code 1 if any finding is at least this severe (for CI)")
    a = p.parse_args(argv)

    train = _read(a.train)
    test = _read(a.test) if a.test else None
    rep = audit(train, test, target=a.target, task=a.task, group=a.group, time=a.time,
                ignore=a.ignore, max_rows=a.max_rows, random_state=a.seed)
    md = rep.to_markdown()
    print(md)
    if a.md:
        with open(a.md, "w", encoding="utf-8") as f:
            f.write(md)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            f.write(rep.to_json())
    if a.fail_on and rep.at_least(a.fail_on):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
