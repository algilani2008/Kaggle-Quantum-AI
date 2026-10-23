#!/usr/bin/env python3
"""Run the shared stacking pipeline for this competition.

Reads settings from config.json next to this file, then calls the repo-level
pipeline.py with this competition's data, output and state paths.

Usage (from the repo root or this folder):
    python competitions/<slug>/run.py
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


def main() -> int:
    with open(os.path.join(HERE, "config.json")) as f:
        cfg = json.load(f)
    if not cfg.get("tabular", False):
        print(f"{cfg.get('slug')}: not marked as tabular in config.json; "
              "the shared stacking pipeline is not applicable. Skipping.")
        return 0
    cmd = [
        sys.executable, os.path.join(ROOT, "pipeline.py"),
        "--data-dir", os.path.join(HERE, "data"),
        "--out-dir", os.path.join(HERE, "output"),
        "--state-file", os.path.join(HERE, "state.json"),
        "--folds", str(cfg.get("folds", 5)),
    ]
    if cfg.get("target"):
        cmd += ["--target", cfg["target"]]
    if cfg.get("id_col"):
        cmd += ["--id-col", cfg["id_col"]]
    print(" ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    sys.exit(main())
