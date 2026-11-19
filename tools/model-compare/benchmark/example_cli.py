"""Make a small example predictions file and show the CLI on it.

Writes results/example_predictions.csv (1,000 Adult test rows with three
models' predicted probabilities) and results/example_cli_output.txt.

Run:  python benchmark/example_cli.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
from run_benchmark import load_adult  # noqa: E402


def main():
    X, y = load_adult()
    perm = np.random.default_rng(0).permutation(y.size)
    train, test = perm[:8000], perm[18000:19000]
    sc = StandardScaler().fit(X[train])
    models = {
        "logreg": LogisticRegression(max_iter=3000),
        "tree": DecisionTreeClassifier(max_depth=3, random_state=0),
        "gbm": HistGradientBoostingClassifier(max_iter=60, max_leaf_nodes=8, learning_rate=0.1, random_state=0),
    }
    df = pd.DataFrame({"y": y[test]})
    for k, m in models.items():
        m.fit(sc.transform(X[train]), y[train])
        df[k] = np.round(m.predict_proba(sc.transform(X[test]))[:, 1], 5)
    out = ROOT / "results"
    out.mkdir(exist_ok=True)
    csv = out / "example_predictions.csv"
    df.to_csv(csv, index=False)

    cmds = [
        ["compare", str(csv), "--target", "y", "--a", "logreg", "--b", "gbm", "--metric", "accuracy", "--method", "mcnemar"],
        ["compare", str(csv), "--target", "y", "--a", "logreg", "--b", "gbm", "--metric", "auc"],
        ["compare", str(csv), "--target", "y", "--a", "logreg", "--b", "gbm,tree", "--metric", "logloss"],
    ]
    lines = []
    for c in cmds:
        r = subprocess.run([sys.executable, "-m", "sigdiff", *c], cwd=ROOT, capture_output=True, text=True, check=True)
        shown = ["python", "-m", "sigdiff", *c]
        shown[4] = "results/example_predictions.csv"
        lines += ["$ " + " ".join(shown), r.stdout.rstrip(), ""]
    (out / "example_cli_output.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
