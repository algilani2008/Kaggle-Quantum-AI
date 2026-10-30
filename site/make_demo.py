"""Regenerates the demo run shown on the site.

Creates a synthetic binary-classification dataset, runs ../pipeline.py on it, and
writes demo-report.js from the resulting report.json.

    pip install -r ../requirements.txt
    python site/make_demo.py
"""
import json, pathlib, subprocess, sys, tempfile
import numpy as np, pandas as pd
from sklearn.datasets import make_classification

HERE = pathlib.Path(__file__).resolve().parent
with tempfile.TemporaryDirectory() as tmp:
    tmp = pathlib.Path(tmp)
    (tmp / "data").mkdir()
    X, y = make_classification(n_samples=8000, n_features=14, n_informative=8, n_redundant=3,
                               flip_y=0.03, class_sep=0.9, random_state=7)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    rng = np.random.default_rng(7)
    df["region"] = rng.choice(["north", "south", "east", "west"], len(df))
    df["plan"] = np.where(df["f0"] > 0.3, "premium", rng.choice(["basic", "standard"], len(df)))
    df.insert(0, "id", range(1, len(df) + 1))
    df["target"] = y
    train, test = df.iloc[:6000], df.iloc[6000:].drop(columns="target")
    train.to_csv(tmp / "data/train.csv", index=False)
    test.to_csv(tmp / "data/test.csv", index=False)
    pd.DataFrame({"id": test["id"], "target": 0.5}).to_csv(tmp / "data/sample_submission.csv", index=False)
    subprocess.run([sys.executable, str(HERE.parent / "pipeline.py"), "--data-dir", str(tmp / "data"),
                    "--out-dir", str(tmp / "output"), "--state-file", str(tmp / "state.json")], check=True)
    report = json.loads((tmp / "output/report.json").read_text())
report["demo_dataset"] = "synthetic: sklearn make_classification, 6,000 train / 2,000 test rows, 14 numeric + 2 categorical columns"
(HERE / "demo-report.js").write_text("window.DEMO_REPORT = " + json.dumps(report, indent=1) + ";\n")
print("wrote demo-report.js")
