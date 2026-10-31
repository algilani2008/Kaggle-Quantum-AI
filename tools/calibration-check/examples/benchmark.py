"""Benchmark: how well-calibrated are common scikit-learn classifiers, and
how much do post-hoc calibrators help?

Real, openly licensed datasets bundled with scikit-learn (no download):
  * breast_cancer  - UCI Breast Cancer Wisconsin (Diagnostic), binary, 569 rows
  * digits         - UCI Optical Recognition of Handwritten Digits, 10 classes, 1797 rows

Each dataset is split once per seed into train / calibration / test
(60 / 20 / 20, stratified). Models are trained on train, calibrators are
fitted on the calibration split, and every number is measured on test.
Results are averaged over 5 seeds.

Run:  python examples/benchmark.py  (writes examples/results/)
"""
from __future__ import annotations

import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from calibcheck import CALIBRATORS, calibration_report  # noqa: E402
from calibcheck.plot import reliability_diagram  # noqa: E402

OUT = os.path.join(HERE, "results")
SEEDS = [0, 1, 2, 3, 4]
N_BINS = 10  # small test sets (114 / 360 rows): keep bins coarse


def models(seed):
    return {
        "GaussianNB": GaussianNB(),
        "LogisticRegression": make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=-1),
        "MLP": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), max_iter=600,
                                                             random_state=seed)),
    }


def run():
    os.makedirs(OUT, exist_ok=True)
    warnings.filterwarnings("ignore")
    datasets = {"breast_cancer": load_breast_cancer(return_X_y=True),
                "digits": load_digits(return_X_y=True)}
    rows = []
    plot_cache = {}
    for dname, (X, y) in datasets.items():
        binary = len(np.unique(y)) == 2
        for seed in SEEDS:
            Xtr, Xtmp, ytr, ytmp = train_test_split(X, y, test_size=0.4, stratify=y, random_state=seed)
            Xcal, Xte, ycal, yte = train_test_split(Xtmp, ytmp, test_size=0.5, stratify=ytmp,
                                                    random_state=seed)
            for mname, model in models(seed).items():
                model.fit(Xtr, ytr)
                pcal, pte = model.predict_proba(Xcal), model.predict_proba(Xte)
                if binary:
                    pcal, pte = pcal[:, 1], pte[:, 1]
                variants = {"uncalibrated": pte}
                for cname, C in CALIBRATORS.items():
                    if cname == "platt" and not binary:
                        continue
                    variants[cname] = C().fit(ycal, pcal).transform(pte)
                for vname, p in variants.items():
                    r = calibration_report(yte, p, N_BINS).to_dict()
                    rows.append({"dataset": dname, "model": mname, "calibrator": vname, "seed": seed,
                                 **{k: r[k] for k in ("accuracy", "ece", "mce", "brier", "log_loss",
                                                      "mean_confidence")}})
                if seed == 0:
                    plot_cache[(dname, mname)] = (yte, variants)
                if seed == 0 and dname == "digits" and mname == "GaussianNB":
                    # combined cal+test predictions, used by the CLI demo in the README
                    demo = pd.DataFrame(np.vstack([model.predict_proba(Xcal), model.predict_proba(Xte)]).round(6),
                                        columns=[f"p{k}" for k in range(10)])
                    demo["label"] = np.concatenate([ycal, yte])
                    demo.to_csv(os.path.join(OUT, "demo_predictions_digits_gnb.csv"), index=False)

    raw = pd.DataFrame(rows)
    raw.to_csv(os.path.join(OUT, "benchmark_raw.csv"), index=False)
    summary = (raw.groupby(["dataset", "model", "calibrator"], sort=False)
               [["accuracy", "ece", "brier", "log_loss", "mean_confidence"]]
               .agg(["mean", "std"]).round(4))
    summary.columns = [f"{a}_{b}" for a, b in summary.columns]
    summary = summary.reset_index()
    summary.to_csv(os.path.join(OUT, "benchmark_summary.csv"), index=False)

    # Markdown table (means only) for the README
    md = ["| dataset | model | calibrator | accuracy | ECE | Brier | NLL |", "|---|---|---|---|---|---|---|"]
    for _, r in summary.iterrows():
        md.append(f"| {r.dataset} | {r.model} | {r.calibrator} | {r.accuracy_mean:.3f} | "
                  f"{r.ece_mean:.3f} ± {r.ece_std:.3f} | {r.brier_mean:.3f} | {r.log_loss_mean:.3f} |")
    with open(os.path.join(OUT, "benchmark_table.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")

    for key in [("digits", "GaussianNB"), ("digits", "RandomForest"), ("breast_cancer", "RandomForest")]:
        yte, variants = plot_cache[key]
        curves = [(v, yte, p) for v, p in variants.items() if v in ("uncalibrated", "temperature", "isotonic")]
        reliability_diagram(curves, os.path.join(OUT, f"reliability_{key[0]}_{key[1]}.png"), N_BINS,
                            title=f"{key[1]} on {key[0]} (test split, seed 0)")
    print("\n".join(md))


if __name__ == "__main__":
    run()
