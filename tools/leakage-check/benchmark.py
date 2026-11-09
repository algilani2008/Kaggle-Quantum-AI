"""Benchmark: does leakcheck catch known leaks, and stay quiet on clean splits?

Uses four real datasets that ship with scikit-learn (no download needed):
breast_cancer, wine, digits (classification) and diabetes (regression).

For each dataset and random seed we build five train/test pairs:

  clean          a plain random 75/25 split (no leak). Any medium/high finding is a false alarm.
  target_proxy   INJECTED LEAK: a column copied from the target with 3% of values corrupted
                 (classification) or target + small noise (regression).
  sorted_id      INJECTED LEAK: rows sorted by target, then a 'row_id' column 0..n-1 added.
  test_copies    INJECTED LEAK: 10% of train rows copied into the test set.
  shifted_split  NON-RANDOM SPLIT: test = the 25% of rows with the largest value of one feature.

The leaks are deliberately injected into real data so we know the ground truth; they are
not properties of the original datasets. Detection = the expected check reports
severity 'medium' or 'high'.

Thresholds were tuned on seeds 0-9 during development; the reported run uses seeds
100-109 so the numbers are not measured on the same splits used for tuning.

Run:  python benchmark.py            (writes results/benchmark.csv, .md, .png)
"""
from __future__ import annotations

import os
import time

import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split

from leakcheck import audit

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")

DATASETS = {
    "breast_cancer": datasets.load_breast_cancer,
    "wine": datasets.load_wine,
    "digits": datasets.load_digits,
    "diabetes": datasets.load_diabetes,
}
EXPECTED = {
    "target_proxy": "target_leakage",
    "sorted_id": "id_like_column",
    "test_copies": "cross_duplicates",
    "shifted_split": "adversarial_validation",
}
SCENARIOS = ["clean", *EXPECTED]


def load(name: str) -> pd.DataFrame:
    df = DATASETS[name](as_frame=True).frame
    df.columns = [str(c) for c in df.columns]
    return df


def make_pair(df: pd.DataFrame, scenario: str, seed: int, task: str):
    rng = np.random.RandomState(seed)
    df = df.copy()
    y = df["target"].to_numpy()
    if scenario == "target_proxy":
        if task == "classification":
            leak = y.copy()
            flip = rng.rand(len(y)) < 0.03
            leak[flip] = rng.choice(np.unique(y), flip.sum())
        else:
            leak = y + rng.normal(0, 0.1 * y.std(), len(y))
        df.insert(0, "leak", leak)
    if scenario == "sorted_id":
        df = df.iloc[np.lexsort((rng.rand(len(df)), y))].reset_index(drop=True)
        df.insert(0, "row_id", np.arange(len(df)))
    if scenario == "shifted_split":
        # a random feature column with enough distinct values for a real shift
        cands = [c for c in df.columns if c != "target" and df[c].nunique() > 10]
        col = cands[rng.randint(len(cands))]
        cut = df[col].rank(method="first") > 0.75 * len(df)
        return df[~cut], df[cut].drop(columns="target")
    tr, te = train_test_split(df, test_size=0.25, random_state=seed)
    te = te.drop(columns="target")
    if scenario == "test_copies":
        copies = tr.sample(frac=0.10, random_state=seed).drop(columns="target")
        te = pd.concat([te, copies]).sample(frac=1, random_state=seed)
    return tr, te


def main(seeds: int = 10, seed_offset: int = 100):
    os.makedirs(OUT, exist_ok=True)
    rows = []
    t0 = time.time()
    for name in DATASETS:
        df = load(name)
        task = "regression" if name == "diabetes" else "classification"
        for scenario in SCENARIOS:
            for seed in range(seed_offset, seed_offset + seeds):
                tr, te = make_pair(df, scenario, seed, task)
                t = time.time()
                rep = audit(tr, te, target="target", task=task, random_state=seed)
                flagged = {f.check for f in rep.at_least("medium")}
                expected = EXPECTED.get(scenario)
                rows.append({
                    "dataset": name, "scenario": scenario, "seed": seed,
                    "detected": (expected in flagged) if expected else None,
                    "false_alarm": bool(flagged - ({expected} if expected else set())),
                    "flagged": ";".join(sorted(flagged)),
                    "adversarial_auc": round(rep.summary["adversarial_auc"], 4),
                    "seconds": round(time.time() - t, 2),
                })
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT, "benchmark.csv"), index=False)

    agg = (res.groupby(["scenario", "dataset"], sort=False)
              .agg(detected=("detected", lambda s: s.mean() if s.notna().all() else np.nan),
                   false_alarm=("false_alarm", "mean"), runs=("seed", "size"),
                   seconds=("seconds", "mean"))
              .reset_index())
    overall = (res.groupby("scenario", sort=False)
                  .agg(detected=("detected", lambda s: s.mean() if s.notna().all() else np.nan),
                       false_alarm=("false_alarm", "mean"), runs=("seed", "size")).reset_index())

    def pct(v):
        return "-" if pd.isna(v) else f"{v:.0%}"

    md = ["# leakcheck benchmark", "",
          f"{len(res)} audits on 4 scikit-learn datasets x 5 scenarios x {seeds} seeds, "
          f"total {time.time() - t0:.0f} s on CPU. Seeds {seed_offset}-{seed_offset + seeds - 1}.", "",
          "Thresholds were tuned while developing on seeds 0-9; the numbers below use fresh seeds",
          "that were not looked at during tuning.", "",
          "Detection rate = share of runs where the expected check reported medium or high.",
          "False-alarm rate = share of runs with a medium/high finding from any *other* check.", "",
          "## Overall", "", "| Scenario | Detection | False alarms | Runs |", "|---|---|---|---|"]
    md += [f"| {r.scenario} | {pct(r.detected)} | {pct(r.false_alarm)} | {r.runs} |"
           for r in overall.itertuples()]
    md += ["", "## By dataset", "", "| Scenario | Dataset | Detection | False alarms | Mean s/audit |",
           "|---|---|---|---|---|"]
    md += [f"| {r.scenario} | {r.dataset} | {pct(r.detected)} | {pct(r.false_alarm)} | {r.seconds:.2f} |"
           for r in agg.itertuples()]
    misses = res[(res.detected == False) | res.false_alarm]  # noqa: E712
    md += ["", "## Misses and false alarms", ""]
    if misses.empty:
        md.append("None.")
    else:
        md += ["| Dataset | Scenario | Seed | Flagged (medium+) | Adv. AUC |", "|---|---|---|---|---|"]
        md += [f"| {r.dataset} | {r.scenario} | {r.seed} | {r.flagged or '-'} | {r.adversarial_auc} |"
               for r in misses.itertuples()]
    with open(os.path.join(OUT, "benchmark.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print("\n".join(md))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.5, 3.6), dpi=150)
    labels = list(overall.scenario)
    vals = [(r.false_alarm if r.scenario == "clean" else r.detected) for r in overall.itertuples()]
    colors = ["#9aa0a6" if s == "clean" else "#2a6fdb" for s in labels]
    bars = ax.bar(labels, [v * 100 for v in vals], color=colors, width=0.6)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 100 + 2, f"{v:.0%}", ha="center", fontsize=9)
    ax.set_ylim(0, 112)
    ax.set_ylabel("% of runs")
    ax.set_title("leakcheck: detection rate per injected leak (blue)\nand false-alarm rate on clean splits (grey)",
                 fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "benchmark.png"))
    return res


if __name__ == "__main__":
    main()
